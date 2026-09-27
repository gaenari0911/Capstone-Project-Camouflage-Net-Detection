from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from capstone_lab.config import canonical_json, sha256_file
from capstone_lab.errors import ArtifactError, ConfigError, StateError
from capstone_lab.planning.planner import (
    build_dag,
    build_plan,
    build_registry,
    failure_isolation,
    load_s7_config,
    target_closure,
    validate_dag,
    validate_registry,
)
from capstone_lab.planning.runner import validate_existing_outputs


class S7Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.config_path = cls.root / "configs/campaigns/s7_plan.json"

    def load(self):
        return load_s7_config(self.config_path, self.root)

    def modified(self, callback):
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        callback(raw)
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", dir=self.root, delete=False, encoding="utf-8"
        )
        try:
            json.dump(raw, handle)
            handle.close()
            return load_s7_config(Path(handle.name), self.root)
        finally:
            Path(handle.name).unlink(missing_ok=True)


class S7RegistryTests(S7Fixture):
    def test_exact_core_head_full_and_seed0_costs(self) -> None:
        plan = build_plan(self.load())
        totals = plan["totals"]
        self.assertEqual(plan["status"], "VERIFIED")
        self.assertEqual(totals["registry_rows"], 60)
        self.assertEqual(totals["core_training_jobs"], 42)
        self.assertEqual(totals["H_logical_evaluation_cells"], 18)
        self.assertEqual(totals["H_additional_training_jobs"], 12)
        self.assertEqual(totals["H_reference_rows"], 6)
        self.assertEqual(totals["unique_training_jobs"], 54)
        self.assertEqual(totals["epoch_job_units"], 6120)
        self.assertEqual(totals["seed0_training_jobs"], 18)
        self.assertEqual(totals["seed0_epoch_job_units"], 2040)
        self.assertEqual(totals["planned_unique_generated_images"], 24000)
        self.assertEqual(totals["planned_generated_bundle_files"], 120000)
        self.assertEqual(totals["planned_training_artifact_files"], 324)
        self.assertEqual(totals["planned_total_artifact_files"], 120330)

    def test_H2_hash_mismatch_rejects_reuse_and_adds_six_jobs(self) -> None:
        config = self.modified(
            lambda raw: raw["planning"].update({"h2_reuse_fixture": "FORCED_MISMATCH"})
        )
        plan = build_plan(config)
        totals = plan["totals"]
        self.assertEqual(totals["H_reference_rows"], 0)
        self.assertEqual(totals["H_additional_training_jobs"], 18)
        self.assertEqual(totals["unique_training_jobs"], 60)
        self.assertEqual(totals["epoch_job_units"], 7020)

    def test_invalid_H2_reference_hash_is_never_silently_reused(self) -> None:
        rows = build_registry(self.load())
        index = next(i for i, row in enumerate(rows) if row.node_kind == "REFERENCE")
        rows[index] = replace(rows[index], comparison_contract_sha256="0" * 64)
        with self.assertRaisesRegex(ArtifactError, "reuse hash mismatch"):
            validate_registry(rows)

    def test_duplicate_job_artifact_and_Test_leakage_are_rejected(self) -> None:
        rows = build_registry(self.load())
        training = [row for row in rows if row.node_kind == "TRAIN"]
        altered = list(rows)
        second_index = altered.index(training[1])
        altered[second_index] = replace(training[1], planned_artifact_uri=training[0].planned_artifact_uri)
        with self.assertRaisesRegex(ConfigError, "duplicate training artifact"):
            validate_registry(altered)
        altered = list(rows)
        altered[0] = replace(altered[0], dataset_manifest_identity="sha256:x#split=test")
        with self.assertRaisesRegex(ConfigError, "Test dataset"):
            validate_registry(altered)


class S7DagTests(S7Fixture):
    def test_all_targets_are_deterministic_blocked_and_never_include_Test(self) -> None:
        config = self.load()
        rows = build_registry(config)
        forward = build_dag(rows)
        reverse = build_dag(list(reversed(rows)))
        self.assertEqual(canonical_json(forward), canonical_json(reverse))
        for target in ("ablation", "main", "low", "head_ablation", "core", "all"):
            closure = target_closure(forward, target, config.raw["approvals"])
            self.assertFalse(closure["runnable"])
            self.assertNotIn("gate_final_Test", closure["nodes"])
        with self.assertRaisesRegex(ConfigError, "Test cannot be selected"):
            target_closure(forward, "test", config.raw["approvals"])

    def test_approval_scope_cannot_be_expanded_by_cli_request(self) -> None:
        config = self.load()
        dag = build_dag(build_registry(config))
        with self.assertRaisesRegex(ConfigError, "cannot expand approval scope"):
            target_closure(
                dag,
                "all",
                config.raw["approvals"],
                requested_approvals=("campaign_execution",),
            )

    def test_cycle_and_missing_dependency_are_rejected(self) -> None:
        missing = {"a": {"node_id": "a", "kind": "X", "dependencies": ["absent"], "planned_unique_images": 0, "approval_gate": ""}}
        with self.assertRaisesRegex(ConfigError, "invalid dependencies"):
            validate_dag(missing)
        cycle = {
            "a": {"node_id": "a", "kind": "X", "dependencies": ["b"], "planned_unique_images": 0, "approval_gate": ""},
            "b": {"node_id": "b", "kind": "X", "dependencies": ["a"], "planned_unique_images": 0, "approval_gate": ""},
        }
        with self.assertRaisesRegex(ConfigError, "cycle"):
            validate_dag(cycle)

    def test_failure_blocks_descendants_but_preserves_independent_branches(self) -> None:
        dag = build_dag(build_registry(self.load()))
        result = failure_isolation(dag, "generate_A2_3000")
        self.assertIn("A2_seed0", result["blocked_descendants"])
        self.assertIn("aggregate_A", result["blocked_descendants"])
        self.assertIn("M0_seed0", result["independent"])
        self.assertIn("generate_A3_3000", result["independent"])


class S7SafetyAndResumeTests(S7Fixture):
    def test_config_is_non_runnable_and_protects_prior_artifacts(self) -> None:
        config = self.load()
        self.assertEqual(config.raw["status"], "BLOCKED_APPROVAL")
        self.assertTrue(all(value == "NOT_APPROVED" for value in config.raw["approvals"].values()))
        with self.assertRaisesRegex(ConfigError, "versioned directory"):
            self.modified(lambda raw: raw.update({"artifact_root": "artifacts/s6_profile/run_v3/illegal"}))
        with self.assertRaisesRegex(ConfigError, "may not grant"):
            self.modified(lambda raw: raw["approvals"].update({"campaign_execution": "APPROVED"}))

    def test_completed_plan_requires_exact_hashes_and_artifacts(self) -> None:
        contract = {"config_sha256": "a", "input_sha256": "b", "code_sha256": "c"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = {"status": "VERIFIED", "contract": contract}
            (output / "summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
            manifest = {
                "contract": contract,
                "artifacts": {"summary.json": sha256_file(output / "summary.json")},
            }
            (output / "artifact_manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
            self.assertEqual(validate_existing_outputs(output, contract), summary)
            with self.assertRaisesRegex(StateError, "different"):
                validate_existing_outputs(output, {**contract, "code_sha256": "changed"})
            (output / "summary.json").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "changed"):
                validate_existing_outputs(output, contract)


if __name__ == "__main__":
    unittest.main()
