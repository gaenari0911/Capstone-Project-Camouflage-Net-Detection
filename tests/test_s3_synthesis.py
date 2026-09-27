from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from capstone_lab.errors import ManifestError
from capstone_lab.synthesis.contract import (
    COMPONENTS,
    CandidateFeature,
    CandidatePlan,
    FixtureConfig,
    build_candidate_plans,
    verify_legacy_contract,
)
from capstone_lab.synthesis.executor import run_fixture_campaign
from capstone_lab.synthesis.scoring import (
    UsageState,
    legacy_reference_final_score,
    score_candidate,
    select_candidate,
)


class S3SynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.legacy = cls.project_root / "build_demo_synthetic_segmentation.py"
        cls.fixture = cls.project_root / "tests" / "fixtures" / "s3_fixture.json"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.protected = self.root / "protected_source"
        self.protected.mkdir()
        (self.protected / "sentinel.txt").write_text("unchanged", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_fixture(
        self,
        workers: int,
        *,
        delay_order: str = "none",
        config: FixtureConfig | None = None,
        fail_sample_ids: tuple[str, ...] = (),
        name: str | None = None,
    ) -> dict:
        return run_fixture_campaign(
            fixture_path=self.fixture,
            legacy_source_path=self.legacy,
            output_dir=self.root / (name or f"output_w{workers}_{delay_order}"),
            allowed_root=self.root,
            protected_paths=(self.protected,),
            workers=workers,
            config=config,
            delay_order=delay_order,
            fail_sample_ids=fail_sample_ids,
        )

    def test_workers_and_completion_order_are_deterministic(self) -> None:
        one = self.run_fixture(1, delay_order="none", name="one")
        two = self.run_fixture(2, delay_order="forward", name="two")
        six = self.run_fixture(6, delay_order="reverse", name="six")
        self.assertTrue(one["single_shared_process_pool"])
        self.assertEqual(one["deterministic_payload"], two["deterministic_payload"])
        self.assertEqual(one["deterministic_payload"], six["deterministic_payload"])
        self.assertEqual(one["deterministic_sha256"], two["deterministic_sha256"])
        self.assertEqual(one["deterministic_sha256"], six["deterministic_sha256"])
        for relative, expected in one["deterministic_payload"]["output_hashes"].items():
            content = (self.root / "one" / relative).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected)

    def test_common_plan_contains_required_fields_and_condition_free_seeds(self) -> None:
        config = FixtureConfig(sample_count=3)
        first = build_candidate_plans(config)
        second = build_candidate_plans(config)
        self.assertEqual(first, second)
        for plan in first:
            value = plan.to_dict()
            self.assertEqual(
                {
                    "sample_id",
                    "background_id",
                    "mode",
                    "domain",
                    "difficulty",
                    "scale_bucket",
                    "scale_ratio",
                    "foreground_candidate_ids",
                    "position_candidate_ids",
                    "candidate_seed",
                    "rendering_seed",
                    "attempt",
                },
                set(value),
            )
            self.assertFalse(any(condition in plan.candidate_seed for condition in ("A0", "A1")))
            self.assertEqual(len(plan.candidate_seed), 64)
            self.assertEqual(len(plan.rendering_seed), 64)

    def test_a1_matches_parsed_legacy_constants_and_reference_formula(self) -> None:
        contract = verify_legacy_contract(self.legacy)
        self.assertEqual(contract["artifact_assignments"], [(51, 0.12), (113, 0.16)])
        self.assertEqual(contract["effective_artifact_penalty_weight"], 0.16)
        feature = self.feature("candidate")
        guidance = {
            "remaining_quota": 3.0,
            "quota_bonus": 0.6,
            "domain_repeat_penalty": 0.2,
            "background_repeat_penalty": 1.0,
            "recent_history_penalty": 0.0,
            "coarse_domain_penalty": 0.0,
        }
        metadata = score_candidate(
            "A1", feature, "natural", guidance, sample_seed="a" * 64, attempt=0
        )
        self.assertAlmostEqual(
            metadata["final_score"],
            legacy_reference_final_score(feature, "natural", guidance),
            places=14,
        )

    def test_ablation_toggles_only_target_component_without_renormalizing(self) -> None:
        feature = self.feature("candidate")
        guidance = UsageState(("fg",)).guidance("fg", "forest_dense", "bg")
        baseline = score_candidate(
            "A1", feature, "natural", guidance, sample_seed="b" * 64, attempt=0
        )
        disabled_by_condition = {
            "A2": "tone",
            "A3": "texture",
            "A4": "edge",
            "A5": "placement",
        }
        for condition, disabled in disabled_by_condition.items():
            metadata = score_candidate(
                condition,
                feature,
                "natural",
                guidance,
                sample_seed="b" * 64,
                attempt=0,
            )
            for component in COMPONENTS:
                expected = 0.0 if component == disabled else baseline["effective_weights"][component]
                self.assertEqual(metadata["effective_weights"][component], expected)
                self.assertEqual(metadata["enabled_flags"][component], component != disabled)
            self.assertEqual(metadata["artifact_contribution"], baseline["artifact_contribution"])
            self.assertEqual(
                metadata["usage_quota_contribution"],
                baseline["usage_quota_contribution"],
            )
        self.assertTrue(
            score_candidate(
                "A2", feature, "natural", guidance, sample_seed="b" * 64, attempt=0
            )["tone_matching_enabled"]
        )
        a4 = score_candidate(
            "A4", feature, "natural", guidance, sample_seed="b" * 64, attempt=0
        )
        self.assertTrue(a4["alpha_blending_enabled"])
        self.assertTrue(a4["boundary_rendering_enabled"])

    def test_a0_never_calls_score_ranking(self) -> None:
        plan = self.plan()
        state = UsageState(("fg",))

        def forbidden(*args, **kwargs):
            raise AssertionError("score ranking was called")

        result = select_candidate(
            "A0",
            plan,
            [self.feature("b"), self.feature("a")],
            state,
            7,
            scoring_function=forbidden,
        )
        self.assertIsNotNone(result)
        _, metadata = result
        self.assertEqual(metadata["selection_method"], "seeded_random_no_score_ranking")
        self.assertIsNone(metadata["final_score"])
        self.assertTrue(metadata["tone_matching_enabled"])
        self.assertTrue(metadata["boundary_rendering_enabled"])

    def test_duplicate_candidates_and_score_ties_use_stable_tie_break(self) -> None:
        plan = self.plan()
        a = self.feature("a")
        b = self.feature("b")
        duplicate_a = self.feature("a")
        state = UsageState(("fg",))
        result = select_candidate("A1", plan, [b, duplicate_a, a], state, 7)
        self.assertIsNotNone(result)
        selected, metadata = result
        self.assertEqual(selected.candidate_id, "a")
        self.assertEqual(metadata["tie_break_key"], "a")

    def test_worker_failure_does_not_update_failed_sample_state(self) -> None:
        result = self.run_fixture(
            2,
            config=FixtureConfig(sample_count=3),
            fail_sample_ids=("sample_000002",),
            name="failure",
        )
        self.assertEqual(result["status"], "PARTIAL_FAILURE")
        rows = result["deterministic_payload"]["samples"]
        self.assertEqual([row["status"] for row in rows], ["SUCCEEDED", "FEATURE_FAILED", "SUCCEEDED"])
        self.assertFalse(rows[1]["state_updated"])
        for snapshot in result["deterministic_payload"]["usage_states"].values():
            self.assertEqual(sum(snapshot["used_total"].values()), 2)

    def test_output_overlap_and_invalid_worker_count_are_rejected(self) -> None:
        with self.assertRaises(ManifestError):
            run_fixture_campaign(
                fixture_path=self.fixture,
                legacy_source_path=self.legacy,
                output_dir=self.protected / "output",
                allowed_root=self.root,
                protected_paths=(self.protected,),
                workers=1,
            )
        with self.assertRaises(ManifestError):
            self.run_fixture(3)

    def test_a5_keeps_common_scale_domain_plan(self) -> None:
        result = self.run_fixture(1, config=FixtureConfig(sample_count=1), name="a5")
        plan = result["deterministic_payload"]["plans"][0]
        metadata = result["deterministic_payload"]["samples"][0]["selections"]["A5"]
        self.assertEqual(metadata["scale_bucket"], plan["scale_bucket"])
        self.assertEqual(metadata["scale_ratio"], plan["scale_ratio"])
        self.assertEqual(metadata["domain"], plan["domain"])
        self.assertEqual(metadata["background_id"], plan["background_id"])

    @staticmethod
    def feature(candidate_id: str) -> CandidateFeature:
        return CandidateFeature(
            sample_id="sample_000001",
            candidate_id=candidate_id,
            foreground_id="fg",
            position_id="pos",
            tone=0.5,
            texture=0.5,
            edge=0.5,
            placement=0.5,
            artifact_penalty=0.25,
            eligible=True,
            eligibility_reasons=("fixture_declared_valid",),
        )

    @staticmethod
    def plan() -> CandidatePlan:
        return CandidatePlan(
            sample_id="sample_000001",
            background_id="bg",
            mode="natural",
            domain="forest_dense",
            difficulty="medium",
            scale_bucket="10to15",
            scale_ratio=0.12,
            foreground_candidate_ids=("fg",),
            position_candidate_ids=("pos",),
            candidate_seed="c" * 64,
            rendering_seed="d" * 64,
            attempt=0,
        )


if __name__ == "__main__":
    unittest.main()
