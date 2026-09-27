from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from capstone_lab.errors import ArtifactError, ConfigError, StateError
from capstone_lab.profiling.contracts import (
    launch_guard,
    load_s6_campaign,
    reservation_cap_mib,
    select_common_batch,
)
from capstone_lab.profiling.runner import (
    _loss_consistency,
    classify_worker_status,
    validate_resume_state,
)


class S6ResourceContractTests(unittest.TestCase):
    def test_reservation_cap_and_external_occupancy_guard(self) -> None:
        self.assertEqual(reservation_cap_mib(16_303), 13_857)
        accepted = launch_guard(
            total_mib=16_303,
            external_used_mib=600,
            reservations_mib=[5_000, 5_000],
        )
        self.assertTrue(accepted["accepted"])
        rejected = launch_guard(
            total_mib=16_303,
            external_used_mib=4_000,
            reservations_mib=[5_000, 5_000],
        )
        self.assertFalse(rejected["accepted"])
        self.assertEqual(
            rejected["reason"],
            "EXTERNAL_OCCUPANCY_PLUS_RESERVATION_EXCEEDS_CAP",
        )
        with self.assertRaises(ConfigError):
            launch_guard(total_mib=16_303, external_used_mib=-1, reservations_mib=[1])

    def test_common_batch_uses_measured_two_job_margin_without_autoshrink(self) -> None:
        results = [
            {"status": "SUCCEEDED", "batch": 4, "memory": {"torch_peak_reserved_mib": 2_000}},
            {"status": "SUCCEEDED", "batch": 8, "memory": {"torch_peak_reserved_mib": 4_000}},
            {"status": "OOM", "batch": 16, "memory": {"torch_peak_reserved_mib": 15_000}},
        ]
        selected = select_common_batch(results, total_mib=16_303, external_used_mib=600)
        self.assertEqual(selected["batch"], 8)
        self.assertFalse(selected["formal_config_frozen"])
        with self.assertRaises(ArtifactError):
            select_common_batch(results[2:], total_mib=16_303, external_used_mib=600)

    def test_resume_requires_exact_config_input_and_code_hashes(self) -> None:
        contract = {"config_sha256": "a", "input_sha256": "b", "code_sha256": "c"}
        validate_resume_state({"contract": contract}, contract)
        with self.assertRaises(StateError):
            validate_resume_state(
                {"contract": {**contract, "code_sha256": "changed"}}, contract
            )

    def test_loss_tolerance_is_relative_and_still_bounded(self) -> None:
        def record(values: list[float]) -> dict:
            return {"result": {"loss": {"values": values}}}

        accepted = _loss_consistency([record([400.0, 200.0]), record([398.0, 199.0])])
        rejected = _loss_consistency([record([400.0, 200.0]), record([380.0, 190.0])])
        self.assertTrue(accepted["within_tolerance"])
        self.assertFalse(rejected["within_tolerance"])

    def test_worker_failure_isolated_and_gpu_oom_is_distinct(self) -> None:
        self.assertEqual(classify_worker_status("gpu", 0, "SUCCEEDED"), "SUCCEEDED")
        self.assertEqual(classify_worker_status("gpu", 42, "OOM"), "OOM")
        self.assertEqual(classify_worker_status("gpu", 1, "FAILED"), "FAILED")
        self.assertEqual(classify_worker_status("cpu", 1, "FAILED"), "FAILED")


class S6FrozenConfigTests(unittest.TestCase):
    def test_repository_contract_is_profile_only_and_preserves_s5(self) -> None:
        root = Path(__file__).resolve().parents[1]
        campaign = load_s6_campaign(root / "configs/campaigns/s6_profile.json", root)
        self.assertEqual(campaign.batches, (4, 8, 16))
        self.assertEqual(campaign.s5.scale, "n")
        self.assertEqual(campaign.s5.num_classes, 1)
        self.assertEqual(campaign.s5.effective_args["optimizer"], "AdamW")
        self.assertEqual(campaign.expected_initial_sha256, "89d7cd90a029ae51a24f7df433c644adc8c6dd8faacf804d687debaba915bc18")
        self.assertIn("artifacts\\s6_profile", str(campaign.artifact_root))

    def test_test_path_is_rejected_before_any_execution(self) -> None:
        root = Path(__file__).resolve().parents[1]
        raw = json.loads((root / "configs/campaigns/s6_profile.json").read_text(encoding="utf-8"))
        raw["inputs"]["s4_source_manifest"] = "tests/fixtures/forbidden_test.json"
        with tempfile.NamedTemporaryFile("w", suffix=".json", dir=root, delete=False, encoding="utf-8") as handle:
            json.dump(raw, handle)
            path = Path(handle.name)
        try:
            with self.assertRaisesRegex(ConfigError, "Test inputs"):
                load_s6_campaign(path, root)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
