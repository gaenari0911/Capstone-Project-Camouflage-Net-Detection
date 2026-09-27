from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from capstone_lab.config import canonical_json, sha256_file
from capstone_lab.errors import ManifestError
from capstone_lab.synthesis.contract import CONDITIONS
from capstone_lab.synthesis.smoke import (
    InjectedInterruption,
    SmokeConfig,
    build_smoke_source_manifest,
    run_actual_smoke_campaign,
    validate_rendered_bundle,
)


class S4SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = Path(__file__).resolve().parents[1]

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        shutil.copy2(
            self.repository / "build_demo_synthetic_segmentation.py",
            self.root / "build_demo_synthetic_segmentation.py",
        )
        self._make_sources()
        self.resolution = self.root / "manifests" / "resolution.jsonl"
        self.resolution.parent.mkdir(parents=True)
        resolved_image = self.root / "Dataset/object_pool/non_snow/images/fg_00.png"
        resolved_mask = self.root / "Dataset/object_pool/non_snow/masks/fg_00.png"
        resolution_row = {
            "chosen_environment": "non_snow",
            "csv_environment": "snow",
            "decision_rule": "disk_directory_is_authoritative",
            "disk_derived_environment": ["non_snow"],
            "image": "Dataset/object_pool/non_snow/images/fg_00.png",
            "image_sha256": sha256_file(resolved_image),
            "mask": "Dataset/object_pool/non_snow/masks/fg_00.png",
            "mask_sha256": sha256_file(resolved_mask),
            "object_source_id": "fg_00",
            "source_review_sha256": "0" * 64,
            "status": "RESOLVED",
        }
        self.resolution.write_text(
            canonical_json(resolution_row) + "\n", encoding="utf-8"
        )
        self.source_manifest = self.root / "artifacts" / "source_manifest.json"
        result = build_smoke_source_manifest(
            project_root=self.root,
            output_path=self.source_manifest,
            resolution_manifest_path=self.resolution,
            max_foregrounds_per_environment=8,
        )
        self.assertEqual(result["status"], "VERIFIED")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_sources(self) -> None:
        background_dir = self.root / "Dataset/synthesis_demo/BackGround/forest_dense"
        image_dir = self.root / "Dataset/object_pool/non_snow/images"
        mask_dir = self.root / "Dataset/object_pool/non_snow/masks"
        background_dir.mkdir(parents=True)
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        yy, xx = np.mgrid[0:160, 0:220]
        for index in range(3):
            background = np.zeros((160, 220, 3), dtype=np.uint8)
            background[:, :, 0] = (xx + 20 * index) % 255
            background[:, :, 1] = (yy * 2 + 35 * index) % 255
            background[:, :, 2] = (xx // 2 + yy + 15 * index) % 255
            cv2.imwrite(str(background_dir / f"bg_{index:02d}.png"), background)
        for index in range(8):
            image = np.zeros((80, 100, 3), dtype=np.uint8)
            image[:, :, 0] = 40 + index * 8
            image[:, :, 1] = (np.arange(100)[None, :] * 2 + index * 3) % 255
            image[:, :, 2] = (np.arange(80)[:, None] * 3 + index * 5) % 255
            mask = np.zeros((80, 100), dtype=np.uint8)
            cv2.ellipse(mask, (50, 40), (28, 22), index * 5, 0, 360, 255, -1)
            cv2.imwrite(str(image_dir / f"fg_{index:02d}.png"), image)
            cv2.imwrite(str(mask_dir / f"fg_{index:02d}.png"), mask)

    def run_smoke(
        self,
        output_name: str,
        workers: int,
        **kwargs,
    ) -> dict:
        return run_actual_smoke_campaign(
            project_root=self.root,
            source_manifest_path=self.source_manifest,
            resolution_manifest_path=self.resolution,
            output_dir=self.root / "artifacts" / output_name,
            workers=workers,
            config=SmokeConfig(
                sample_count=2,
                candidates_per_sample=2,
                positions_per_candidate=2,
                target_long_side=128,
            ),
            enforce_smoke_minimum=False,
            **kwargs,
        )

    def test_workers_1_2_6_have_identical_real_outputs_and_score_contract(self) -> None:
        one = self.run_smoke("w1", 1)
        two = self.run_smoke("w2", 2)
        six = self.run_smoke("w6", 6)
        self.assertEqual(one["status"], "VERIFIED")
        self.assertEqual(one["deterministic_payload"], two["deterministic_payload"])
        self.assertEqual(one["deterministic_payload"], six["deterministic_payload"])
        self.assertEqual(one["deterministic_sha256"], two["deterministic_sha256"])
        self.assertEqual(one["deterministic_sha256"], six["deterministic_sha256"])
        self.assertEqual(one["condition_counts"], {condition: 2 for condition in CONDITIONS})
        for sample in one["deterministic_payload"]["samples"]:
            for condition, selection in sample["selections"].items():
                if condition == "A0":
                    self.assertEqual(
                        selection["selection_method"], "seeded_random_no_score_ranking"
                    )
                    self.assertIsNone(selection["final_score"])
                    continue
                disabled = {"A2": "tone", "A3": "texture", "A4": "edge", "A5": "placement"}.get(condition)
                for component, weight in selection["effective_weights"].items():
                    if component == disabled:
                        self.assertEqual(weight, 0.0)
                    elif condition != "A0":
                        self.assertGreater(weight, 0.0)

    def test_interrupted_coordinator_resumes_without_overwriting_success(self) -> None:
        output = self.root / "artifacts" / "resume"
        with self.assertRaises(InjectedInterruption):
            self.run_smoke("resume", 2, interrupt_after_commits=3)
        committed = sorted(output.glob("samples/*/*/COMMITTED.json"))
        self.assertEqual(len(committed), 3)
        before = {
            str(path): (sha256_file(path), path.stat().st_mtime_ns) for path in committed
        }
        resumed = self.run_smoke("resume", 6)
        clean = self.run_smoke("clean", 1)
        self.assertEqual(resumed["status"], "VERIFIED")
        self.assertEqual(
            resumed["deterministic_payload"], clean["deterministic_payload"]
        )
        for path_text, expected in before.items():
            path = Path(path_text)
            self.assertEqual((sha256_file(path), path.stat().st_mtime_ns), expected)

    def test_worker_failure_partial_file_and_validation_failure_do_not_count(self) -> None:
        failed = self.run_smoke(
            "failures",
            2,
            fail_render_keys=("s4_sample_000001/A1",),
            corrupt_render_keys=("s4_sample_000001/A2",),
        )
        self.assertEqual(failed["status"], "PARTIAL_FAILURE")
        self.assertEqual(failed["condition_counts"]["A1"], 1)
        self.assertEqual(failed["condition_counts"]["A2"], 1)
        self.assertFalse(
            (
                self.root
                / "artifacts/failures/samples/s4_sample_000001/A1/COMMITTED.json"
            ).exists()
        )
        self.assertTrue(
            any("mask_decode_failed" in reason for row in failed["failures"] for reason in row["reasons"])
        )
        partial = (
            self.root
            / "artifacts/failures/samples/s4_sample_000001/A1"
        )
        partial.mkdir(parents=True)
        (partial / "image.png").write_bytes(b"partial")
        resumed = self.run_smoke("failures", 6)
        self.assertEqual(resumed["status"], "VERIFIED")
        self.assertTrue(
            any((self.root / "artifacts/failures/_failed/s4_sample_000001").iterdir())
        )

    def test_pending_bundle_and_feature_worker_termination_resume(self) -> None:
        for artifact, failure_stage in (
            ("image.png", "after_image"),
            ("mask.png", "after_mask"),
            ("polygon.txt", "after_polygon"),
        ):
            output_name = "pending_" + failure_stage
            with self.assertRaises(InjectedInterruption):
                self.run_smoke(
                    output_name,
                    2,
                    inject_commit_failure=("s4_sample_000001/A0", failure_stage),
                )
            self.assertFalse(
                (
                    self.root
                    / f"artifacts/{output_name}/samples/s4_sample_000001/A0/COMMITTED.json"
                ).exists()
            )
            self.assertTrue(
                any(
                    (self.root / f"artifacts/{output_name}/.pending").rglob(artifact)
                )
            )
            self.assertEqual(self.run_smoke(output_name, 6)["status"], "VERIFIED")

        worker_failed = self.run_smoke(
            "worker_failed", 2, fail_feature_sample_ids=("s4_sample_000001",)
        )
        self.assertEqual(worker_failed["status"], "PARTIAL_FAILURE")
        for condition in CONDITIONS:
            self.assertEqual(worker_failed["condition_counts"][condition], 1)
        self.assertEqual(self.run_smoke("worker_failed", 1)["status"], "VERIFIED")

        worker_terminated = self.run_smoke(
            "worker_terminated",
            2,
            terminate_feature_sample_ids=("s4_sample_000001",),
        )
        self.assertEqual(worker_terminated["status"], "PARTIAL_FAILURE")
        self.assertTrue(
            any(
                "BrokenProcessPool" in reason
                for row in worker_terminated["failures"]
                for reason in row["reasons"]
            )
        )
        self.assertEqual(self.run_smoke("worker_terminated", 1)["status"], "VERIFIED")

    def test_hash_change_duplicate_and_protected_output_are_rejected(self) -> None:
        self.assertEqual(self.run_smoke("contract", 1)["status"], "VERIFIED")
        with self.assertRaises(ManifestError):
            run_actual_smoke_campaign(
                project_root=self.root,
                source_manifest_path=self.source_manifest,
                resolution_manifest_path=self.resolution,
                output_dir=self.root / "artifacts/contract",
                workers=1,
                config=SmokeConfig(
                    campaign_seed=8,
                    sample_count=2,
                    candidates_per_sample=2,
                    positions_per_candidate=2,
                    target_long_side=128,
                ),
                enforce_smoke_minimum=False,
            )

        payload = json.loads(self.source_manifest.read_text(encoding="utf-8"))
        payload["backgrounds"].append(dict(payload["backgrounds"][0]))
        duplicate = self.root / "artifacts/duplicate.json"
        duplicate.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        with self.assertRaises(ManifestError):
            run_actual_smoke_campaign(
                project_root=self.root,
                source_manifest_path=duplicate,
                resolution_manifest_path=self.resolution,
                output_dir=self.root / "artifacts/duplicate_output",
                workers=1,
                config=SmokeConfig(
                    sample_count=2,
                    candidates_per_sample=2,
                    positions_per_candidate=2,
                    target_long_side=128,
                ),
                enforce_smoke_minimum=False,
            )
        with self.assertRaises(ManifestError):
            run_actual_smoke_campaign(
                project_root=self.root,
                source_manifest_path=self.source_manifest,
                resolution_manifest_path=self.resolution,
                output_dir=self.root / "Dataset/output",
                workers=1,
                config=SmokeConfig(
                    sample_count=2,
                    candidates_per_sample=2,
                    positions_per_candidate=2,
                    target_long_side=128,
                ),
                enforce_smoke_minimum=False,
            )

    def test_validator_records_decode_binary_polygon_and_metadata_reasons(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.ones((16, 16), dtype=np.uint8) * 17
        _, image_bytes = cv2.imencode(".png", image)
        _, mask_bytes = cv2.imencode(".png", mask)
        reasons = validate_rendered_bundle(
            image=image_bytes.tobytes(),
            mask=mask_bytes.tobytes(),
            polygon=b"0 1.2 0.0 0.5 0.5 0.0 1.0\n",
            metadata={},
        )
        self.assertIn("image_mask_size_mismatch", reasons)
        self.assertIn("mask_not_binary", reasons)
        self.assertIn("polygon_coordinate_out_of_range", reasons)
        self.assertTrue(any(reason.startswith("metadata_missing:") for reason in reasons))


if __name__ == "__main__":
    unittest.main()
