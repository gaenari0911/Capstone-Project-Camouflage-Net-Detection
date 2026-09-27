from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from capstone_lab.errors import ArtifactError, ConfigError
from capstone_lab.training.checkpoints import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from capstone_lab.training.metrics import evaluate_paired_masks
from capstone_lab.training.scheduler import (
    S5Job,
    _assert_non_test_manifest,
    dependency_blocked_jobs,
    load_s5_campaign,
)


class S5MetricTests(unittest.TestCase):
    def test_mask_fixture_matches_independent_reference(self) -> None:
        targets = [[1, 1, 1, 1] for _ in range(4)]
        predictions = [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ]
        result = evaluate_paired_masks(
            predictions,
            targets,
            [0.9, 0.8, 0.7, 0.6],
            operating_confidence=0.25,
            operating_iou=0.5,
        )
        self.assertAlmostEqual(result["mask"]["map50"], 0.75)
        self.assertAlmostEqual(result["mask"]["map50_95"], 0.425)
        self.assertAlmostEqual(result["mask"]["precision"], 0.75)
        self.assertAlmostEqual(result["mask"]["recall"], 0.75)
        self.assertAlmostEqual(result["mask"]["f1"], 0.75)
        self.assertIsNone(result["box"])


class S5SafetyAndStateTests(unittest.TestCase):
    def test_test_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps({"schema_version": 1, "split": "test"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "Test manifest"):
                _assert_non_test_manifest(path, "train")

    def test_failed_job_blocks_only_descendants(self) -> None:
        jobs = (
            S5Job("failed", (), 1, 1, 0),
            S5Job("dependent", ("failed",), 1, 1, 0),
            S5Job("independent", (), 1, 1, 0),
        )
        blocked = dependency_blocked_jobs(
            jobs,
            {"failed": "FAILED", "dependent": "PENDING", "independent": "SUCCEEDED"},
        )
        self.assertEqual(blocked, {"dependent"})

    def test_frozen_campaign_has_two_explicit_gpu_jobs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_s5_campaign(
            root / "configs/campaigns/s5_dualhead_smoke.json", root
        )
        self.assertEqual(config.scale, "n")
        self.assertEqual(config.num_classes, 1)
        self.assertEqual(len(config.jobs), 2)
        self.assertEqual(config.jobs[1].dependencies, (config.jobs[0].job_id,))
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.pretrained_sha256, "51fa7e5ef385efa6d5b1d8e31b73399be6ed5d7ca71bda4bd4b2794bb445c4f4")


class S5CheckpointTests(unittest.TestCase):
    def test_full_checkpoint_resume_and_hash_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            loss = model(torch.ones(1, 3)).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            contract = {
                "config_sha256": "a" * 64,
                "input_sha256": "b" * 64,
                "code_sha256": "c" * 64,
            }
            digest = save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=0,
                global_step=1,
                contract=contract,
            )
            raw = torch.load(path, weights_only=False)
            self.assertEqual(
                set(raw),
                {
                    "schema_version",
                    "model",
                    "optimizer",
                    "scheduler",
                    "scaler",
                    "epoch",
                    "global_step",
                    "rng",
                    "contract",
                },
            )
            restored = torch.nn.Linear(3, 2)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
            restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                restored_optimizer, T_max=2
            )
            restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
            record = load_training_checkpoint(
                path,
                model=restored,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                scaler=restored_scaler,
                expected_contract=contract,
            )
            self.assertEqual(record["checkpoint_sha256"], digest)
            self.assertEqual(record["global_step"], 1)
            for expected, actual in zip(
                model.parameters(), restored.parameters(), strict=True
            ):
                self.assertTrue(torch.equal(expected, actual))
            with self.assertRaisesRegex(ArtifactError, "contract"):
                load_training_checkpoint(
                    path,
                    model=restored,
                    optimizer=restored_optimizer,
                    scheduler=restored_scheduler,
                    scaler=restored_scaler,
                    expected_contract={**contract, "input_sha256": "d" * 64},
                )


if __name__ == "__main__":
    unittest.main()
