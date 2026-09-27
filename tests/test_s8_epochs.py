from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from capstone_lab.campaign.contracts import validate_config
from capstone_lab.campaign.epochs import commit_epoch, learning_rate, make_optimizer, verify_pointer
from capstone_lab.campaign.io import atomic_json
from capstone_lab.campaign.resources import admission
from capstone_lab.errors import ArtifactError, ConfigError

ROOT = Path(__file__).resolve().parents[1]


def epoch_config(output="artifacts/s8_tests/epoch_schema"):
    return {"schema_version": 1, "mode": "s8_preflight", "output": output, "cpu_slots": 6,
            "max_parallel": 2, "inputs": {}, "disk_budget_gib": 100, "min_free_gib": 20,
            "active_seconds_max": 604800, "jobs": [{"id": "epoch_smoke", "kind": "epoch_train",
            "dependencies": [], "cpu": 3, "max_attempts": 3, "duration_seconds": 0,
            "fail_attempts": [], "partial_attempts": [], "training": {"experiment": "L0", "seed": 0,
            "epochs": 2, "stop_after_epoch": 1}}]}


class S8EpochContracts(unittest.TestCase):
    def test_fixed_schedule(self):
        self.assertAlmostEqual(learning_rate(0, 12, 150), .001 / 36)
        self.assertAlmostEqual(learning_rate(35, 12, 150), .001)
        self.assertAlmostEqual(learning_rate(36, 12, 150), .001)
        self.assertAlmostEqual(learning_rate(1799, 12, 150), .00001)

    def test_optimizer_groups(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
        optimizer, names = make_optimizer(model)
        self.assertEqual(names, {"weight": ["0.weight"], "bias": ["0.bias", "1.bias"], "norm": ["1.weight"]})
        self.assertEqual([g["weight_decay"] for g in optimizer.param_groups], [.0005, 0., 0.])
        self.assertEqual(sum(len(g["params"]) for g in optimizer.param_groups), 4)

    def test_strict_epoch_schema(self):
        config = epoch_config()
        validate_config(ROOT, config)
        for key, value in (("experiment", "Test"), ("seed", 3), ("epochs", 150), ("stop_after_epoch", 2)):
            changed = copy.deepcopy(config)
            changed["jobs"][0]["training"][key] = value
            with self.assertRaises(ConfigError):
                validate_config(ROOT, changed)
        changed = copy.deepcopy(config)
        changed["jobs"][0]["cpu"] = 2
        with self.assertRaises(ConfigError):
            validate_config(ROOT, changed)

    def test_committed_pointer_survives_incomplete_next_epoch(self):
        base = ROOT / "artifacts/s8_tests"
        base.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(dir=base))
        model = torch.nn.Linear(2, 1)
        optimizer, _ = make_optimizer(model)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        binding = {"test": "transaction"}
        history = [{"epoch": 1, "global_step": 1, "mask_map50_95": .2}]
        first = commit_epoch(directory, model, optimizer, scaler, binding, history, None, {})
        history.append({"epoch": 2, "global_step": 2, "mask_map50_95": .3})
        with patch("capstone_lab.campaign.epochs.atomic_json", side_effect=OSError("power loss before pointer")):
            with self.assertRaises(OSError):
                commit_epoch(directory, model, optimizer, scaler, binding, history, first, {})
        self.assertEqual(verify_pointer(directory, binding), first)
        second = commit_epoch(directory, model, optimizer, scaler, binding, history, first, {})
        self.assertEqual(verify_pointer(directory, binding), second)
        history.append({"epoch": 3, "global_step": 3, "mask_map50_95": .3})
        third = commit_epoch(directory, model, optimizer, scaler, binding, history, second, {})
        self.assertEqual(third["best_mask"], second["best_mask"])
        self.assertLessEqual(len(list(directory.glob("*.pt"))), 4)
        with self.assertRaises(ArtifactError):
            verify_pointer(directory, {"changed": True})
        atomic_json(directory / third["last"]["file"], {"tampered": True})
        with self.assertRaises(ArtifactError):
            verify_pointer(directory, binding)

    def test_gpu_single_slot_external_usage_and_ram(self):
        from types import SimpleNamespace
        job = epoch_config()["jobs"][0]
        with patch("capstone_lab.campaign.resources.psutil.virtual_memory", return_value=SimpleNamespace(total=32*2**30, available=24*2**30)):
            with patch("capstone_lab.campaign.resources.gpu_memory", return_value=(16303, 500)):
                self.assertTrue(admission(job, [], {})[0])
                self.assertFalse(admission(job, [{"id": job["id"]}], {job["id"]: job})[0])
            with patch("capstone_lab.campaign.resources.gpu_memory", return_value=(16303, 5000)):
                self.assertFalse(admission(job, [], {})[0])
        with patch("capstone_lab.campaign.resources.psutil.virtual_memory", return_value=SimpleNamespace(total=32*2**30, available=10*2**30)):
            self.assertFalse(admission(job, [], {})[0])


if __name__ == "__main__":
    unittest.main()
