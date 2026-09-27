from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from capstone_lab.errors import ArtifactError
from .contracts import output_path, read_json
from .epochs import train_epochs
from .io import atomic_json


def equal_state(left, right):
    import numpy as np
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def verify(root, output, experiment):
    if output.exists():
        raise ArtifactError("new epoch preflight output required")
    output.mkdir(parents=True)
    common = dict(experiment=experiment, seed=0, epochs=2, smoke=True)
    continuous = train_epochs(root, output / "continuous", **common)
    gc.collect()
    torch.cuda.empty_cache()
    paused = train_epochs(root, output / "resumed", stop_after=1, **common)
    if paused["status"] != "PAUSED" or paused["epoch"] != 1:
        raise ArtifactError("epoch checkpoint pause failed")
    gc.collect()
    torch.cuda.empty_cache()
    resumed = train_epochs(root, output / "resumed", **common)
    states = []
    for name in ("continuous", "resumed"):
        pointer = read_json(output / name / "current.json")
        states.append(torch.load(output / name / pointer["last"]["file"], map_location="cpu", weights_only=False))
    checks = {key: equal_state(states[0][key], states[1][key]) for key in ("model", "optimizer", "scaler", "scheduler", "rng", "initial")}
    checks["data_streams"] = [h["data_stream_sha256"] for h in continuous["history"]] == [h["data_stream_sha256"] for h in resumed["history"]]
    checks["losses"] = [h["loss"] for h in continuous["history"]] == [h["loss"] for h in resumed["history"]]
    checks["best_epoch_score"] = all(continuous["best_mask"][k] == resumed["best_mask"][k] for k in ("epoch", "score"))
    result = {"status": "VERIFIED" if all(checks.values()) else "FAILED", "checks": checks,
              "experiment": experiment, "scope": states[0]['contract']['training_subset'],
              "state_sha256": resumed["state_sha256"], "resumed_from_epoch": resumed["resumed_from_epoch"],
              "peak_reserved_mib": max(h["train_peak_reserved_mib"] for h in continuous["history"]),
              "code_sha256": __import__("capstone_lab.campaign.contracts", fromlist=["digest"]).digest(states[0]["contract"]["code"])}
    atomic_json(output / "summary.json", result, immutable=True)
    if result["status"] != "VERIFIED":
        raise ArtifactError(f"epoch continuation mismatch: {checks}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", choices=["L0", "H0_R", "H1_R"], default="L0")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    print(verify(root, output_path(root, args.output), args.experiment))
