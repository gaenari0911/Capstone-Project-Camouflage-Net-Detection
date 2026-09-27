"""Produce an evidence-bound real-only release. This does not approve AnyDoor or Test."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from capstone_lab.config import sha256_file
from capstone_lab.errors import ArtifactError
from .contracts import APPROVAL_SHA256, code_records, digest, output_path, read_json
from .io import atomic_json


def run_suite(root, output):
    if output.exists():
        raise ArtifactError("new release output required")
    output.mkdir(parents=True)
    before = code_records(root)
    variants = ["L0", "H0_R", "H1_R"]
    evidence = {}
    def run(module, relative, extra=()):
        destination = root / relative
        with (output / (destination.name + "_" + module.rsplit(".", 1)[-1] + ".log")).open("xb") as log:
            subprocess.run([sys.executable, "-m", module, "--output", relative, *extra],
                           cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        path = destination / "summary.json"
        result = read_json(path)
        if result["status"] != "VERIFIED":
            raise ArtifactError("release prerequisite did not pass")
        evidence[path.relative_to(root).as_posix()] = sha256_file(path)
        return result
    for experiment in variants:
        relative = (output / experiment).relative_to(root).as_posix()
        result = run("capstone_lab.campaign.epoch_preflight", relative, ("--experiment", experiment))
        if not all(result["checks"].values()) or result["code_sha256"] != digest(before):
            raise ArtifactError("epoch stream/optimizer/RNG or code mismatch")
    safety = run("capstone_lab.campaign.epoch_safety", (output / "safety").relative_to(root).as_posix())
    if safety["state_sha256"] != read_json(output / "L0/summary.json")["state_sha256"]:
        raise ArtifactError("supervised safety resume differs from continuous epoch result")
    regression = run("capstone_lab.campaign.verification", (output / "regression").relative_to(root).as_posix())
    if regression["code_before"] != before or regression["code_after"] != before or code_records(root) != before:
        raise ArtifactError("code changed during release acceptance")
    summary = {"status": "VERIFIED_REAL_EPOCH_RELEASE", "variants": variants,
               "approval_sha256": APPROVAL_SHA256, "code_sha256": digest(before), "evidence": evidence,
               "tests": regression["tests"], "safe_epoch_pause_verified": True,
               "gpu_concurrency_effective": 1, "gpu_reserved_mib": 10044,
               "formal_training_completed": 0, "scope": "real_only_training_release_not_S8_complete"}
    atomic_json(output / "summary.json", summary, immutable=True)
    jobs = []
    # All four seed0 branches first. No metric-improvement gate or per-seed approval.
    for seed in (0, 1, 2):
        for experiment in ("L0", "M0", "H0_R", "H1_R"):
            jobs.append({"id": f"{experiment}_seed{seed}", "kind": "epoch_train",
                         "dependencies": [f"{experiment}_seed0"] if seed else [], "cpu": 3,
                         "max_attempts": 3, "duration_seconds": 0, "fail_attempts": [], "partial_attempts": [],
                         "training": {"experiment": experiment, "seed": seed, "epochs": 150, "stop_after_epoch": 0}})
    config = {"schema_version": 1, "mode": "s8_real_campaign", "output": "artifacts/s8_real_campaign/run_v1",
              "cpu_slots": 6, "max_parallel": 2, "disk_budget_gib": 100, "min_free_gib": 20,
              "active_seconds_max": 604800, "jobs": jobs,
              "inputs": {(output / "summary.json").relative_to(root).as_posix(): sha256_file(output / "summary.json")}}
    atomic_json(output / "real_campaign.json", config, immutable=True)
    print({"status": summary["status"], "config": str(output / "real_campaign.json"),
           "config_sha256": digest(config), "code_sha256": summary["code_sha256"], "jobs": len(jobs)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    run_suite(root, output_path(root, args.output))
