from __future__ import annotations

import argparse
from pathlib import Path

from capstone_lab.config import sha256_file
from capstone_lab.manifests import tree_fingerprint
from .contracts import APPROVAL_SHA256, code_records, output_path, read_json, readiness_plan
from .io import atomic_json
from .sources import DATASET_SHA256
from .supervisor import snapshot


def assemble(root, output):
    plan = readiness_plan(root)
    evidence_paths = {
        "regression_tests": "artifacts/s8_regression/run_v2/summary.json",
        "source_membership": "artifacts/s8_source_audit/run_v1/low187_object_provenance.json",
        "real_batch_and_validation": "artifacts/s8_real_preflight/run_v2/summary.json",
        "real_checkpoint_resume": "artifacts/s8_resume_preflight/run_v4/summary.json",
        "dense_batch_memory": "artifacts/s8_stress_preflight/run_v1/summary.json",
    }
    evidence = {}
    for key, relative in evidence_paths.items():
        path = root / relative
        evidence[key] = {"path": relative, "exists": path.exists()}
        if path.exists():
            evidence[key].update(sha256=sha256_file(path), status=read_json(path)["status"])
    original = tree_fingerprint(root / "Dataset", workers=2)
    if original["sha256"] != DATASET_SHA256:
        raise RuntimeError("protected Dataset hash changed")
    pipeline = snapshot(root / "artifacts/s8_preflight/run_v2")
    result = {"schema_version": 1, "status": "S8_IN_PROGRESS_NOT_FULLY_VERIFIED",
              "approval_sha256": APPROVAL_SHA256, "evidence": evidence, "dataset": original,
              "code": code_records(root), "preflight_pipeline": pipeline,
              "formal_campaign": {"training_jobs_completed": 0, "synthetic_images_created": 0, "test_evaluations": 0},
              "ssh_disconnect": "BLOCKED_USER_ACTION_UNTESTED",
              "resource_conclusion": {"batch":16, "gpu_train_concurrency_safe_for_dense_batch":1,
                                     "measured_peak_mib":9532, "reservation_mib":10044,
                                     "concurrency2_not_verified":True},
              "remaining": ["formal epoch loop, optimizer/warmup/accumulation and epoch resume",
                            "resource-aware formal GPU dispatch and budget/checkpoint pause",
                            "paired full-resolution 3000-per-condition synthesis adapter",
                            "AnyDoor compatibility/install/pilot and actual quality review",
                            "shared pool selection/Judge inference",
                            "final checkpoint/evaluation freeze and approved Test dispatch",
                            "real SSH disconnect/reconnect verification"],
              "planning_scope": {"training_jobs":plan["training_jobs"], "epoch_job_units":plan["epoch_job_units"],
                                 "formal_images":plan["formal_images"]}}
    atomic_json(output / "approved_dag_readiness.json", plan, immutable=True)
    atomic_json(output / "summary.json", result, immutable=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    result = assemble(root, output_path(root, args.output))
    print(result["status"])


if __name__ == "__main__":
    main()
