from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from capstone_lab.config import sha256_file
from capstone_lab.errors import ConfigError
from .contracts import WORKERS, digest, inside, read_json
from .io import atomic_json


def execute(spec_path, token):
    spec = read_json(spec_path)
    run = Path(spec["run"]).resolve()
    output = inside(run, spec["output"])
    if spec["token"] != token or read_json(run / "token.json")["token"] != token:
        raise ConfigError("worker campaign token mismatch")
    binding = read_json(run / "contract.json")
    if digest(binding) != spec["contract_sha256"]:
        raise ConfigError("worker contract mismatch")
    job = spec["job"]
    if job["kind"] not in WORKERS:
        raise ConfigError("unregistered worker")
    # Fail closed if coordinator died before recording this process ownership.
    for _ in range(200):
        if (output / "dispatch.json").exists():
            dispatch = read_json(output / "dispatch.json")
            if dispatch["owner"]["pid"] != os.getpid() or dispatch["contract_sha256"] != spec["contract_sha256"]:
                raise ConfigError("invalid durable dispatch")
            break
        time.sleep(.05)
    else:
        raise ConfigError("coordinator did not commit dispatch")
    root = Path(spec["root"]).resolve()
    for group in ("inputs", "code"):
        for path, expected in binding[group].items():
            base = run / "runtime" if group == "code" else root
            if sha256_file(inside(base, path)) != expected:
                raise ConfigError(f"worker {group} changed: {path}")
    if job["kind"] == "fixture":
        atomic_json(output / "partial.json", {"partial": True, "job": job["id"]}, immutable=True)
        time.sleep(job["duration_seconds"])
        if spec["attempt"] in job["partial_attempts"]:
            return 76
        if spec["attempt"] in job["fail_attempts"]:
            return 75
        payload = {"fixture_only": True, "job": job["id"], "deterministic_value": digest(job["id"])}
    elif job["kind"] == "source_audit":
        from .sources import audit_sources
        payload = audit_sources(root, output)
    elif job["kind"] == "epoch_train":
        from .epochs import train_epochs
        from .io import RunLock
        config = read_json(run / "config.json")
        training = job["training"]
        directory = run / "jobs" / job["id"] / "training_state"
        def should_pause():
            import shutil
            from .resources import budget_roots
            from .supervisor import directory_bytes
            request = read_json(run / "control.json")
            if request.get("token") != token:
                raise ConfigError("epoch control token mismatch")
            budget = read_json(run / "epoch_pause.json") if (run / "epoch_pause.json").exists() else {}
            status = read_json(run / "status.json") if (run / "status.json").exists() else {}
            # A worker can outlive its coordinator. Enforce disk/time independently
            # at every epoch boundary, not only when a coordinator writes a flag.
            elapsed = status.get("active_seconds", 0.) + status.get("prior_campaign_active_seconds", 0.)
            elapsed += max(0., time.time() - status.get("updated", time.time()))
            used = sum(directory_bytes(p) for p in budget_roots(root)) if config["mode"] == "s8_real_campaign" else directory_bytes(run)
            exhausted = (elapsed >= config["active_seconds_max"] or
                         used + 2**30 >= config["disk_budget_gib"] * 2**30 or
                         shutil.disk_usage(run).free < (config["min_free_gib"] + 1) * 2**30)
            return request.get("action") == "pause" or budget.get("pause", False) or exhausted
        # Shared OS lock prevents two S8 runs racing onto the same physical GPU.
        with RunLock(root / "artifacts/s8_gpu.lock"):
            payload = train_epochs(root, directory, experiment=training["experiment"], seed=training["seed"],
                                   epochs=training["epochs"], smoke=config["mode"] == "s8_preflight",
                                   stop_after=training["stop_after_epoch"] if spec["attempt"] == 1 else None,
                                   should_pause=should_pause, progress_path=directory / "progress.json")
        if payload["status"] == "PAUSED":
            if should_pause():
                atomic_json(output / "safe_pause.json", {"contract_sha256": spec["contract_sha256"],
                            "epoch": payload["epoch"], "job_id": job["id"]}, immutable=True)
                return 77
            return 76  # Preflight-only committed-epoch interruption, bounded retry.
        artifacts = {}
        for path in [directory / "result.json", directory / "current.json", directory / "contract.json",
                     directory / "optimizer_groups.json", directory / payload["last"]["file"],
                     directory / payload["best_mask"]["file"]]:
            artifacts[path.relative_to(run).as_posix()] = sha256_file(path)
        atomic_json(output / "result.json", {"status": "SUCCEEDED", "job_id": job["id"],
                    "kind": job["kind"], "contract_sha256": spec["contract_sha256"],
                    "artifact_scope": "job", "artifacts": artifacts}, immutable=True)
        return 0
    atomic_json(output / "payload.json", payload, immutable=True)
    artifacts = {"payload.json": sha256_file(output / "payload.json")}
    for name in payload.get("output_files", []):
        artifacts[name] = sha256_file(inside(output, name))
    atomic_json(output / "result.json",
                {"status": "SUCCEEDED", "job_id": job["id"], "kind": job["kind"],
                 "contract_sha256": spec["contract_sha256"], "artifacts": artifacts}, immutable=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    try:
        return execute(args.spec, args.token)
    except Exception as exc:
        # A single real-job OOM is terminal, never silently shrink batch or repeat it.
        if type(exc).__name__ == "OutOfMemoryError":
            spec = read_json(args.spec)
            atomic_json(Path(spec["output"]) / "oom.json", {"job_id": spec["job"]["id"],
                        "contract_sha256": spec["contract_sha256"], "error": str(exc)}, immutable=True)
            return 78
        raise


if __name__ == "__main__":
    raise SystemExit(main())
