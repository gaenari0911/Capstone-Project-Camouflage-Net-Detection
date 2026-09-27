"""Restart an immutable run's owned supervisor without reading live JSON."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from .contracts import output_path, read_json
from .io import RunLock, atomic_json
from .supervisor import launch, owned_process
from .detached import job_info
import psutil


def database_counts(run):
    path = run / "state.sqlite"
    if not path.exists():
        return {}
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        records = [json.loads(row[0]) for row in db.execute("SELECT record FROM jobs")]
        return dict(Counter(record["state"] for record in records))
    finally:
        db.close()


def watch(root, config_path, interval):
    config = read_json(config_path)
    run = output_path(root, config["output"])
    if os.environ.get("CAPSTONE_ATOMIC_RETRY_REQUIRED") == "1":
        expected_root = str(run.resolve()).casefold()
        active_root = os.environ.get("CAPSTONE_ATOMIC_RETRY_ROOT", "").casefold()
        if os.environ.get("CAPSTONE_ATOMIC_RETRY_ACTIVE") != "1" or active_root != expected_root:
            raise RuntimeError("required scoped atomic retry shim is not active")
    with RunLock(run / "watchdog.lock"):
        job = job_info()
        if job["in_job"]:
            raise RuntimeError("watchdog must be launched outside Windows session job objects")
        atomic_json(run / "watchdog_owner.json", {"pid": os.getpid(), "started": time.time(),
                    "created": psutil.Process().create_time(), "job_object": job,
                    "config": str(config_path), "policy": "restart_owned_supervisor_only_no_live_status_reads"})
        restarts = 0
        errors = 0
        while True:
            try:
                counts = database_counts(run)
                if counts and sum(counts.values()) == counts.get("SUCCEEDED", 0) + counts.get("CANCELLED_BY_USER", 0):
                    atomic_json(run / "watchdog_status.json", {"status": "CAMPAIGN_SUCCEEDED", "counts": counts,
                                "restarts": restarts, "errors": errors, "updated": time.time()})
                    return 0
                terminal = counts and not counts.get("RUNNING", 0) and all(
                    key in {"SUCCEEDED", "FAILED", "BLOCKED_DEPENDENCY", "CANCELLED_BY_USER"} for key in counts)
                if terminal:
                    atomic_json(run / "watchdog_status.json", {"status": "CAMPAIGN_TERMINAL", "counts": counts,
                                "restarts": restarts, "errors": errors, "updated": time.time()})
                    return 0
                owner = read_json(run / "owner.json") if (run / "owner.json").exists() else None
                if owner is None or owned_process(owner) is None:
                    control = read_json(run / "control.json") if (run / "control.json").exists() else {}
                    status = read_json(run / "status.json") if (run / "status.json").exists() else {}
                    if control.get("action") == "pause" or status.get("status") in {"PAUSED", "BLOCKED_RESOURCES"}:
                        atomic_json(run / "watchdog_status.json", {"status": "STOPPED_AT_GATE", "updated": time.time()})
                        return 0
                    result = launch(root, config_path)
                    restarts += 1
                    atomic_json(run / "watchdog_status.json", {"status": "SUPERVISOR_RESTARTED",
                                "counts": counts, "restarts": restarts, "errors": errors,
                                "owner": result.get("owner"), "updated": time.time()})
                else:
                    atomic_json(run / "watchdog_status.json", {"status": "MONITORING", "counts": counts,
                                "restarts": restarts, "errors": errors, "owner": owner,
                                "updated": time.time(), "job_object": job})
            except Exception as exc:
                errors += 1
                print(f"watchdog error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                try:
                    atomic_json(run / "watchdog_status.json", {"status": "RETRYING_AFTER_ERROR",
                                "restarts": restarts, "errors": errors,
                                "error": f"{type(exc).__name__}: {exc}", "updated": time.time()})
                except Exception:
                    pass
            time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    raise SystemExit(watch(root, config.resolve(), max(.5, args.interval)))
