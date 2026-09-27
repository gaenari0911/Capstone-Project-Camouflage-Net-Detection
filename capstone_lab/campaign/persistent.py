"""Launch a campaign watchdog outside the SSH/terminal Windows job hierarchy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import psutil

from .contracts import output_path, read_json
from .detached import job_info, spawn
from .io import RunLock, atomic_json


def launch_persistent(root, config_path, hotfix=None):
    run = output_path(root, read_json(config_path)["output"])
    run.mkdir(parents=True, exist_ok=True)
    with RunLock(run / "watchdog_launch.lock"):
        owner_path = run / "watchdog_owner.json"
        if owner_path.exists():
            old = read_json(owner_path)
            try:
                process = psutil.Process(old["pid"])
                if (abs(process.create_time() - old.get("created", old["started"])) < 1
                        and "capstone_lab.campaign.watchdog" in process.cmdline()
                        and str(config_path) in process.cmdline()):
                    if old.get("job_object", {}).get("in_job", True):
                        raise RuntimeError("a session-owned watchdog is running; safely migrate it first")
                    return {"status": "ALREADY_RUNNING", "owner": old}
            except psutil.NoSuchProcess:
                pass
        env = os.environ.copy()
        env.update(PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")
        shim_hash = None
        if hotfix:
            manifest = read_json(run / "runtime_hotfix_manifest_v1.json")
            shim = hotfix.resolve() / "sitecustomize.py"
            if shim != (root / manifest["sitecustomize"]).resolve():
                raise RuntimeError("hotfix location differs from recorded deployment")
            shim_hash = hashlib.sha256(shim.read_bytes()).hexdigest()
            if shim_hash != manifest["sitecustomize_sha256"]:
                raise RuntimeError("atomic retry shim hash changed")
            env.update(PYTHONPATH=str(shim.parent), CAPSTONE_ATOMIC_RETRY_ROOT=str(run.resolve()),
                       CAPSTONE_ATOMIC_RETRY_REQUIRED="1")
        elif (run / "runtime_hotfix_manifest_v1.json").exists():
            raise RuntimeError("this run requires its recorded --hotfix")
        command = [sys.executable, "-u", "-X", "faulthandler", "-m", "capstone_lab.campaign.watchdog",
                   "--project-root", str(root), "--config", str(config_path)]
        logs = run / "watchdog_launches" / uuid.uuid4().hex
        logs.mkdir(parents=True)
        atomic_json(logs / "launch.json", {"time": time.time(), "parent_job": job_info(),
                    "command": command, "hotfix_sha256": shim_hash,
                    "policy": "CREATE_BREAKAWAY_FROM_JOB; no session-owned fallback"}, immutable=True)
        with (logs / "stdout.log").open("xb") as out, (logs / "stderr.log").open("xb") as err:
            child = spawn(command, cwd=root, stdout=out, stderr=err, env=env)
        created = psutil.Process(child.pid).create_time()
        for _ in range(150):
            if child.poll() is not None:
                raise RuntimeError(f"watchdog exited {child.returncode}; inspect {logs}")
            if owner_path.exists():
                owner = read_json(owner_path)
                if owner.get("pid") == child.pid and abs(owner.get("created", 0) - created) < .001:
                    if owner["job_object"]["in_job"]:
                        raise RuntimeError("watchdog failed to escape Windows job hierarchy")
                    result = {"status": "STARTED", "owner": owner, "logs": str(logs)}
                    atomic_json(logs / "started.json", result, immutable=True)
                    return result
            time.sleep(.1)
        raise RuntimeError(f"watchdog handshake timeout; inspect {logs} before retrying")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hotfix", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = (root / args.config).resolve()
    hotfix = (root / args.hotfix).resolve() if args.hotfix else None
    print(json.dumps(launch_persistent(root, config, hotfix), indent=2))
