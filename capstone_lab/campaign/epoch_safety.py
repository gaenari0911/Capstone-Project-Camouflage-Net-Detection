"""Bounded real-GPU pause/resume acceptance test; not a formal experiment."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from capstone_lab.errors import ArtifactError
from .contracts import output_path, read_json
from .io import atomic_json
from .supervisor import launch, snapshot


def await_state(run, predicate, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = snapshot(run)
        if predicate(state):
            return state
        if state["status"] in {"PARTIAL_FAILURE", "FAILED_SUPERVISOR", "BLOCKED_RESOURCES"}:
            raise ArtifactError(str(state))
        time.sleep(.2)
    raise ArtifactError("epoch safety verification timed out; inspect owned processes, do not duplicate")


def verify(root, output):
    if output.exists():
        raise ArtifactError("new safety preflight output required")
    output.mkdir(parents=True)
    data = read_json(root / "configs/campaigns/s8_epoch_dispatch_v1.json")
    data["output"] = (output / "run").relative_to(root).as_posix()
    data["jobs"] = [data["jobs"][0]]
    data["jobs"][0]["training"]["stop_after_epoch"] = 0
    path = output / "config.json"
    atomic_json(path, data, immutable=True)
    run = output / "run"
    launch(root, path)
    await_state(run, lambda s: s["counts"].get("RUNNING") == 1)
    token = read_json(run / "token.json")["token"]
    atomic_json(run / "control.json", {"token": token, "action": "pause"})
    paused = await_state(run, lambda s: s["status"] == "PAUSED" and not s.get("supervisor_alive"))
    job = paused["jobs"][0]
    if job.get("paused_epoch") != 1 or job.get("safe_pauses") != 1:
        raise ArtifactError("real worker did not pause at first committed epoch")
    launch(root, path)
    completed = await_state(run, lambda s: s["status"] == "SUCCEEDED" and not s.get("supervisor_alive"))
    result = read_json(run / "jobs/epoch_H2/training_state/result.json")
    if result["resumed_from_epoch"] != 1 or completed["jobs"][0]["attempts"] != 2:
        raise ArtifactError("safe pause did not resume from epoch 1")
    summary = {"status": "VERIFIED", "scope": "real_epoch_safe_pause_not_formal_training",
               "paused_epoch": 1, "resumed_from_epoch": 1, "attempts": 2,
               "safe_pauses_not_failure_budget": completed["jobs"][0]["safe_pauses"],
               "state_sha256": result["state_sha256"], "contract": read_json(run / "contract.json")}
    atomic_json(output / "summary.json", summary, immutable=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    print(verify(root, output_path(root, args.output))["status"])
