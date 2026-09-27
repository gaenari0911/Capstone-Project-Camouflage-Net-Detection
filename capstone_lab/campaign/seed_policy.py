"""Explicit schedule amendment; never changes a worker's training contract."""
import json
import re
import time
from pathlib import Path

from capstone_lab.config import sha256_file
from capstone_lab.errors import StateError

AMENDMENT = "configs/approvals/s8_seed_amendment_v1.json"


def excluded(name):
    return bool(re.fullmatch(r"[AH][0-9]+(?:_[RS])?_seed[12]", name))


def amendment(root):
    path = Path(root) / AMENDMENT
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "APPROVED_BY_USER" or value.get("seeds") != {"A": [0], "M": [0, 1, 2], "L": [0, 1, 2], "H": [0]}:
        raise StateError("invalid seed amendment")
    return value, sha256_file(path)


def load_queue_policy(root, run, config):
    path = run / "queue_policy.json"
    if not path.exists():
        return None
    policy = json.loads(path.read_text(encoding="utf-8"))
    _, expected = amendment(root)
    wanted = sorted(j["id"] for j in config["jobs"] if excluded(j["id"]))
    if (policy.get("amendment") != AMENDMENT or policy.get("amendment_sha256") != expected
            or policy.get("excluded_jobs") != wanted):
        raise StateError("queue amendment identity mismatch")
    return policy


def cancel_pending(db, policy):
    if not policy:
        return
    records = {name: json.loads(raw) for name, raw in db.execute("SELECT id,record FROM jobs")}
    pending = []
    for name in policy["excluded_jobs"]:
        row = records[name]
        if row["state"] == "CANCELLED_BY_USER" and row.get("amendment_sha256") == policy["amendment_sha256"]:
            continue
        if row["state"] != "PLANNED" or row["attempts"] != 0:
            raise StateError(f"refusing to cancel already-started job: {name}")
        row.update(state="CANCELLED_BY_USER", amendment_sha256=policy["amendment_sha256"],
                   reason="A/H ablation restricted to seed0 by user", cancelled=time.time())
        pending.append((name, row))
    with db:
        for name, row in pending:
            db.execute("UPDATE jobs SET record=? WHERE id=?", (json.dumps(row), name))
            db.execute("INSERT INTO events(created,kind,detail) VALUES(?,?,?)",
                       (time.time(), "JOB_CANCELLED_BY_USER", json.dumps(row)))


def activate(root, config_path):
    import psutil
    from .contracts import contract, digest, output_path, read_json
    from .io import RunLock, atomic_json
    from .supervisor import snapshot, owned_process, stop_owned
    from .persistent import launch_persistent
    config = read_json(config_path)
    run = output_path(root, config["output"])
    _, binding = contract(root, config_path, run / "runtime")
    _, amendment_hash = amendment(root)
    policy = {"amendment": AMENDMENT, "amendment_sha256": amendment_hash,
              "excluded_jobs": sorted(j["id"] for j in config["jobs"] if excluded(j["id"]))}
    before = snapshot(run)
    for row in before["jobs"]:
        if row["id"] in policy["excluded_jobs"] and not (
                row["state"] == "PLANNED" and row["attempts"] == 0 or
                row["state"] == "CANCELLED_BY_USER" and row.get("amendment_sha256") == amendment_hash):
            raise StateError(f"excluded job already started: {row['id']}")
    audit = run / "seed_amendment_migration_v1.json"
    if not audit.exists():
        atomic_json(audit, {"before": before, "policy": policy,
                    "unchanged_worker_contract": digest(binding)}, immutable=True)
    owner_path = run / "watchdog_owner.json"
    if owner_path.exists():
        owner = read_json(owner_path)
        try:
            process = psutil.Process(owner["pid"])
            if abs(process.create_time() - owner.get("created", 0)) < .001:
                command = process.cmdline()
                if "capstone_lab.campaign.watchdog" not in command or str(config_path) not in command:
                    raise StateError("watchdog ownership mismatch")
                process.terminate()
                process.wait(timeout=10)
        except psutil.NoSuchProcess:
            pass
    owner = read_json(run / "owner.json")
    if owned_process(owner):
        stop_owned(owner)
    with RunLock(run / "launch.lock"), RunLock(run / "supervisor.lock"):
        atomic_json(run / "queue_policy.json", policy, immutable=True)
    hotfix_manifest = read_json(run / "runtime_hotfix_manifest_v1.json")
    result = launch_persistent(root, config_path, (root / hotfix_manifest["sitecustomize"]).parent)
    result["preserved_workers"] = [{"id": row["id"], "pid": row["owner"]["pid"],
                                   "alive": owned_process(row["owner"]) is not None}
                                  for row in before["jobs"] if row["state"] == "RUNNING"]
    result["excluded_jobs"] = policy["excluded_jobs"]
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    print(json.dumps(activate(root, (root / args.config).resolve()), indent=2))
