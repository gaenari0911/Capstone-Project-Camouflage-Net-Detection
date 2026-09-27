from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import nullcontext
from collections import Counter
from pathlib import Path

import psutil

from capstone_lab.config import sha256_file
from capstone_lab.errors import ArtifactError, StateError
from .contracts import contract, digest, inside, output_path, read_json
from .io import RunLock, atomic_json
from .detached import spawn as spawn_detached
from .seed_policy import load_queue_policy, cancel_pending
from .resources import admission, budget_roots

_LAUNCHED = []


def directory_bytes(run):
    total = 0
    for path in run.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # Workers atomically rename temporary files during live budget scans.
            # This tolerance is ONLY for storage accounting, never artifact validation.
            continue
    return total


def identity(pid, token, module):
    process = psutil.Process(pid)
    command = process.cmdline()
    if token not in command or module not in command:
        raise StateError("process token/module ownership mismatch")
    return {"pid": pid, "created": process.create_time(), "token": token, "module": module}


def owned_process(record):
    try:
        process = psutil.Process(record["pid"])
        if abs(process.create_time() - record["created"]) > .001:
            return None
        command = process.cmdline()
        if record["token"] not in command or record["module"] not in command:
            return None
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except psutil.NoSuchProcess:
        return None
    except psutil.AccessDenied as exc:
        raise StateError("cannot establish process ownership; refusing duplicate or termination") from exc


def stop_owned(record):
    process = owned_process(record)
    if process is None:
        raise StateError("PID/creation time/token/module ownership is not valid")
    process.terminate()
    try:
        process.wait(timeout=10)
    except psutil.TimeoutExpired:
        raise StateError("owned process did not exit; no broad process kill performed")


def open_db(run):
    db = sqlite3.connect(run / "state.sqlite", timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, record TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, created REAL, kind TEXT, detail TEXT)")
    db.commit()
    return db


def put(db, record):
    db.execute("INSERT OR REPLACE INTO jobs VALUES(?,?)", (record["id"], json.dumps(record)))
    db.commit()


def event(db, kind, detail):
    db.execute("INSERT INTO events(created,kind,detail) VALUES(?,?,?)", (time.time(), kind, json.dumps(detail)))
    db.commit()


def rows(db):
    return {key: json.loads(raw) for key, raw in db.execute("SELECT id,record FROM jobs ORDER BY id")}


def verify_result(run, job, binding, record):
    path = inside(run, record["result"])
    data = read_json(path)
    if data.get("contract_sha256") != binding or data.get("job_id") != job["id"] or data.get("kind") != job["kind"] or data.get("status") != "SUCCEEDED":
        raise ArtifactError("worker result identity mismatch")
    if not isinstance(data.get("artifacts"), dict) or not data["artifacts"]:
        raise ArtifactError("empty completion artifacts")
    for relative, expected in data["artifacts"].items():
        if data.get("artifact_scope") == "job" and job["kind"] == "epoch_train":
            item = inside(run, relative)
            if not item.is_relative_to(run / "jobs" / job["id"] / "training_state"):
                raise ArtifactError("epoch artifact escapes own training state")
        else:
            item = inside(path.parent, relative)
        if sha256_file(item) != expected:
            raise ArtifactError(f"completed artifact changed: {item}")
    current = sha256_file(path)
    if record.get("result_sha256") and record["result_sha256"] != current:
        raise ArtifactError("completed marker changed")
    return current


def snapshot(run):
    db_path = run / "state.sqlite"
    result = {"run": str(run), "status": "NOT_STARTED", "counts": {}, "jobs": []}
    if (run / "status.json").exists():
        result.update(read_json(run / "status.json"))
    if db_path.exists():
        db = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True, timeout=30)
        try:
            exists = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'").fetchone()
            result["jobs"] = [json.loads(r[0]) for r in db.execute("SELECT record FROM jobs ORDER BY id")] if exists else []
            result["counts"] = dict(Counter(j["state"] for j in result["jobs"]))
        finally:
            db.close()
    if (run / "owner.json").exists():
        result["supervisor_alive"] = owned_process(read_json(run / "owner.json")) is not None
        if result.get("status") == "RUNNING" and not result["supervisor_alive"]:
            result["status"] = "INTERRUPTED_RESUME_REQUIRED"
    for job in result["jobs"]:
        directory = run / "jobs" / job["id"] / "training_state"
        if (directory / "progress.json").exists():
            job["training_progress"] = read_json(directory / "progress.json")
        if (directory / "current.json").exists():
            pointer = read_json(directory / "current.json")
            job["committed_epoch"] = pointer["epoch"]
            job["best_mask"] = pointer["best_mask"]
    return result


def launch(root, config_path):
    config = read_json(config_path)
    run = output_path(root, config["output"])
    run.mkdir(parents=True, exist_ok=True)
    with RunLock(run / "launch.lock"):
        if (run / "owner.json").exists() and owned_process(read_json(run / "owner.json")) is not None:
            return snapshot(run)
        runtime = run / "runtime"
        if not (run / "contract.json").exists():
            config, binding = contract(root, config_path)
            for relative, expected in binding["code"].items():
                destination = inside(runtime, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if sha256_file(destination) != expected:
                        raise ArtifactError("partially prepared runtime differs; use a new run")
                else:
                    shutil.copyfile(inside(root, relative), destination)
                if sha256_file(destination) != expected:
                    raise ArtifactError("runtime copy integrity failure")
        config, binding = contract(root, config_path, runtime)
        # Validate all frozen identities BEFORE creating a replacement process.
        atomic_json(run / "contract.json", binding, immutable=True)
        atomic_json(run / "config.json", config, immutable=True)
        known_jobs = {j["id"]: j for j in config["jobs"]}
        for record in snapshot(run)["jobs"]:
            if record["state"] == "SUCCEEDED":
                verify_result(run, known_jobs[record["id"]], digest(binding), record)
        token_file = run / "token.json"
        if not token_file.exists():
            atomic_json(token_file, {"token": uuid.uuid4().hex}, immutable=True)
        token = read_json(token_file)["token"]
        atomic_json(run / "control.json", {"token": token, "action": "run"})
        log_dir = run / "launches" / uuid.uuid4().hex
        log_dir.mkdir(parents=True)
        command = [sys.executable, "-m", "capstone_lab.campaign", "serve", "--config",
                   str(config_path.resolve()), "--project-root", str(root), "--token", token]
        policy = load_queue_policy(root, run, config)
        coordinator_cwd = root if policy else runtime
        if policy:
            from .contracts import code_records
            atomic_json(log_dir / "coordinator_revision.json", {"code": code_records(root),
                        "queue_policy": policy, "worker_runtime": str(runtime),
                        "worker_contract_sha256": digest(binding)}, immutable=True)
        with (log_dir / "stdout.log").open("xb") as out, (log_dir / "stderr.log").open("xb") as err:
            process = spawn_detached(command, cwd=coordinator_cwd, stdout=out, stderr=err)
        _LAUNCHED[:] = [p for p in _LAUNCHED if p.poll() is None]
        _LAUNCHED.append(process)
        # Child records its own ownership after acquiring the OS run lock.
        for _ in range(100):
            if process.poll() is not None:
                raise StateError(f"supervisor exited {process.returncode}; inspect {log_dir}")
            if (run / "owner.json").exists():
                owner = read_json(run / "owner.json")
                if owner["pid"] == process.pid and owned_process(owner):
                    return {"status": "STARTED", "run": str(run), "owner": owner,
                            "logs": str(log_dir), "ssh_disconnect_verified": False}
            time.sleep(.05)
        raise StateError(f"supervisor handshake timeout; inspect {log_dir}; do not blindly launch twice")


def supervise(root, config_path, token):
    initial_config = read_json(config_path)
    run = output_path(root, initial_config["output"])
    policy = load_queue_policy(root, run, initial_config)
    code_root = run / "runtime" if policy else Path(__file__).resolve().parents[2]
    config, binding = contract(root, config_path, code_root)
    expected_hash = digest(binding)
    with RunLock(run / "supervisor.lock"), (RunLock(root / "artifacts/s8_formal_campaign.lock") if config["mode"] == "s8_real_campaign" else nullcontext()):
        atomic_json(run / "contract.json", binding, immutable=True)
        atomic_json(run / "config.json", config, immutable=True)
        if read_json(run / "token.json")["token"] != token:
            raise StateError("invalid campaign token")
        if config["mode"] == "s8_real_campaign":
            ledger_path = root / "artifacts/s8_formal_ledger/claimed_jobs.json"
            ledger = read_json(ledger_path) if ledger_path.exists() else {}
            for job in config["jobs"]:
                claim = {"run": config["output"], "config_sha256": binding["config_sha256"]}
                if job["id"] in ledger and ledger[job["id"]] != claim:
                    raise StateError("formal job already claimed by another frozen run; do not retrain it")
                ledger[job["id"]] = claim
            atomic_json(ledger_path, ledger)
        atomic_json(run / "owner.json", identity(os.getpid(), token, "capstone_lab.campaign"))
        db = open_db(run)
        jobs = {j["id"]: j for j in config["jobs"]}
        current = rows(db)
        for name in jobs:
            if name not in current:
                put(db, {"id": name, "state": "PLANNED", "attempts": 0})
        cancel_pending(db, policy)
        event(db, "SUPERVISOR_STARTED", {"pid": os.getpid(), "contract": expected_hash})
        # Integrity failure is fail-closed, never overwrite completed research artifacts.
        for name, record in rows(db).items():
            if record["state"] == "SUCCEEDED":
                verify_result(run, jobs[name], expected_hash, record)
        active_seconds = read_json(run / "status.json").get("active_seconds", 0) if (run / "status.json").exists() else 0
        prior_active_seconds = 0.
        if config["mode"] == "s8_real_campaign":
            synthesis_ledger = root / 'artifacts/s8_heuristic_formal/run_v1/active_time_ledger.json'
            if synthesis_ledger.exists():
                prior_active_seconds += read_json(synthesis_ledger)['formal_active_seconds']
            for prior in (root / "artifacts").glob("s8_*/*/config.json"):
                if prior.parent != run and read_json(prior).get("mode") == "s8_real_campaign":
                    previous_status = prior.parent / "status.json"
                    if previous_status.exists():
                        prior_active_seconds += read_json(previous_status).get("active_seconds", 0.)
        last_tick = time.monotonic()
        last_saved = 0.
        last_budget_scan = 0.
        budget_used = 0
        status = "RUNNING"
        children = {}
        try:
            while True:
                now = time.monotonic()
                active_seconds += now - last_tick
                last_tick = now
                current = rows(db)
                # Reattach surviving workers after coordinator death; don't duplicate them.
                for name, record in current.items():
                    if record["state"] != "RUNNING":
                        continue
                    if name in children:
                        live = children[name].poll() is None
                    else:
                        live = owned_process(record["owner"]) is not None
                    if live:
                        continue
                    attempt_dir = inside(run, record["result"]).parent
                    if jobs[name]["kind"] == "epoch_train" and (attempt_dir / "safe_pause.json").exists():
                        marker = read_json(attempt_dir / "safe_pause.json")
                        if marker.get("contract_sha256") != expected_hash or marker.get("job_id") != name:
                            raise ArtifactError("invalid safe epoch pause marker")
                        record.update(state="RETRY_WAIT", paused_epoch=marker["epoch"],
                                      safe_pauses=record.get("safe_pauses", 0) + 1)
                        put(db, record)
                        children.pop(name, None)
                        event(db, "EPOCH_PAUSED", {"id": name, "epoch": marker["epoch"]})
                        continue
                    try:
                        if name in children and children[name].returncode != 0:
                            raise ArtifactError(f"worker exited {children[name].returncode}")
                        result_hash = verify_result(run, jobs[name], expected_hash, record)
                        record.update(state="SUCCEEDED", result_sha256=result_hash, finished=time.time())
                        event(db, "JOB_SUCCEEDED", {"id": name, "attempt": record["attempts"]})
                    except (OSError, ValueError, ArtifactError, KeyError) as exc:
                        terminal_oom = jobs[name]["kind"] == "epoch_train" and (attempt_dir / "oom.json").exists()
                        state = "RETRY_WAIT" if not terminal_oom and record["attempts"] - record.get("safe_pauses", 0) < jobs[name]["max_attempts"] else "FAILED"
                        record.update(state=state, error=str(exc), finished=time.time())
                        event(db, "JOB_FAILED", {"id": name, "state": state, "error": str(exc)})
                    put(db, record)
                    children.pop(name, None)
                current = rows(db)
                control = read_json(run / "control.json")
                if control.get("token") != token or control.get("action") not in {"run", "pause"}:
                    raise StateError("invalid control token/action")
                pause = control["action"] == "pause" or active_seconds + prior_active_seconds >= config["active_seconds_max"]
                if now - last_budget_scan >= 5:
                    budget_used = sum(directory_bytes(path) for path in budget_roots(root)) if config["mode"] == "s8_real_campaign" else directory_bytes(run)
                    last_budget_scan = now
                checkpoint_margin = 2**30 if any(j["kind"] == "epoch_train" for j in jobs.values()) else 0
                disk_block = (shutil.disk_usage(run).free < config["min_free_gib"] * 2**30 + checkpoint_margin
                              or budget_used + checkpoint_margin >= config["disk_budget_gib"] * 2**30)
                resource_block = (disk_block
                                  or psutil.virtual_memory().available < 4 * 2**30)
                atomic_json(run / "epoch_pause.json", {"pause": pause or disk_block, "token": token,
                            "active_seconds": active_seconds, "budget_bytes": budget_used}) if now - last_saved > .5 else None
                if not pause and not resource_block:
                    for name, job in jobs.items():
                        record = current[name]
                        if record["state"] not in {"PLANNED", "RETRY_WAIT"}:
                            continue
                        dep_states = [current[d]["state"] for d in job["dependencies"]]
                        if any(s in {"FAILED", "BLOCKED_DEPENDENCY", "CANCELLED_BY_USER"} for s in dep_states):
                            record.update(state="BLOCKED_DEPENDENCY")
                            put(db, record)
                            continue
                        if any(s != "SUCCEEDED" for s in dep_states):
                            continue
                        running = [r for r in current.values() if r["state"] == "RUNNING"]
                        cpu = sum(jobs[r["id"]]["cpu"] for r in running)
                        if len(running) >= config["max_parallel"] or cpu + job["cpu"] > config["cpu_slots"]:
                            continue
                        admitted, reason = admission(job, running, jobs)
                        if not admitted:
                            continue
                        # No pending dispatch means no code reload: an already running
                        # SSH probe may finish while independent source files are developed.
                        _, fresh = contract(root, config_path, run / "runtime")
                        if fresh != binding:
                            raise StateError("config/input/code/approval changed before dispatch")
                        attempt = record["attempts"] + 1
                        if attempt - record.get("safe_pauses", 0) > job["max_attempts"]:
                            record.update(state="FAILED", error="attempt budget exhausted")
                            put(db, record)
                            continue
                        directory = run / "jobs" / name / f"attempt_{attempt:03d}"
                        # Prior partially prepared launch is preserved, never overwritten.
                        if directory.exists():
                            directory = run / "jobs" / name / f"attempt_{attempt:03d}_{uuid.uuid4().hex[:8]}"
                        directory.mkdir(parents=True)
                        spec = {"root": str(root), "run": str(run), "job": job, "attempt": attempt,
                                "contract_sha256": expected_hash, "output": str(directory), "token": token}
                        atomic_json(directory / "spec.json", spec, immutable=True)
                        command = [sys.executable, "-m", "capstone_lab.campaign.worker", "--spec",
                                   str(directory / "spec.json"), "--token", token]
                        env = os.environ.copy()
                        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                        with (directory / "stdout.log").open("xb") as out, (directory / "stderr.log").open("xb") as err:
                            child = subprocess.Popen(command, cwd=run / "runtime", stdin=subprocess.DEVNULL,
                                                     stdout=out, stderr=err, shell=False, env=env,
                                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        # The worker cannot do work until this durable start handshake exists.
                        owner = identity(child.pid, token, "capstone_lab.campaign.worker")
                        record.update(state="RUNNING", attempts=attempt, owner=owner,
                                      result=(directory / "result.json").relative_to(run).as_posix(),
                                      started=time.time(), error=None)
                        put(db, record)
                        atomic_json(directory / "dispatch.json", {"owner": owner, "contract_sha256": expected_hash}, immutable=True)
                        children[name] = child
                        event(db, "JOB_STARTED", {"id": name, "attempt": attempt, "cpu_reserved": cpu + job["cpu"],
                                                 "cpu_limit": config["cpu_slots"], "owner": owner})
                current = rows(db)
                counts = Counter(r["state"] for r in current.values())
                active = counts["RUNNING"]
                if counts["SUCCEEDED"] + counts["CANCELLED_BY_USER"] == len(jobs):
                    status = "SUCCEEDED"
                elif not active and pause:
                    status = "PAUSED"
                elif not active and resource_block:
                    status = "BLOCKED_RESOURCES"
                elif not active and all(r["state"] in {"SUCCEEDED", "FAILED", "BLOCKED_DEPENDENCY", "CANCELLED_BY_USER"} for r in current.values()):
                    status = "PARTIAL_FAILURE"
                if now - last_saved > .5 or status != "RUNNING":
                    atomic_json(run / "status.json", {"status": status, "scope": config["mode"],
                                "counts": dict(counts), "active_seconds": active_seconds,
                                "budget_bytes": budget_used, "gpu_concurrency_effective": 1,
                                "prior_campaign_active_seconds": prior_active_seconds,
                                "contract_sha256": expected_hash, "updated": time.time(),
                                "ssh_disconnect": "BLOCKED_USER_ACTION_UNTESTED"})
                    last_saved = now
                if status != "RUNNING":
                    break
                day = int(active_seconds // 86400)
                if day and not (run / "daily_reports" / f"day_{day:03d}.json").exists():
                    atomic_json(run / "daily_reports" / f"day_{day:03d}.json", snapshot(run), immutable=True)
                time.sleep(.05)
            event(db, "SUPERVISOR_FINISHED", {"status": status})
        except Exception as exc:
            atomic_json(run / "status.json", {"status": "FAILED_SUPERVISOR", "error": str(exc),
                                             "active_seconds": active_seconds})
            event(db, "SUPERVISOR_ERROR", {"error": str(exc)})
            raise
        finally:
            db.close()
        result = snapshot(run)
        atomic_json(run / "report.json", result)
        return result
