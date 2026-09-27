from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
from pathlib import Path

from capstone_lab.config import canonical_json, sha256_file
from capstone_lab.errors import ConfigError

APPROVAL_PATH = "configs/approvals/s7_delegated_decisions_v1.json"
APPROVAL_SHA256 = "21da4972d061371f239078b0e229d1eda393a1bd2e3242b03e6e447545095154"
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
WORKERS = {"fixture", "source_audit", "epoch_train"}
KINDS = {"TRAIN", "REFERENCE", "VALIDATION_AGGREGATE", "FROZEN_MANIFEST",
         "MANUAL_FREEZE", "APPROVAL", "CPU_SYNTHESIS", "GENERATIVE_POOL",
         "JUDGE_FREEZE", "SELECTION_MANIFEST", "SYNTHESIS_VALIDATION"}


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path):
    deadline = time.monotonic() + 5.0
    delay = .005
    while True:
        try:
            content = Path(path).read_text(encoding="utf-8")
            break
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(.2, delay * 2)
    # Parse errors, missing paths, and hash mismatches are not transient access errors.
    return json.loads(content)


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ConfigError(f"path escapes project: {relative}")
    return path


def output_path(root, relative):
    path = inside(root, relative)
    parts = path.relative_to(root.resolve()).parts
    if len(parts) < 3 or parts[0] != "artifacts" or not parts[1].startswith("s8_"):
        raise ConfigError("output must be a run below artifacts/s8_*/")
    return path


def code_records(root):
    paths = sorted((root / "capstone_lab").rglob("*.py"))
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}


def approved_plan(root):
    approval_file = root / APPROVAL_PATH
    if sha256_file(approval_file) != APPROVAL_SHA256:
        raise ConfigError("approval is absent or changed; a new explicit authorization binding is required")
    approval = read_json(approval_file)
    if approval["status"] != "APPROVED_BY_USER_DELEGATION" or [d["id"] for d in approval["decisions"]] != list(range(1, 14)):
        raise ConfigError("invalid delegated approval")
    records = {APPROVAL_PATH: APPROVAL_SHA256}
    for ref in approval["references"].values():
        expected = ref.get("sha256", ref.get("file_sha256"))
        if sha256_file(inside(root, ref["path"])) != expected:
            raise ConfigError(f"approval reference changed: {ref['path']}")
        records[ref["path"]] = expected
    plan = read_json(root / approval["references"]["s7_summary"]["path"])
    recipe = read_json(root / approval["references"]["proposed_recipe"]["path"])
    for key, relative in recipe["inputs"].items():
        expected = recipe["expected_input_hashes"][key]
        if sha256_file(inside(root, relative)) != expected:
            raise ConfigError(f"S7 input changed: {relative}")
        records[relative] = expected
    for relative in ("Build_Object_Pool_From_Txt_ObjectBased.py", "object_pool_metadata.csv"):
        records[relative] = sha256_file(root / relative)
    if plan["status"] != "VERIFIED":
        raise ConfigError("S7 is not verified")
    plan_dir = (root / approval["references"]["s7_summary"]["path"]).parent
    artifact_manifest = read_json(plan_dir / "artifact_manifest.json")
    if artifact_manifest["contract"] != plan["contract"]:
        raise ConfigError("S7 artifact contract mismatch")
    for name, expected in artifact_manifest["artifacts"].items():
        path = inside(plan_dir, name)
        if sha256_file(path) != expected:
            raise ConfigError(f"S7 artifact changed: {name}")
        records[path.relative_to(root).as_posix()] = expected
    dag = read_json(plan_dir / "dag_v1.json")
    registry = list(csv.DictReader(io.StringIO((plan_dir / "registry_v1.csv").read_text(encoding="utf-8"))))
    if dag != plan["dag"] or len(registry) != 60:
        raise ConfigError("S7 registry/DAG disagree")
    training = [r for r in registry if r["node_kind"] == "TRAIN"]
    if len(training) != 54 or sum(int(r["epochs"]) for r in training) != 6120:
        raise ConfigError("training scope exceeds approved 54 jobs/6120 epochs")
    for node in dag["nodes"]:
        if node["kind"] not in KINDS:
            raise ConfigError(f"unknown S7 node kind: {node['kind']}")
    return approval, plan, records


def readiness_plan(root):
    approval, plan, records = approved_plan(root)
    nodes = json.loads(json.dumps(plan["dag"]["nodes"]))
    for node in nodes:
        node["execution_status"] = "PENDING_IMPLEMENTATION_AND_PREFLIGHT"
        name = node["node_id"]
        if node["kind"] == "TRAIN" and not name.endswith("_seed0"):
            node["dependencies"].append(name.rsplit("_seed", 1)[0] + "_seed0")
        if node["kind"] == "CPU_SYNTHESIS":
            node["dependencies"].append("verify_low187_sources")
        if name == "gate_generative_pilot":
            node["dependencies"] = ["validate_anydoor_pilot100"]
            node["execution_status"] = "WAITING_ACTUAL_PILOT_REVIEW"
        if name == "generate_generative_pool_6000":
            node["dependencies"].append("verify_low187_sources")
    nodes.extend([
        {"node_id": "verify_low187_sources", "kind": "SOURCE_PROVENANCE", "dependencies": ["data_low187"], "execution_status": "PENDING_SOURCE_AUDIT"},
        {"node_id": "generate_anydoor_pilot100", "kind": "ANYDOOR_PILOT", "dependencies": ["aggregate_A", "verify_low187_sources", "gate_campaign_execution"], "execution_status": "APPROVED_SCOPE_PENDING_IMPLEMENTATION"},
        {"node_id": "validate_anydoor_pilot100", "kind": "PILOT_VALIDATION", "dependencies": ["generate_anydoor_pilot100"], "execution_status": "PENDING_IMPLEMENTATION"},
    ])
    from .seed_policy import amendment, excluded, AMENDMENT
    revised, amendment_hash = amendment(root)
    removed = {n["node_id"] for n in nodes if excluded(n["node_id"])}
    nodes = [n for n in nodes if n["node_id"] not in removed]
    for node in nodes:
        node["dependencies"] = [d for d in node["dependencies"] if d not in removed]
    registry = [r for r in plan["registry"] if not excluded(r["job_id"])]
    records[AMENDMENT] = amendment_hash
    from .foreground_policy import amend_dag
    nodes, registry, source_records = amend_dag(root, nodes, registry)
    records.update(source_records)
    return {"schema_version": 1, "status": "APPROVED_SCOPE_NOT_EXECUTABLE",
            "approval_sha256": APPROVAL_SHA256, "reference_hashes": records,
            "training_jobs": 34, "epoch_job_units": 4440, "formal_images": 33000,
            "heuristic_images": 21000, "generative_images": 12000,
            "pilot_images": 100, "test_unique_checkpoints": 28, "test_logical_cells": 30,
            "seed_policy": revised, "nodes": nodes, "registry": registry,
            "ssh_disconnect": "BLOCKED_USER_ACTION_UNTESTED",
            "remaining": ["formal train/val loader and validator", "H0/H1/H2 real-data preflight",
                          "real-loader resource profile", "formal synthesis adapter",
                          "AnyDoor compatibility/pilot review", "selection and final Test freeze"]}


def validate_config(root, data):
    allowed = {"schema_version", "mode", "output", "cpu_slots", "max_parallel",
               "jobs", "inputs", "disk_budget_gib", "min_free_gib", "active_seconds_max"}
    if set(data) != allowed or data["schema_version"] != 1 or data["mode"] not in {"s8_preflight", "s8_real_campaign"}:
        raise ConfigError("unsupported S8 config/schema/mode")
    output_path(root, data["output"])
    for key, maximum in (("cpu_slots", 6), ("max_parallel", 6)):
        if type(data[key]) is not int or not 1 <= data[key] <= maximum:
            raise ConfigError(f"invalid {key}")
    for key, maximum in (("disk_budget_gib", 100), ("active_seconds_max", 604800)):
        if type(data[key]) not in (int, float) or not 0 < data[key] <= maximum:
            raise ConfigError(f"invalid {key}")
    if type(data["min_free_gib"]) not in (int, float) or data["min_free_gib"] < 20:
        raise ConfigError("at least 20 GiB free disk must remain")
    if not isinstance(data["inputs"], dict):
        raise ConfigError("inputs must be hash mapping")
    for relative, expected in data["inputs"].items():
        if sha256_file(inside(root, relative)) != expected:
            raise ConfigError(f"input changed: {relative}")
    jobs = data["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ConfigError("jobs must be nonempty")
    ids = []
    keys = {"id", "kind", "dependencies", "cpu", "max_attempts", "duration_seconds", "fail_attempts", "partial_attempts"}
    for job in jobs:
        expected_keys = keys | ({"training"} if job.get("kind") == "epoch_train" else set())
        if set(job) != expected_keys or not ID.fullmatch(job["id"]) or job["kind"] not in WORKERS:
            raise ConfigError("unregistered worker or invalid job schema")
        if job["kind"] == "epoch_train":
            train = job["training"]
            if not isinstance(train, dict) or set(train) != {"experiment", "seed", "epochs", "stop_after_epoch"}:
                raise ConfigError("invalid epoch training schema")
            from .mixed import MIXTURES
            if train["experiment"] not in ({"M0", "L0", "H0_R", "H1_R"} | set(MIXTURES)) or type(train["seed"]) is not int or train["seed"] not in (0, 1, 2):
                raise ConfigError("only registered real-only branches are implemented")
            if train['experiment'] in MIXTURES and train['experiment'].startswith(('A','H')) and train['seed'] != 0:
                raise ConfigError('A/H mixtures allow seed0 only')
            formal = data["mode"] == "s8_real_campaign"
            formal_epochs = MIXTURES[train['experiment']][4] if train['experiment'] in MIXTURES else 150
            if type(train["epochs"]) is not int or train["epochs"] != (formal_epochs if formal else 2):
                raise ConfigError("epoch count does not match scope")
            if type(train["stop_after_epoch"]) is not int or train["stop_after_epoch"] not in ({0} if formal else {0, 1}):
                raise ConfigError("injected epoch interruption is preflight-only")
            if job["cpu"] != 3 or job["duration_seconds"] != 0:
                raise ConfigError("epoch worker reserves main + 2 loader CPU slots")
            if formal:
                if job["id"] != f"{train['experiment']}_seed{train['seed']}":
                    raise ConfigError("formal job identity must match S7")
                if train["seed"] and f"{train['experiment']}_seed0" not in job["dependencies"]:
                    raise ConfigError("seed1/2 require own verified seed0")
        elif data["mode"] == "s8_real_campaign":
            raise ConfigError("formal real campaign cannot count fixtures or unimplemented workers")
        if type(job["cpu"]) is not int or not 1 <= job["cpu"] <= data["cpu_slots"]:
            raise ConfigError("invalid CPU reservation")
        if type(job["max_attempts"]) is not int or not 1 <= job["max_attempts"] <= 3:
            raise ConfigError("attempt budget must be 1..3")
        if type(job["duration_seconds"]) not in (int, float) or not 0 <= job["duration_seconds"] <= 900:
            raise ConfigError("invalid fixture duration")
        for key in ("fail_attempts", "partial_attempts"):
            if not isinstance(job[key], list) or any(type(x) is not int or not 1 <= x <= job["max_attempts"] for x in job[key]):
                raise ConfigError("invalid failure injection")
        if job["kind"] != "fixture" and (job["fail_attempts"] or job["partial_attempts"]):
            raise ConfigError("failure injection is fixture-only")
        ids.append(job["id"])
    if len(ids) != len(set(ids)):
        raise ConfigError("duplicate job ID")
    seen = set()
    while len(seen) < len(ids):
        ready = [j for j in jobs if j["id"] not in seen and isinstance(j["dependencies"], list)
                 and set(j["dependencies"]) <= seen]
        if not ready:
            raise ConfigError("cycle, missing or malformed dependency")
        seen.update(j["id"] for j in ready)
    return data


def contract(root, config_path, code_root=None):
    data = validate_config(root, read_json(config_path))
    approval, plan, records = approved_plan(root)
    records.update(data["inputs"])
    records[config_path.resolve().relative_to(root).as_posix()] = sha256_file(config_path)
    if data["mode"] == "s8_real_campaign":
        from .mixed import MIXTURES
        mixed_jobs = [j for j in data['jobs'] if j['training']['experiment'] in MIXTURES]
        if mixed_jobs:
            if len(mixed_jobs) != len(data['jobs']):
                raise ConfigError('Do not mix completed Real-only and new mixed jobs')
            gate_path = 'artifacts/s8_mixed_release/run_v2/summary.json'
            if gate_path not in data['inputs']:
                raise ConfigError('Mixed release required')
            gate = read_json(inside(root,gate_path))
            if gate.get('status') != 'VERIFIED_MIXED_EPOCH_RELEASE' or gate.get('code_sha256') != digest(code_records(code_root or root)):
                raise ConfigError('Mixed release/code mismatch')
            for relative, expected in gate['evidence'].items():
                if sha256_file(inside(root,relative)) != expected:
                    raise ConfigError('Mixed evidence changed: '+relative)
                records[relative] = expected
            return data, {'config_sha256':digest(data),'approval_sha256':APPROVAL_SHA256,
                          'inputs':records,'code':code_records(code_root or root)}
        gates = [relative for relative in data["inputs"] if relative.startswith("artifacts/s8_epoch_release/") and relative.endswith("/summary.json")]
        if len(gates) != 1:
            raise ConfigError("formal epoch training requires exactly one verified release gate")
        gate = read_json(inside(root, gates[0]))
        if gate.get("status") != "VERIFIED_REAL_EPOCH_RELEASE" or gate.get("code_sha256") != digest(code_records(code_root or root)):
            raise ConfigError("formal release gate/code mismatch")
        if set(gate.get("variants", [])) != {"L0", "H0_R", "H1_R"} or not gate.get("safe_epoch_pause_verified"):
            raise ConfigError("incomplete real epoch release gate")
        for relative, expected in gate["evidence"].items():
            if sha256_file(inside(root, relative)) != expected:
                raise ConfigError("release evidence changed")
            records[relative] = expected
    return data, {"config_sha256": digest(data), "approval_sha256": APPROVAL_SHA256,
                  "inputs": records, "code": code_records(code_root or root)}
