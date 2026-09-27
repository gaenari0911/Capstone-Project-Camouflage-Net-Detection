from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from capstone_lab.config import canonical_json, sha256_file
from capstone_lab.errors import ArtifactError, StateError

from .planner import PlanRow, build_plan, load_s7_config, registry_csv, target_closure


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise StateError(f"existing S7 artifact differs from the current contract: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def validate_existing_outputs(output: Path, contract: dict[str, str]) -> dict[str, Any] | None:
    summary_path = output / "summary.json"
    manifest_path = output / "artifact_manifest.json"
    if not summary_path.exists() and not manifest_path.exists():
        return None
    if not summary_path.is_file() or not manifest_path.is_file():
        raise StateError("partial S7 output cannot be treated as a completed plan")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != contract:
        raise StateError("S7 output belongs to a different config/input/code contract")
    for relative, expected in manifest.get("artifacts", {}).items():
        path = output / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ArtifactError(f"completed S7 artifact is missing or changed: {relative}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("contract") != contract or summary.get("status") != "VERIFIED":
        raise ArtifactError("S7 summary status/contract mismatch")
    return summary


def _target_view(plan: dict[str, Any], target: str, requested_approvals: tuple[str, ...]) -> dict[str, Any]:
    nodes = {item["node_id"]: item for item in plan["dag"]["nodes"]}
    selected = target_closure(nodes, target, plan["approvals"], requested_approvals)
    recorded = plan.get("target_closures", {}).get(target)
    if recorded is not None:
        return dict(recorded)
    training = {
        row["job_id"]: row
        for row in plan["registry"]
        if row["node_kind"] == "TRAIN"
    }
    selected_rows = [training[item] for item in selected["nodes"] if item in training]
    selected["training_jobs"] = len(selected_rows)
    selected["epoch_job_units"] = sum(row["epochs"] for row in selected_rows)
    selected["planned_unique_generated_images"] = sum(
        nodes[item]["planned_unique_images"] for item in selected["nodes"]
    )
    return selected


def run_plan(config_path: Path, project_root: Path, *, target: str, requested_approvals: tuple[str, ...] = ()) -> dict[str, Any]:
    config = load_s7_config(config_path, project_root)
    contract = {"config_sha256": config.config_sha256, "input_sha256": config.input_sha256, "code_sha256": config.code_sha256}
    reused = validate_existing_outputs(config.artifact_root, contract)
    if reused is not None:
        selected = _target_view(reused, target, requested_approvals)
        return {"status": reused["status"], "scope": reused["scope"], "contract": contract, "successful_plan_reused": True, "selected_target": selected, "totals": reused["totals"], "checks": reused["checks"]}
    plan = build_plan(config)
    selected = _target_view(plan, target, requested_approvals)
    files = {
        "summary.json": _json_bytes(plan),
        "registry_v1.csv": registry_csv(
            [PlanRow(**{**row, "approval_gates": tuple(row["approval_gates"])}) for row in plan["registry"]]
        ).encode("utf-8"),
        "dag_v1.json": _json_bytes(plan["dag"]),
        "target_closures_v1.json": _json_bytes(plan["target_closures"]),
        "cost_summary_v1.json": _json_bytes(plan["totals"]),
    }
    for relative, content in files.items():
        _write_immutable(config.artifact_root / relative, content)
    artifact_hashes = {relative: sha256_file(config.artifact_root / relative) for relative in sorted(files)}
    _write_immutable(config.artifact_root / "artifact_manifest.json", _json_bytes({"schema_version": 1, "contract": contract, "artifacts": artifact_hashes}))
    return {"status": plan["status"], "scope": plan["scope"], "contract": contract, "successful_plan_reused": False, "selected_target": selected, "totals": plan["totals"], "checks": plan["checks"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m capstone_lab.planning.runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--target", choices=("ablation", "main", "low", "head_ablation", "core", "all"), required=True)
    parser.add_argument("--approval", action="append", default=[])
    args = parser.parse_args(argv)
    result = run_plan(args.config, args.project_root, target=args.target, requested_approvals=tuple(args.approval))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
