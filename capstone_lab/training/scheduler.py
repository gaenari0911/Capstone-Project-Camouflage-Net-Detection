from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from capstone_lab.config import canonical_json, sha256_file
from capstone_lab.errors import ArtifactError, ConfigError, StateError


@dataclass(frozen=True)
class S5Job:
    job_id: str
    dependencies: tuple[str, ...]
    steps: int
    max_attempts: int
    inject_failure_after_step: int


@dataclass(frozen=True)
class S5Campaign:
    source_path: Path
    project_root: Path
    campaign_name: str
    artifact_root: Path
    source_zip: Path
    model_yaml: Path
    pretrained: Path
    train_manifest: Path
    val_manifest: Path
    source_zip_sha256: str
    scale: str
    num_classes: int
    seed: int
    device: str
    effective_args: dict[str, Any]
    metrics: dict[str, float]
    jobs: tuple[S5Job, ...]
    config_sha256: str
    input_sha256: str
    code_sha256: str
    library_sha256: str
    model_source_sha256: str
    pretrained_sha256: str
    train_manifest_sha256: str
    val_manifest_sha256: str


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _hash_records(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


def _tree_hash(paths: list[Path]) -> str:
    records = [
        {"path": path.as_posix(), "sha256": sha256_file(path)}
        for path in sorted(paths, key=lambda item: item.as_posix().casefold())
    ]
    return _hash_records(records)


def _assert_non_test_manifest(path: Path, expected_split: str) -> None:
    text = path.read_text(encoding="utf-8")
    rows: list[Any] = []
    try:
        value = json.loads(text)
        rows = value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        for line in text.splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ConfigError(f"manifest must contain JSON objects: {path}")
    observed = {str(row.get("split", "")).strip().lower() for row in rows}
    if "test" in observed or "testing" in observed:
        raise ConfigError(f"Test manifest is forbidden for S5 train/eval: {path}")
    if observed != {expected_split}:
        raise ConfigError(
            f"{expected_split} manifest must contain only split={expected_split}: {path}"
        )


def load_s5_campaign(config_path: Path, project_root: Path) -> S5Campaign:
    config_path = config_path.resolve(strict=True)
    project_root = project_root.resolve(strict=True)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or raw.get("status") != "frozen":
        raise ConfigError("S5 campaign must use schema_version=1 and status=frozen")
    if raw.get("gpu_slots") != 1:
        raise ConfigError("S5 smoke requires exactly one GPU slot")
    paths = raw["paths"]

    def resolved(key: str) -> Path:
        value = Path(paths[key])
        return (value if value.is_absolute() else project_root / value).resolve(strict=False)

    artifact_root = resolved("artifact_root")
    artifacts = (project_root / "artifacts").resolve(strict=False)
    if artifact_root == artifacts or not _is_relative_to(artifact_root, artifacts):
        raise ConfigError("S5 output must be a dedicated directory below artifacts")
    source_zip = resolved("source_zip")
    model_yaml = resolved("model_yaml")
    pretrained = resolved("pretrained")
    train_manifest = resolved("train_manifest")
    val_manifest = resolved("val_manifest")
    for path in (source_zip, model_yaml, pretrained, train_manifest, val_manifest):
        if not path.is_file():
            raise ConfigError(f"S5 input is missing: {path}")
        if _is_relative_to(path, artifact_root):
            raise ConfigError(f"S5 input may not be inside its output directory: {path}")
    if _is_relative_to(artifact_root, project_root / "Dataset"):
        raise ConfigError("S5 output may not overlap Dataset")

    _assert_non_test_manifest(train_manifest, "train")
    _assert_non_test_manifest(val_manifest, "val")
    expected_zip_hash = str(raw["source_zip_sha256"]).lower()
    if sha256_file(source_zip) != expected_zip_hash:
        raise ArtifactError("custom source ZIP hash changed")
    jobs = tuple(
        S5Job(
            job_id=str(row["job_id"]),
            dependencies=tuple(row.get("dependencies", [])),
            steps=int(row["steps"]),
            max_attempts=int(row["max_attempts"]),
            inject_failure_after_step=int(row.get("inject_failure_after_step", 0)),
        )
        for row in raw["jobs"]
    )
    if len(jobs) != 2 or jobs[0].dependencies or jobs[1].dependencies != (jobs[0].job_id,):
        raise ConfigError("S5 smoke must be an explicit two-job dependency chain")
    if len({job.job_id for job in jobs}) != 2:
        raise ConfigError("S5 job IDs must be unique")
    if any(job.steps not in {1, 2} or not 1 <= job.max_attempts <= 2 for job in jobs):
        raise ConfigError("S5 jobs allow only 1-2 steps and 1-2 attempts")
    if jobs[0].inject_failure_after_step and jobs[0].max_attempts < 2:
        raise ConfigError("injected checkpoint failure requires a bounded retry")

    config_sha256 = hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()
    input_paths = [source_zip, model_yaml, pretrained, train_manifest, val_manifest]
    input_records = [
        {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in input_paths
    ]
    input_sha256 = _hash_records(input_records)
    code_paths = sorted((project_root / "capstone_lab" / "training").glob("*.py"))
    code_sha256 = _tree_hash(code_paths)
    library_root = model_yaml.parents[1] / "ultralytics_lib" / "ultralytics"
    critical = [
        library_root / "__init__.py",
        library_root / "nn" / "modules" / "head.py",
        library_root / "nn" / "tasks.py",
        library_root / "data" / "dataset.py",
        library_root / "utils" / "loss.py",
    ]
    if not all(path.is_file() for path in critical):
        raise ConfigError("custom fork critical source files are missing")
    library_sha256 = _tree_hash(critical)
    effective = dict(raw["effective_args"])
    required_args = {
        "imgsz",
        "batch",
        "optimizer",
        "lr0",
        "weight_decay",
        "lrf",
        "amp",
        "compile",
        "workers",
        "mask_ratio",
        "overlap_mask",
        "box",
        "cls",
        "dfl",
        "mosaic",
        "mixup",
        "copy_paste",
        "cutmix",
    }
    if set(effective) != required_args or effective["optimizer"] != "AdamW":
        raise ConfigError("effective_args are incomplete or optimizer is not explicit AdamW")
    return S5Campaign(
        source_path=config_path,
        project_root=project_root,
        campaign_name=str(raw["campaign_name"]),
        artifact_root=artifact_root,
        source_zip=source_zip,
        model_yaml=model_yaml,
        pretrained=pretrained,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        source_zip_sha256=expected_zip_hash,
        scale=str(raw["model"]["scale"]),
        num_classes=int(raw["model"]["num_classes"]),
        seed=int(raw["model"]["initialization_seed"]),
        device=str(raw["device"]),
        effective_args=effective,
        metrics={key: float(value) for key, value in raw["metrics"].items()},
        jobs=jobs,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        code_sha256=code_sha256,
        library_sha256=library_sha256,
        model_source_sha256=sha256_file(model_yaml),
        pretrained_sha256=sha256_file(pretrained),
        train_manifest_sha256=sha256_file(train_manifest),
        val_manifest_sha256=sha256_file(val_manifest),
    )


def dependency_blocked_jobs(
    jobs: tuple[S5Job, ...], states: dict[str, str]
) -> set[str]:
    failed = {job_id for job_id, state in states.items() if state == "FAILED"}
    blocked: set[str] = set()
    changed = True
    while changed:
        changed = False
        for job in jobs:
            if job.job_id in failed or job.job_id in blocked:
                continue
            if set(job.dependencies) & (failed | blocked):
                blocked.add(job.job_id)
                changed = True
    return blocked


def _event(state: dict[str, Any], event_type: str, job_id: str | None, detail: dict[str, Any]) -> None:
    state["events"].append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "job_id": job_id,
            "detail": detail,
        }
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_success(result_path: Path, campaign: S5Campaign, job: S5Job) -> dict[str, Any]:
    if not result_path.is_file():
        raise ArtifactError(f"worker result is missing: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_contract = {
        "config_sha256": campaign.config_sha256,
        "input_sha256": campaign.input_sha256,
        "code_sha256": campaign.code_sha256,
        "library_sha256": campaign.library_sha256,
        "model_source_sha256": campaign.model_source_sha256,
        "pretrained_sha256": campaign.pretrained_sha256,
        "source_zip_sha256": campaign.source_zip_sha256,
        "train_manifest_sha256": campaign.train_manifest_sha256,
        "val_manifest_sha256": campaign.val_manifest_sha256,
    }
    if result.get("status") != "SUCCEEDED" or result.get("job_id") != job.job_id:
        raise ArtifactError("worker result identity/status mismatch")
    for key, value in expected_contract.items():
        if result.get("contract", {}).get(key) != value:
            raise ArtifactError(f"worker result contract mismatch: {key}")
    if result.get("model", {}).get("class") != "DualHeadSegment":
        raise ArtifactError("worker did not use DualHeadSegment")
    if result.get("model", {}).get("scale") != campaign.scale:
        raise ArtifactError("worker used the wrong model scale")
    if result.get("model", {}).get("num_classes") != campaign.num_classes:
        raise ArtifactError("worker used the wrong class count")
    if result.get("final_global_step") != job.steps:
        raise ArtifactError("worker did not complete the requested smoke steps")
    if result.get("metrics", {}).get("box", "missing") is not None:
        raise ArtifactError("mask fixture must not be mislabeled as box metrics")
    return result


def run_s5_smoke_campaign(
    *, config_path: Path, project_root: Path
) -> dict[str, Any]:
    campaign = load_s5_campaign(config_path, project_root)
    campaign.artifact_root.mkdir(parents=True, exist_ok=True)
    state_path = campaign.artifact_root / "scheduler_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != campaign.config_sha256 or state.get("input_sha256") != campaign.input_sha256:
            raise StateError("persisted S5 state belongs to a different config/input contract")
    else:
        state = {
            "schema_version": 1,
            "campaign": campaign.campaign_name,
            "config_sha256": campaign.config_sha256,
            "input_sha256": campaign.input_sha256,
            "gpu_slots": 1,
            "jobs": {
                job.job_id: {
                    "state": "PENDING",
                    "attempt_count": 0,
                    "result_path": None,
                    "result_sha256": None,
                    "last_error": None,
                }
                for job in campaign.jobs
            },
            "events": [],
        }
        _write_json(state_path, state)

    results: dict[str, dict[str, Any]] = {}
    for job in campaign.jobs:
        row = state["jobs"][job.job_id]
        if row["state"] == "SUCCEEDED":
            result_path = Path(row["result_path"])
            if sha256_file(result_path) != row["result_sha256"]:
                raise ArtifactError(f"completed result hash changed: {job.job_id}")
            results[job.job_id] = _validate_success(result_path, campaign, job)
            continue
        if any(state["jobs"][dependency]["state"] != "SUCCEEDED" for dependency in job.dependencies):
            row["state"] = "BLOCKED_DEPENDENCY"
            _event(state, "JOB_BLOCKED_DEPENDENCY", job.job_id, {})
            _write_json(state_path, state)
            continue
        while row["attempt_count"] < job.max_attempts and row["state"] != "SUCCEEDED":
            row["attempt_count"] += 1
            attempt = row["attempt_count"]
            job_dir = campaign.artifact_root / "jobs" / job.job_id
            attempt_dir = job_dir / f"attempt_{attempt:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            spec = {
                "schema_version": 1,
                "campaign": campaign.campaign_name,
                "job_id": job.job_id,
                "attempt": attempt,
                "steps": job.steps,
                "inject_failure_after_step": job.inject_failure_after_step,
                "output_dir": str(job_dir),
                "allowed_output_root": str(campaign.artifact_root),
                "model_yaml": str(campaign.model_yaml),
                "pretrained": str(campaign.pretrained),
                "scale": campaign.scale,
                "num_classes": campaign.num_classes,
                "seed": campaign.seed,
                "device": campaign.device,
                "effective_args": campaign.effective_args,
                "metrics": campaign.metrics,
                "hashes": {
                    "config_sha256": campaign.config_sha256,
                    "input_sha256": campaign.input_sha256,
                    "code_sha256": campaign.code_sha256,
                    "library_sha256": campaign.library_sha256,
                    "model_source_sha256": campaign.model_source_sha256,
                    "pretrained_sha256": campaign.pretrained_sha256,
                    "source_zip_sha256": campaign.source_zip_sha256,
                    "train_manifest_sha256": campaign.train_manifest_sha256,
                    "val_manifest_sha256": campaign.val_manifest_sha256,
                },
            }
            spec_path = attempt_dir / "worker_spec.json"
            _write_json(spec_path, spec)
            stdout_path = attempt_dir / "stdout.log"
            stderr_path = attempt_dir / "stderr.log"
            row["state"] = "RUNNING"
            _event(state, "JOB_STARTED", job.job_id, {"attempt": attempt, "gpu_slots_used": 1})
            _write_json(state_path, state)
            environment = os.environ.copy()
            environment.update(
                {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OPENCV_FOR_THREADS_NUM": "1",
                    "YOLO_CONFIG_DIR": str(campaign.artifact_root / "ultralytics_config"),
                }
            )
            (campaign.artifact_root / "ultralytics_config").mkdir(parents=True, exist_ok=True)
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                completed = subprocess.run(
                    [sys.executable, "-m", "capstone_lab.training.worker", "--spec", str(spec_path)],
                    cwd=campaign.project_root,
                    stdout=stdout,
                    stderr=stderr,
                    env=environment,
                    shell=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    check=False,
                )
            if completed.returncode == 0:
                result_path = job_dir / "result.json"
                try:
                    result = _validate_success(result_path, campaign, job)
                except ArtifactError as exc:
                    row["last_error"] = str(exc)
                else:
                    row.update(
                        {
                            "state": "SUCCEEDED",
                            "result_path": str(result_path),
                            "result_sha256": sha256_file(result_path),
                            "last_error": None,
                        }
                    )
                    results[job.job_id] = result
                    _event(state, "JOB_SUCCEEDED", job.job_id, {"attempt": attempt})
                    _write_json(state_path, state)
                    break
            else:
                row["last_error"] = f"worker exited with code {completed.returncode}"
            if attempt < job.max_attempts:
                row["state"] = "RETRY_WAIT"
                _event(
                    state,
                    "JOB_RETRY",
                    job.job_id,
                    {"attempt": attempt, "error": row["last_error"]},
                )
            else:
                row["state"] = "FAILED"
                _event(
                    state,
                    "JOB_FAILED",
                    job.job_id,
                    {"attempt": attempt, "error": row["last_error"]},
                )
            _write_json(state_path, state)

    states = {job_id: row["state"] for job_id, row in state["jobs"].items()}
    blocked = dependency_blocked_jobs(campaign.jobs, states)
    for job_id in blocked:
        if state["jobs"][job_id]["state"] not in {"SUCCEEDED", "FAILED"}:
            state["jobs"][job_id]["state"] = "BLOCKED_DEPENDENCY"
    status = "VERIFIED" if all(value == "SUCCEEDED" for value in states.values()) else "PARTIAL_FAILURE"
    initialization = {
        job_id: result["model"]["initial_hashes"] for job_id, result in results.items()
    }
    if status == "VERIFIED":
        full_hashes = {row["full_state_sha256"] for row in initialization.values()}
        common_hashes = {row["common_state_sha256"] for row in initialization.values()}
        boundary_hashes = {row["boundary_head_sha256"] for row in initialization.values()}
        if len(full_hashes) != 1 or len(common_hashes) != 1 or len(boundary_hashes) != 1:
            raise ArtifactError("same-structure jobs did not independently reproduce initial state")
        first_events = [event for event in state["events"] if event["event_type"] == "JOB_STARTED"]
        first_success = next(
            index
            for index, event in enumerate(state["events"])
            if event["event_type"] == "JOB_SUCCEEDED" and event["job_id"] == campaign.jobs[0].job_id
        )
        second_start = next(
            index
            for index, event in enumerate(state["events"])
            if event["event_type"] == "JOB_STARTED" and event["job_id"] == campaign.jobs[1].job_id
        )
        if not first_events or second_start <= first_success:
            raise ArtifactError("second GPU job started before the first succeeded")
    summary = {
        "schema_version": 1,
        "status": status,
        "campaign": campaign.campaign_name,
        "gpu_slots": 1,
        "states": {job_id: row["state"] for job_id, row in state["jobs"].items()},
        "attempt_counts": {
            job_id: row["attempt_count"] for job_id, row in state["jobs"].items()
        },
        "hashes": {
            "config_sha256": campaign.config_sha256,
            "input_sha256": campaign.input_sha256,
            "code_sha256": campaign.code_sha256,
            "library_sha256": campaign.library_sha256,
            "source_zip_sha256": campaign.source_zip_sha256,
            "model_source_sha256": campaign.model_source_sha256,
            "pretrained_sha256": campaign.pretrained_sha256,
            "train_manifest_sha256": campaign.train_manifest_sha256,
            "val_manifest_sha256": campaign.val_manifest_sha256,
        },
        "model": {
            "class": "DualHeadSegment",
            "scale": campaign.scale,
            "num_classes": campaign.num_classes,
            "initialization_seed": campaign.seed,
            "initialization_policy": "independent base pretrained plus seed-initialized cv5 boundary head",
        },
        "effective_args": campaign.effective_args,
        "initialization": initialization,
        "results": {
            job_id: {
                "result_path": state["jobs"][job_id]["result_path"],
                "result_sha256": state["jobs"][job_id]["result_sha256"],
                "resumed": result.get("resumed"),
                "resume_from_global_step": result.get("resume_from_global_step"),
                "checkpoint_sha256": result.get("checkpoint_sha256"),
                "metrics": result.get("metrics"),
            }
            for job_id, result in results.items()
        },
    }
    _write_json(campaign.artifact_root / "summary.json", summary)
    _write_json(state_path, state)
    return summary
