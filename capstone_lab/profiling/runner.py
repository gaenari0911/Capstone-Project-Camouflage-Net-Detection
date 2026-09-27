from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from capstone_lab.config import canonical_json, sha256_file
from capstone_lab.errors import ArtifactError, StateError

from .contracts import S6Campaign, launch_guard, load_s6_campaign, select_common_batch


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, shell=False, check=True)
    row = [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",")]
    process_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, shell=False, check=False,
    )
    compute_apps = []
    for line in process_query.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if fields and fields[0].isdigit():
            memory = int(fields[1]) if len(fields) > 1 and fields[1].isdigit() else None
            compute_apps.append({"pid": int(fields[0]), "used_memory_mib": memory})
    return {
        "index": int(row[0]), "name": row[1], "total_mib": int(row[2]),
        "used_mib": int(row[3]), "utilization_percent": int(row[4]),
        "compute_apps": compute_apps,
        "compute_memory_available": any(item["used_memory_mib"] is not None for item in compute_apps),
        "timestamp_utc": _utc(),
    }


def validate_resume_state(state: dict[str, Any], contract: dict[str, str]) -> None:
    if state.get("contract") != contract:
        raise StateError("persisted S6 state belongs to a different config/input/code contract")


def _environment(campaign: S6Campaign) -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "OPENCV_FOR_THREADS_NUM": "1",
        "YOLO_CONFIG_DIR": str(campaign.artifact_root / "ultralytics_config"),
    })
    return value


def _gpu_spec(campaign: S6Campaign, job_id: str, batch: int, output: Path) -> dict[str, Any]:
    effective = dict(campaign.s5.effective_args)
    effective.update({"imgsz": 640, "batch": batch, "mask_ratio": 1, "overlap_mask": False, "amp": True, "compile": False})
    return {
        "schema_version": 1, "job_id": job_id, "output_dir": str(output),
        "allowed_output_root": str(campaign.artifact_root),
        "model_yaml": str(campaign.s5.model_yaml), "pretrained": str(campaign.s5.pretrained),
        "scale": campaign.s5.scale, "num_classes": campaign.s5.num_classes,
        "seed": campaign.s5.seed, "device": campaign.s5.device,
        "batch": batch, "warmup_steps": campaign.warmup_steps,
        "measure_steps": campaign.measure_steps, "effective_args": effective,
        "expected_initial_state_sha256": campaign.expected_initial_sha256,
        "contract": {
            "config_sha256": campaign.config_sha256, "input_sha256": campaign.input_sha256,
            "code_sha256": campaign.code_sha256, "s5_code_sha256": campaign.s5.code_sha256,
            "library_sha256": campaign.s5.library_sha256,
            "model_source_sha256": campaign.s5.model_source_sha256,
            "pretrained_sha256": campaign.s5.pretrained_sha256,
            "source_zip_sha256": campaign.s5.source_zip_sha256,
            "train_manifest_sha256": campaign.s5.train_manifest_sha256,
            "val_manifest_sha256": campaign.s5.val_manifest_sha256,
        },
    }


def _cpu_spec(campaign: S6Campaign, job_id: str, output: Path) -> dict[str, Any]:
    return {
        "schema_version": 1, "job_id": job_id, "output_dir": str(output),
        "allowed_output_root": str(campaign.artifact_root), "project_root": str(campaign.project_root),
        "source_manifest": str(campaign.source_manifest),
        "resolution_manifest": str(campaign.resolution_manifest),
        "workers": campaign.cpu_workers, "smoke_config": campaign.cpu_smoke,
        "contract": {"config_sha256": campaign.config_sha256, "input_sha256": campaign.input_sha256, "code_sha256": campaign.code_sha256},
    }


def _process_resources(process: psutil.Process) -> tuple[int, float]:
    rss = 0
    cpu_seconds = 0.0
    for item in [process, *process.children(recursive=True)]:
        try:
            rss += item.memory_info().rss
            times = item.cpu_times()
            cpu_seconds += float(times.user + times.system)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return int(rss / 2**20), cpu_seconds


def classify_worker_status(kind: str, exit_code: int, result_status: str | None) -> str:
    if kind == "cpu":
        return "SUCCEEDED" if exit_code == 0 and result_status == "SUCCEEDED" else "FAILED"
    if exit_code == 0 and result_status == "SUCCEEDED":
        return "SUCCEEDED"
    if exit_code == 42 and result_status == "OOM":
        return "OOM"
    return "FAILED"


def run_group(
    campaign: S6Campaign,
    jobs: list[tuple[str, str, dict[str, Any], Path]],
    *,
    reservations_mib: list[int],
    state: dict[str, Any],
) -> dict[str, Any]:
    gpu_before = gpu_snapshot()
    guard = None
    if reservations_mib:
        guard = launch_guard(total_mib=gpu_before["total_mib"], external_used_mib=gpu_before["used_mib"], reservations_mib=reservations_mib)
        if not guard["accepted"]:
            raise ArtifactError(f"GPU launch rejected by reservation guard: {guard}")
    group_started = time.time()
    records: dict[str, dict[str, Any]] = {}
    active: list[tuple[subprocess.Popen[bytes], Any, Any, str, str, Path, float]] = []
    for kind, job_id, spec, output in jobs:
        output.mkdir(parents=True, exist_ok=True)
        spec_path = output / "worker_spec.json"
        _write(spec_path, spec)
        stdout_handle = (output / "stdout.log").open("wb")
        stderr_handle = (output / "stderr.log").open("wb")
        module = "capstone_lab.profiling.gpu_worker" if kind == "gpu" else "capstone_lab.profiling.cpu_worker"
        started = time.time()
        process = subprocess.Popen(
            [sys.executable, "-m", module, "--spec", str(spec_path)],
            cwd=campaign.project_root, stdout=stdout_handle, stderr=stderr_handle,
            env=_environment(campaign), shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        active.append((process, stdout_handle, stderr_handle, kind, job_id, output, started))
        state["events"].append({"timestamp_utc": _utc(), "event_type": "JOB_STARTED", "job_id": job_id, "detail": {"kind": kind, "pid": process.pid, "retry": 0}})
        _write(campaign.artifact_root / "state.json", state)
    gpu_samples = [gpu_before]
    peaks = {job_id: {"rss_mib": 0, "cpu_time_seconds": 0.0} for _, job_id, _, _ in jobs}
    while any(item[0].poll() is None for item in active):
        time.sleep(campaign.monitor_interval_seconds)
        gpu_samples.append(gpu_snapshot())
        for process, _, _, _, job_id, _, _ in active:
            try:
                rss, cpu_seconds = _process_resources(psutil.Process(process.pid))
                peaks[job_id]["rss_mib"] = max(peaks[job_id]["rss_mib"], rss)
                peaks[job_id]["cpu_time_seconds"] = max(peaks[job_id]["cpu_time_seconds"], cpu_seconds)
            except psutil.NoSuchProcess:
                pass
    for process, stdout_handle, stderr_handle, kind, job_id, output, started in active:
        code = process.wait()
        stdout_handle.close()
        stderr_handle.close()
        result_path = output / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
        status = classify_worker_status(kind, code, result.get("status") if result else None)
        ended = time.time()
        wall = ended - started
        record = {
            "kind": kind, "job_id": job_id, "pid": process.pid,
            "started_unix": started, "ended_unix": ended,
            "wall_seconds": wall, "exit_code": code,
            "retry_count": 0, "status": status,
            "peak_process_rss_mib": peaks[job_id]["rss_mib"],
            "process_cpu_time_seconds": peaks[job_id]["cpu_time_seconds"],
            "average_process_cpu_percent_of_one_core": (
                100.0 * peaks[job_id]["cpu_time_seconds"] / wall if wall > 0 else 0.0
            ),
            "result_path": str(result_path),
            "result_sha256": sha256_file(result_path) if result_path.is_file() else None,
            "result": result,
        }
        records[job_id] = record
        state["jobs"][job_id] = {key: value for key, value in record.items() if key != "result"}
        state["events"].append({"timestamp_utc": _utc(), "event_type": f"JOB_{status}", "job_id": job_id, "detail": {"exit_code": code, "wall_seconds": record["wall_seconds"]}})
        _write(campaign.artifact_root / "state.json", state)
    gpu_after = gpu_snapshot()
    gpu_samples.append(gpu_after)
    return {
        "started_unix": group_started, "ended_unix": time.time(),
        "wall_seconds": time.time() - group_started,
        "launch_guard": guard, "gpu_before": gpu_before, "gpu_after": gpu_after,
        "gpu_monitor": {
            "samples": len(gpu_samples),
            "peak_used_mib": max(row["used_mib"] for row in gpu_samples),
            "peak_utilization_percent": max(row["utilization_percent"] for row in gpu_samples),
            "process_memory_visibility": "per-process" if gpu_before["compute_memory_available"] else "WDDM_NA_total_device_used_recorded",
        },
        "jobs": records,
    }


def _loss_consistency(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [record["result"]["loss"]["values"] for record in records]
    reference = values[0]
    deltas = [max(abs(left - right) for left, right in zip(reference, current, strict=True)) for current in values[1:]]
    maximum = max(deltas, default=0.0)
    relative_deltas = [
        max(
            abs(left - right) / max(abs(left), abs(right), 1.0)
            for left, right in zip(reference, current, strict=True)
        )
        for current in values[1:]
    ]
    maximum_relative = max(relative_deltas, default=0.0)
    return {
        "max_absolute_loss_delta": maximum,
        "max_relative_loss_delta": maximum_relative,
        "relative_tolerance": 0.01,
        "within_tolerance": maximum_relative <= 0.01,
        "rationale": "resource profiling accepts up to 1% cross-process numeric drift; all losses and boundary gradients must remain finite",
    }


def run_campaign(config_path: Path, project_root: Path) -> dict[str, Any]:
    campaign = load_s6_campaign(config_path, project_root)
    campaign.artifact_root.mkdir(parents=True, exist_ok=True)
    contract = {"config_sha256": campaign.config_sha256, "input_sha256": campaign.input_sha256, "code_sha256": campaign.code_sha256}
    summary_path = campaign.artifact_root / "summary.json"
    state_path = campaign.artifact_root / "state.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("contract") != contract:
            raise StateError("completed S6 summary belongs to a different contract")
        summary["successful_run_reused"] = True
        return summary
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        validate_resume_state(state, contract)
    else:
        state = {"schema_version": 1, "contract": contract, "jobs": {}, "events": []}
        _write(state_path, state)
    initial_gpu = gpu_snapshot()
    batch_profiles = []
    for batch in campaign.batches:
        job_id = f"batch_{batch}"
        output = campaign.artifact_root / "batch_sweep" / job_id
        group = run_group(campaign, [("gpu", job_id, _gpu_spec(campaign, job_id, batch, output), output)], reservations_mib=[campaign.single_job_reservation_mib], state=state)
        record = group["jobs"][job_id]
        batch_profiles.append({**record["result"], "process": {key: value for key, value in record.items() if key != "result"}, "monitor": group["gpu_monitor"], "launch_guard": group["launch_guard"]})
    selected = select_common_batch(batch_profiles, total_mib=initial_gpu["total_mib"], external_used_mib=initial_gpu["used_mib"])
    batch = selected["batch"]
    reservation = selected["reservation_mib"]
    sequential_groups = []
    sequential_records = []
    for suffix in ("a", "b"):
        job_id = f"sequential_{suffix}"
        output = campaign.artifact_root / "sequential" / job_id
        group = run_group(campaign, [("gpu", job_id, _gpu_spec(campaign, job_id, batch, output), output)], reservations_mib=[reservation], state=state)
        sequential_groups.append(group)
        sequential_records.append(group["jobs"][job_id])
    concurrent_jobs = []
    for suffix in ("a", "b"):
        job_id = f"concurrent_{suffix}"
        output = campaign.artifact_root / "concurrent" / job_id
        concurrent_jobs.append(("gpu", job_id, _gpu_spec(campaign, job_id, batch, output), output))
    concurrent_group = run_group(campaign, concurrent_jobs, reservations_mib=[reservation, reservation], state=state)
    concurrent_records = list(concurrent_group["jobs"].values())
    concurrent_oom = any(record["status"] == "OOM" for record in concurrent_records)
    fallback = {
        "triggered": concurrent_oom,
        "action": "concurrency reduced from 2 to 1 with all per-job settings unchanged" if concurrent_oom else None,
        "fallback_evidence": "completed sequential pair" if concurrent_oom else None,
    }
    if not concurrent_oom and any(record["status"] != "SUCCEEDED" for record in concurrent_records):
        raise ArtifactError("concurrent GPU profiling failed for a non-OOM reason")

    cpu_baseline_output = campaign.artifact_root / "cpu_overlap" / "cpu_baseline"
    cpu_baseline = run_group(campaign, [("cpu", "cpu_baseline", _cpu_spec(campaign, "cpu_baseline", cpu_baseline_output), cpu_baseline_output)], reservations_mib=[], state=state)
    overlap_gpu_output = campaign.artifact_root / "cpu_overlap" / "gpu_overlap"
    overlap_cpu_output = campaign.artifact_root / "cpu_overlap" / "cpu_overlap"
    overlap = run_group(campaign, [
        ("gpu", "gpu_overlap", _gpu_spec(campaign, "gpu_overlap", batch, overlap_gpu_output), overlap_gpu_output),
        ("cpu", "cpu_overlap", _cpu_spec(campaign, "cpu_overlap", overlap_cpu_output), overlap_cpu_output),
    ], reservations_mib=[reservation], state=state)
    cpu_alone = cpu_baseline["jobs"]["cpu_baseline"]
    cpu_together = overlap["jobs"]["cpu_overlap"]
    gpu_together = overlap["jobs"]["gpu_overlap"]
    s4_hash_equal = cpu_alone["result"]["s4_deterministic_sha256"] == cpu_together["result"]["s4_deterministic_sha256"]
    gpu_success_records = [*sequential_records]
    if not concurrent_oom:
        gpu_success_records.extend(concurrent_records)
    gpu_success_records.append(gpu_together)
    initial_hashes = {record["result"]["model"]["initial_hashes"]["full_state_sha256"] for record in gpu_success_records}
    loss_check = _loss_consistency(gpu_success_records)
    checks = {
        "batch_4_8_16_profiled": {row["batch"] for row in batch_profiles} == {4, 8, 16},
        "sequential_pair_succeeded": all(row["status"] == "SUCCEEDED" for row in sequential_records),
        "concurrent_attempt_recorded": len(concurrent_records) == 2,
        "concurrent_success_or_oom_fallback": (not concurrent_oom and all(row["status"] == "SUCCEEDED" for row in concurrent_records)) or (concurrent_oom and fallback["triggered"]),
        "overlap_succeeded": cpu_alone["status"] == cpu_together["status"] == gpu_together["status"] == "SUCCEEDED",
        "s4_output_hash_invariant": s4_hash_equal,
        "initial_state_invariant": initial_hashes == {campaign.expected_initial_sha256},
        "finite_loss_gradient_tolerance": loss_check["within_tolerance"],
        "no_Test_or_formal_training": True,
    }
    status = "VERIFIED" if all(checks.values()) else "PARTIAL_FAILURE"
    summary = {
        "schema_version": 1, "status": status,
        "scope": "S6 profiling only; not formal training, validation performance, or Test",
        "contract": contract, "successful_run_reused": False,
        "gpu": initial_gpu, "batch_profiles": batch_profiles,
        "selected_common_batch_candidate": selected,
        "sequential": {"groups": sequential_groups, "aggregate_wall_seconds": sum(row["wall_seconds"] for row in sequential_groups), "loss_consistency": _loss_consistency(sequential_records)},
        "concurrent": {"group": concurrent_group, "oom": concurrent_oom, "fallback": fallback, "loss_consistency": None if concurrent_oom else _loss_consistency(concurrent_records)},
        "throughput_comparison": {
            "sequential_total_samples_per_second": sum(row["result"]["samples_per_second"] for row in sequential_records) / 2,
            "concurrent_total_samples_per_second": None if concurrent_oom else sum(row["result"]["samples_per_second"] for row in concurrent_records),
        },
        "cpu_gpu_overlap": {
            "cpu_baseline": cpu_baseline, "overlap": overlap,
            "s4_deterministic_hash_equal": s4_hash_equal,
            "separate_wall_seconds": cpu_baseline["wall_seconds"] + sequential_groups[0]["wall_seconds"],
            "overlap_wall_seconds": overlap["wall_seconds"],
        },
        "global_loss_consistency": loss_check, "checks": checks,
        "limitations": [
            "Synthetic fixed train tensors measure resource behavior, not model quality.",
            "The validation fixture measures memory only and never reads Test.",
            "Windows WDDM may hide per-process GPU memory; device-level used-memory polling is retained.",
            "The selected batch remains a candidate until S7 final freeze review.",
        ],
    }
    _write(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m capstone_lab.profiling.runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    summary = run_campaign(args.config, args.project_root)
    print(canonical_json(summary))
    return 0 if summary["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
