"""Conservative single-GPU admission based on the real 191-instance stress test."""
from __future__ import annotations

import subprocess
from pathlib import Path

import psutil

GPU_RESERVED_MIB = 10044
RAM_RESERVED_GIB = 8


def gpu_memory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=10,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("exactly one approved GPU required")
    total, used = [int(value.strip()) for value in lines[0].split(",")]
    return total, used


def admission(job, running, jobs):
    heavy = [jobs[row["id"]] for row in running if jobs[row["id"]]["kind"] != "fixture"]
    if job["kind"] == "fixture":
        return True, "fixture"
    if len(heavy) >= 2:
        return False, "IO reservations full (2)"
    memory = psutil.virtual_memory()
    reserved = sum(RAM_RESERVED_GIB if j["kind"] == "epoch_train" else 1 for j in heavy)
    requested = RAM_RESERVED_GIB if job["kind"] == "epoch_train" else 1
    if (reserved + requested + 8) * 2**30 > memory.total or memory.available < (requested + 4) * 2**30:
        return False, "RAM reservation/headroom unavailable"
    if job["kind"] == "epoch_train":
        if any(j["kind"] == "epoch_train" for j in heavy):
            return False, "real-loader GPU concurrency is 1; approved maximum 2 is not forced"
        try:
            total, used = gpu_memory()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return False, f"GPU telemetry unavailable: {exc}"
        if used + GPU_RESERVED_MIB > min(total * .85, total - 2048):
            return False, f"GPU occupied: used={used}, reservation={GPU_RESERVED_MIB} MiB"
    return True, "reserved"


def budget_roots(root):
    # Conservatively include ALL S8 outputs and separately named future S8 environments.
    # Never subtract old preflight runs or discard failures to conceal incremental cost.
    return sorted([p for p in (root / "artifacts").glob("s8_*") if p.is_dir()] +
                  [p for p in root.glob(".venv-s8*") if p.is_dir()])
