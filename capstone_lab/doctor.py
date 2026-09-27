from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from typing import Any

from .audit import inspect_recovery
from .config import CampaignConfig


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def _memory() -> dict[str, int] | None:
    if os.name != "nt":
        return None
    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return {
        "total_bytes": int(status.total_physical),
        "available_bytes": int(status.available_physical),
    }


def _gpu() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "reason": "nvidia-smi not found"}
    command = [
        executable,
        "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        rows = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 5:
                rows.append(
                    {
                        "index": int(fields[0]),
                        "name": fields[1],
                        "memory_total_mib": int(fields[2]),
                        "memory_free_mib": int(fields[3]),
                        "driver": fields[4],
                    }
                )
        return {"available": bool(rows), "devices": rows}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"available": False, "reason": str(exc)}


def inspect_environment(config: CampaignConfig | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_logical": os.cpu_count(),
        "memory": _memory(),
        "gpu": _gpu(),
    }
    if config:
        report["campaign"] = {
            "name": config.campaign_name,
            "status": config.status,
            "config_hash": config.config_hash,
            "input_hash": config.input_hash,
            "inputs": list(config.input_records),
            "artifact_root": str(config.artifact_root),
            "protected_roots": [str(path) for path in config.protected_roots],
        }
        report["recovery_audit"] = inspect_recovery(config.source_path)
    return report
