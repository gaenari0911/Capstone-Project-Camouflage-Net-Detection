from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capstone_lab.config import canonical_json, sha256_file
from capstone_lab.errors import ArtifactError, ConfigError
from capstone_lab.training.scheduler import S5Campaign, load_s5_campaign


@dataclass(frozen=True)
class S6Campaign:
    project_root: Path
    source_path: Path
    artifact_root: Path
    s5: S5Campaign
    batches: tuple[int, ...]
    warmup_steps: int
    measure_steps: int
    cpu_workers: int
    cpu_smoke: dict[str, Any]
    source_manifest: Path
    resolution_manifest: Path
    config_sha256: str
    input_sha256: str
    code_sha256: str
    expected_initial_sha256: str
    single_job_reservation_mib: int
    monitor_interval_seconds: float


def _hash_records(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def reservation_cap_mib(total_mib: int) -> int:
    if total_mib <= 2048:
        raise ConfigError("GPU total memory must exceed the 2048 MiB safety margin")
    return min(int(total_mib * 0.85), total_mib - 2048)


def launch_guard(*, total_mib: int, external_used_mib: int, reservations_mib: list[int]) -> dict[str, Any]:
    if external_used_mib < 0 or any(value <= 0 for value in reservations_mib):
        raise ConfigError("GPU occupancy and reservations must be positive bounded values")
    cap = reservation_cap_mib(total_mib)
    requested = sum(reservations_mib)
    accepted = external_used_mib + requested <= cap
    return {
        "accepted": accepted,
        "total_mib": total_mib,
        "external_used_mib": external_used_mib,
        "reservation_cap_mib": cap,
        "reservations_mib": reservations_mib,
        "requested_mib": requested,
        "projected_used_mib": external_used_mib + requested,
        "reason": None if accepted else "EXTERNAL_OCCUPANCY_PLUS_RESERVATION_EXCEEDS_CAP",
    }


def select_common_batch(results: list[dict[str, Any]], *, total_mib: int, external_used_mib: int) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    cap = reservation_cap_mib(total_mib)
    for result in results:
        if result.get("status") != "SUCCEEDED":
            continue
        peak = int(result["memory"]["torch_peak_reserved_mib"])
        reservation = peak + 512
        if external_used_mib + 2 * reservation <= cap:
            candidates.append({"batch": int(result["batch"]), "reservation_mib": reservation})
    if not candidates:
        raise ArtifactError("no successful batch has enough measured margin for two identical jobs")
    selected = max(candidates, key=lambda row: row["batch"])
    return {
        **selected,
        "reason": "largest successful candidate whose two measured reservations fit the S6 cap",
        "formal_config_frozen": False,
    }


def load_s6_campaign(config_path: Path, project_root: Path) -> S6Campaign:
    root = project_root.resolve(strict=True)
    path = config_path.resolve(strict=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or raw.get("status") != "frozen_profile_contract":
        raise ConfigError("S6 requires schema_version=1 and status=frozen_profile_contract")
    if raw.get("scope") != "profiling_only_no_formal_training":
        raise ConfigError("S6 config may only authorize profiling")
    batches = tuple(int(value) for value in raw["gpu_profile"]["batches"])
    if batches != (4, 8, 16):
        raise ConfigError("S6 batch candidates must be exactly 4, 8, 16")
    effective = raw["gpu_profile"]["effective_args"]
    required = {
        "imgsz": 640, "optimizer": "AdamW", "lr0": 0.001,
        "weight_decay": 0.0005, "mask_ratio": 1, "overlap_mask": False,
        "amp": True, "compile": False,
    }
    if any(effective.get(key) != value for key, value in required.items()):
        raise ConfigError("S6 effective GPU arguments differ from the approved profile contract")
    if any("test" in str(value).casefold() for value in raw.get("inputs", {}).values()):
        raise ConfigError("Test inputs are forbidden in S6")

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return (candidate if candidate.is_absolute() else root / candidate).resolve(strict=False)

    artifact_root = resolve(raw["paths"]["artifact_root"])
    artifacts_root = (root / "artifacts").resolve(strict=False)
    if artifact_root == artifacts_root or not _inside(artifact_root, artifacts_root):
        raise ConfigError("S6 output must be a dedicated artifacts subdirectory")
    if _inside(artifact_root, root / "Dataset"):
        raise ConfigError("S6 output may not overlap Dataset")
    s5 = load_s5_campaign(resolve(raw["paths"]["s5_config"]), root)
    expected_s5 = raw["expected_s5_hashes"]
    observed_s5 = {
        "config_sha256": s5.config_sha256,
        "code_sha256": s5.code_sha256,
        "pretrained_sha256": s5.pretrained_sha256,
        "source_zip_sha256": s5.source_zip_sha256,
    }
    if observed_s5 != expected_s5:
        raise ArtifactError(f"S5 immutable contract changed: {observed_s5}")
    source_manifest = resolve(raw["inputs"]["s4_source_manifest"])
    resolution_manifest = resolve(raw["inputs"]["environment_resolution"])
    for required_path in (source_manifest, resolution_manifest):
        if not required_path.is_file():
            raise ConfigError(f"S6 input is missing: {required_path}")
        if _inside(required_path, artifact_root):
            raise ConfigError("S6 input may not be inside S6 output")
    expected_inputs = raw["expected_input_hashes"]
    observed_inputs = {
        "s4_source_manifest_sha256": sha256_file(source_manifest),
        "environment_resolution_sha256": sha256_file(resolution_manifest),
    }
    if observed_inputs != expected_inputs:
        raise ArtifactError(f"S4 immutable input changed: {observed_inputs}")
    code_files = sorted((root / "capstone_lab" / "profiling").glob("*.py"))
    code_sha256 = _hash_records([
        {"path": file.relative_to(root).as_posix(), "sha256": sha256_file(file)}
        for file in code_files
    ])
    config_sha256 = hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()
    input_sha256 = _hash_records([
        {"name": key, "sha256": value}
        for key, value in sorted({**observed_s5, **observed_inputs}.items())
    ])
    cpu = dict(raw["cpu_smoke"])
    if int(cpu["sample_count"]) < 1 or int(cpu["sample_count"]) > 4:
        raise ConfigError("S6 CPU smoke is intentionally limited to 1-4 samples per condition")
    return S6Campaign(
        project_root=root, source_path=path, artifact_root=artifact_root, s5=s5,
        batches=batches, warmup_steps=int(raw["gpu_profile"]["warmup_steps"]),
        measure_steps=int(raw["gpu_profile"]["measure_steps"]),
        cpu_workers=int(cpu.pop("workers")), cpu_smoke=cpu,
        source_manifest=source_manifest, resolution_manifest=resolution_manifest,
        config_sha256=config_sha256, input_sha256=input_sha256, code_sha256=code_sha256,
        expected_initial_sha256=str(raw["expected_initial_state_sha256"]),
        single_job_reservation_mib=int(raw["gpu_profile"]["single_job_reservation_mib"]),
        monitor_interval_seconds=float(raw["monitor_interval_seconds"]),
    )
