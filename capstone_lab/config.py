from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigError

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SENTINELS = {"UNDECIDED", "UNKNOWN", ""}


@dataclass(frozen=True)
class Resources:
    cpu: int
    gpu_mib: int


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    dependencies: tuple[str, ...]
    resources: Resources
    duration_seconds: float
    max_attempts: int
    fail_attempts: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dependencies"] = list(self.dependencies)
        value["fail_attempts"] = list(self.fail_attempts)
        return value


@dataclass(frozen=True)
class CampaignConfig:
    source_path: Path
    schema_version: int
    status: str
    campaign_name: str
    artifact_root: Path
    protected_roots: tuple[Path, ...]
    input_files: tuple[Path, ...]
    limits: Resources
    jobs: tuple[JobSpec, ...]
    resolved: dict[str, Any]
    config_hash: str
    input_hash: str
    input_records: tuple[dict[str, Any], ...]
    unresolved_field: str | None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(base: Path, raw: Any, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{field} must be a non-empty path string")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _overlaps(left: Path, right: Path) -> bool:
    return _is_relative_to(left, right) or _is_relative_to(right, left)


def _find_sentinel(value: Any, prefix: str = "") -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            found = _find_sentinel(item, f"{prefix}.{key}" if prefix else str(key))
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_sentinel(item, f"{prefix}[{index}]")
            if found:
                return found
    elif isinstance(value, str) and value.strip().upper() in _SENTINELS:
        return prefix or "<root>"
    return None


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be an object")
    return value


def _require_int(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ConfigError(f"{field} contains unsupported keys: {sorted(unknown)}")


def _input_snapshot(paths: tuple[Path, ...]) -> tuple[tuple[dict[str, Any], ...], str]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise ConfigError(f"input file does not exist: {path}")
        if not path.is_file():
            raise ConfigError(
                f"S0 fingerprints explicit files only; use a frozen manifest for directories: {path}"
            )
        records.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    records.sort(key=lambda item: item["path"].casefold())
    frozen = tuple(records)
    return frozen, hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


def _check_cycle(jobs: tuple[JobSpec, ...]) -> None:
    by_id = {job.job_id: job for job in jobs}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(job_id: str) -> None:
        if job_id in visiting:
            raise ConfigError(f"job dependency cycle includes {job_id}")
        if job_id in visited:
            return
        visiting.add(job_id)
        for dependency in by_id[job_id].dependencies:
            visit(dependency)
        visiting.remove(job_id)
        visited.add(job_id)

    for key in by_id:
        visit(key)


def load_config(path: str | Path, *, require_runnable: bool = False) -> CampaignConfig:
    source_path = Path(path).resolve(strict=True)
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"configuration must be valid JSON: {exc}") from exc
    raw = _require_dict(raw, "configuration")
    _reject_unknown(
        raw,
        {"schema_version", "status", "campaign_name", "paths", "resources", "jobs"},
        "configuration",
    )

    if raw.get("schema_version") != 1:
        raise ConfigError("schema_version must equal 1")
    status = raw.get("status")
    if status not in {"draft", "frozen"}:
        raise ConfigError("status must be draft or frozen")
    if require_runnable and status != "frozen":
        raise ConfigError("draft configuration cannot be executed; freeze it after review")
    sentinel = _find_sentinel(raw)
    if require_runnable and sentinel:
        raise ConfigError(f"runnable configuration contains unresolved value at {sentinel}")

    campaign_name = raw.get("campaign_name")
    if not isinstance(campaign_name, str) or not _ID_RE.fullmatch(campaign_name):
        raise ConfigError("campaign_name must be a safe identifier")

    path_section = _require_dict(raw.get("paths"), "paths")
    _reject_unknown(path_section, {"artifact_root", "protected_roots", "input_files"}, "paths")
    base = source_path.parent
    artifact_root = _resolve(base, path_section.get("artifact_root"), "paths.artifact_root")
    protected_raw = path_section.get("protected_roots", [])
    inputs_raw = path_section.get("input_files", [])
    if not isinstance(protected_raw, list) or not isinstance(inputs_raw, list):
        raise ConfigError("protected_roots and input_files must be lists")
    protected_roots = tuple(_resolve(base, item, "paths.protected_roots") for item in protected_raw)
    input_files = tuple(_resolve(base, item, "paths.input_files") for item in inputs_raw)
    for protected in protected_roots:
        if _overlaps(artifact_root, protected):
            raise ConfigError(
                f"artifact_root overlaps protected source: {artifact_root} <-> {protected}"
            )
    for input_file in input_files:
        if _is_relative_to(input_file, artifact_root):
            raise ConfigError(f"input file cannot be inside artifact_root: {input_file}")

    resources = _require_dict(raw.get("resources"), "resources")
    _reject_unknown(resources, {"cpu_slots", "gpu_mib"}, "resources")
    limits = Resources(
        cpu=_require_int(resources.get("cpu_slots"), "resources.cpu_slots", 1),
        gpu_mib=_require_int(resources.get("gpu_mib"), "resources.gpu_mib", 0),
    )

    raw_jobs = raw.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ConfigError("jobs must be a non-empty list")
    jobs: list[JobSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_jobs):
        item = _require_dict(item, f"jobs[{index}]")
        _reject_unknown(
            item,
            {
                "job_id",
                "dependencies",
                "resources",
                "duration_seconds",
                "max_attempts",
                "fail_attempts",
            },
            f"jobs[{index}]",
        )
        job_id = item.get("job_id")
        if not isinstance(job_id, str) or not _ID_RE.fullmatch(job_id) or job_id in seen:
            raise ConfigError(f"jobs[{index}].job_id is invalid or duplicated")
        seen.add(job_id)
        dependencies = item.get("dependencies", [])
        fail_attempts = item.get("fail_attempts", [])
        if not isinstance(dependencies, list) or not all(isinstance(x, str) for x in dependencies):
            raise ConfigError(f"jobs[{index}].dependencies must be a string list")
        if not isinstance(fail_attempts, list):
            raise ConfigError(f"jobs[{index}].fail_attempts must be a list")
        request = _require_dict(item.get("resources"), f"jobs[{index}].resources")
        _reject_unknown(request, {"cpu", "gpu_mib"}, f"jobs[{index}].resources")
        cpu = _require_int(request.get("cpu"), f"jobs[{index}].resources.cpu", 1)
        gpu_mib = _require_int(request.get("gpu_mib"), f"jobs[{index}].resources.gpu_mib", 0)
        if cpu > limits.cpu or gpu_mib > limits.gpu_mib:
            raise ConfigError(f"job {job_id} requests more resources than the campaign limit")
        duration = item.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not 0 <= duration <= 30:
            raise ConfigError(f"jobs[{index}].duration_seconds must be between 0 and 30")
        max_attempts = _require_int(item.get("max_attempts", 1), f"jobs[{index}].max_attempts", 1)
        attempts = tuple(_require_int(x, f"jobs[{index}].fail_attempts", 1) for x in fail_attempts)
        if any(x > max_attempts for x in attempts):
            raise ConfigError(f"job {job_id} has fail_attempt beyond max_attempts")
        jobs.append(
            JobSpec(
                job_id=job_id,
                dependencies=tuple(dependencies),
                resources=Resources(cpu=cpu, gpu_mib=gpu_mib),
                duration_seconds=float(duration),
                max_attempts=max_attempts,
                fail_attempts=attempts,
            )
        )
    for job in jobs:
        unknown = set(job.dependencies) - seen
        if unknown or job.job_id in job.dependencies:
            raise ConfigError(f"job {job.job_id} has invalid dependencies: {sorted(unknown)}")
    frozen_jobs = tuple(jobs)
    _check_cycle(frozen_jobs)
    input_records, input_hash = _input_snapshot(input_files)

    resolved = {
        "schema_version": 1,
        "status": status,
        "campaign_name": campaign_name,
        "paths": {
            "artifact_root": str(artifact_root),
            "protected_roots": [str(item) for item in protected_roots],
            "input_files": [str(item) for item in input_files],
        },
        "resources": {"cpu_slots": limits.cpu, "gpu_mib": limits.gpu_mib},
        "jobs": [job.to_dict() for job in frozen_jobs],
    }
    config_hash = hashlib.sha256(canonical_json(resolved).encode("utf-8")).hexdigest()
    return CampaignConfig(
        source_path=source_path,
        schema_version=1,
        status=status,
        campaign_name=campaign_name,
        artifact_root=artifact_root,
        protected_roots=protected_roots,
        input_files=input_files,
        limits=limits,
        jobs=frozen_jobs,
        resolved=resolved,
        config_hash=config_hash,
        input_hash=input_hash,
        input_records=input_records,
        unresolved_field=sentinel,
    )
