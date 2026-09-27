from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from capstone_lab.config import canonical_json, sha256_file
from capstone_lab.errors import ManifestError

from .contract import (
    CONDITIONS,
    CandidateFeature,
    CandidatePlan,
    FixtureConfig,
    build_candidate_plans,
    verify_legacy_contract,
)
from .scoring import UsageState, select_candidate


@dataclass(frozen=True)
class FeatureTask:
    fixture_path: str
    fixture_sha256: str
    sample_id: str
    foreground_id: str
    position_id: str
    candidate_seed: str
    delay_ms: int = 0
    fail: bool = False


@dataclass(frozen=True)
class RenderTask:
    sample_id: str
    condition_id: str
    rendering_seed: str
    selection_metadata: dict[str, Any]
    delay_ms: int = 0
    fail: bool = False


def _worker_init() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OPENCV_FOR_THREADS_NUM",
    ):
        os.environ[name] = "1"


def _unit_interval(digest: bytes, offset: int) -> float:
    value = int.from_bytes(digest[offset : offset + 4], "big")
    return value / float(2**32 - 1)


def compute_fixture_features(task: FeatureTask) -> CandidateFeature:
    if task.delay_ms:
        time.sleep(task.delay_ms / 1000.0)
    if task.fail:
        raise RuntimeError(f"injected feature failure for {task.sample_id}")
    fixture = Path(task.fixture_path)
    if sha256_file(fixture) != task.fixture_sha256:
        raise RuntimeError("fixture changed after dispatch")
    candidate_id = f"{task.foreground_id}@{task.position_id}"
    digest = hashlib.sha256(
        (
            task.fixture_sha256
            + "|"
            + task.sample_id
            + "|"
            + candidate_id
            + "|"
            + task.candidate_seed
        ).encode("utf-8")
    ).digest()
    return CandidateFeature(
        sample_id=task.sample_id,
        candidate_id=candidate_id,
        foreground_id=task.foreground_id,
        position_id=task.position_id,
        tone=_unit_interval(digest, 0),
        texture=_unit_interval(digest, 4),
        edge=_unit_interval(digest, 8),
        placement=_unit_interval(digest, 12),
        artifact_penalty=_unit_interval(digest, 16),
        eligible=True,
        eligibility_reasons=("fixture_declared_valid",),
    )


def render_fixture(task: RenderTask) -> bytes:
    if task.delay_ms:
        time.sleep(task.delay_ms / 1000.0)
    if task.fail:
        raise RuntimeError(f"injected render failure for {task.sample_id}")
    record = {
        "fixture_renderer": "deterministic_json_boundary_v1",
        "sample_id": task.sample_id,
        "condition_id": task.condition_id,
        "rendering_seed": task.rendering_seed,
        "selection": task.selection_metadata,
    }
    return (canonical_json(record) + "\n").encode("utf-8")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_fixture_output(
    output_dir: Path, *, allowed_root: Path, protected_paths: Iterable[Path]
) -> Path:
    root = allowed_root.resolve(strict=True)
    output = output_dir.resolve(strict=False)
    if output == root or not _inside(output, root):
        raise ManifestError("fixture output must be a non-root path inside allowed_root")
    for protected in protected_paths:
        source = protected.resolve(strict=False)
        if output == source or _inside(output, source) or _inside(source, output):
            raise ManifestError(f"fixture output overlaps protected source: {source}")
    return output


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _plan_tasks(
    plans: list[CandidatePlan],
    fixture_path: Path,
    *,
    delay_order: str,
    fail_sample_ids: set[str],
) -> list[FeatureTask]:
    fixture_hash = sha256_file(fixture_path)
    raw: list[tuple[CandidatePlan, str, str]] = []
    for plan in plans:
        for foreground_id in plan.foreground_candidate_ids:
            for position_id in plan.position_candidate_ids:
                raw.append((plan, foreground_id, position_id))
    tasks: list[FeatureTask] = []
    total = len(raw)
    for index, (plan, foreground_id, position_id) in enumerate(raw):
        if delay_order == "reverse":
            delay_ms = (total - index) % 5
        elif delay_order == "forward":
            delay_ms = index % 5
        elif delay_order == "none":
            delay_ms = 0
        else:
            raise ManifestError("delay_order must be none, forward, or reverse")
        tasks.append(
            FeatureTask(
                fixture_path=str(fixture_path),
                fixture_sha256=fixture_hash,
                sample_id=plan.sample_id,
                foreground_id=foreground_id,
                position_id=position_id,
                candidate_seed=plan.candidate_seed,
                delay_ms=delay_ms,
                fail=plan.sample_id in fail_sample_ids,
            )
        )
    return tasks


def run_fixture_campaign(
    *,
    fixture_path: Path,
    legacy_source_path: Path,
    output_dir: Path,
    allowed_root: Path,
    protected_paths: Iterable[Path],
    workers: int,
    config: FixtureConfig | None = None,
    delay_order: str = "none",
    fail_sample_ids: Iterable[str] = (),
) -> dict[str, Any]:
    if workers not in {1, 2, 6}:
        raise ManifestError("S3 fixture workers must be one of 1, 2, or 6")
    fixture = fixture_path.resolve(strict=True)
    legacy = legacy_source_path.resolve(strict=True)
    output = validate_fixture_output(
        output_dir,
        allowed_root=allowed_root,
        protected_paths=tuple(protected_paths) + (fixture, legacy),
    )
    fixture_config = config or FixtureConfig()
    plans = build_candidate_plans(fixture_config)
    legacy_contract = verify_legacy_contract(legacy)
    fail_ids = set(fail_sample_ids)
    unknown_failures = fail_ids - {plan.sample_id for plan in plans}
    if unknown_failures:
        raise ManifestError(f"unknown injected failure samples: {sorted(unknown_failures)}")

    tasks = _plan_tasks(
        plans, fixture, delay_order=delay_order, fail_sample_ids=fail_ids
    )
    features_by_sample: dict[str, list[CandidateFeature]] = {
        plan.sample_id: [] for plan in plans
    }
    errors_by_sample: dict[str, list[str]] = {plan.sample_id: [] for plan in plans}
    states = {
        condition: UsageState(fixture_config.foreground_ids)
        for condition in CONDITIONS
    }
    sample_results: list[dict[str, Any]] = []
    output_hashes: dict[str, str] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=_worker_init
    ) as pool:
        future_to_task = {
            pool.submit(compute_fixture_features, task): task for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                features_by_sample[task.sample_id].append(future.result())
            except Exception as exc:
                errors_by_sample[task.sample_id].append(
                    f"{type(exc).__name__}: {exc}"
                )

        for plan in sorted(plans, key=lambda item: item.sample_id):
            if errors_by_sample[plan.sample_id]:
                sample_results.append(
                    {
                        "sample_id": plan.sample_id,
                        "status": "FEATURE_FAILED",
                        "errors": sorted(set(errors_by_sample[plan.sample_id])),
                        "state_updated": False,
                    }
                )
                continue
            selected: dict[str, tuple[CandidateFeature, dict[str, Any]]] = {}
            sample_failed = False
            for condition in CONDITIONS:
                choice = select_candidate(
                    condition,
                    plan,
                    features_by_sample[plan.sample_id],
                    states[condition],
                    fixture_config.campaign_seed,
                )
                if choice is None:
                    sample_failed = True
                    break
                selected[condition] = choice
            if sample_failed:
                sample_results.append(
                    {
                        "sample_id": plan.sample_id,
                        "status": "SELECTION_FAILED",
                        "errors": ["no eligible candidate"],
                        "state_updated": False,
                    }
                )
                continue

            render_futures = {}
            for index, condition in enumerate(CONDITIONS):
                feature, metadata = selected[condition]
                enriched = dict(metadata)
                enriched.update(
                    {
                        "foreground_id": feature.foreground_id,
                        "position_id": feature.position_id,
                        "background_id": plan.background_id,
                        "mode": plan.mode,
                        "domain": plan.domain,
                        "difficulty": plan.difficulty,
                        "scale_bucket": plan.scale_bucket,
                        "scale_ratio": plan.scale_ratio,
                        "candidate_seed": plan.candidate_seed,
                        "rendering_seed": plan.rendering_seed,
                    }
                )
                selected[condition] = (feature, enriched)
                render_task = RenderTask(
                    sample_id=plan.sample_id,
                    condition_id=condition,
                    rendering_seed=plan.rendering_seed,
                    selection_metadata=enriched,
                    delay_ms=(len(CONDITIONS) - index) % 3
                    if delay_order == "reverse"
                    else index % 3
                    if delay_order == "forward"
                    else 0,
                )
                render_futures[pool.submit(render_fixture, render_task)] = condition

            rendered: dict[str, bytes] = {}
            render_errors: list[str] = []
            for future in as_completed(render_futures):
                condition = render_futures[future]
                try:
                    rendered[condition] = future.result()
                except Exception as exc:
                    render_errors.append(f"{condition}: {type(exc).__name__}: {exc}")
            if render_errors:
                sample_results.append(
                    {
                        "sample_id": plan.sample_id,
                        "status": "RENDER_FAILED",
                        "errors": sorted(render_errors),
                        "state_updated": False,
                    }
                )
                continue

            selections: dict[str, Any] = {}
            for condition in CONDITIONS:
                feature, metadata = selected[condition]
                states[condition].commit(
                    feature.foreground_id, plan.domain, plan.background_id
                )
                relative = Path(plan.sample_id) / f"{condition}.fixture.json"
                content = rendered[condition]
                _atomic_write(output / relative, content)
                digest = hashlib.sha256(content).hexdigest()
                output_hashes[relative.as_posix()] = digest
                selections[condition] = metadata
            sample_results.append(
                {
                    "sample_id": plan.sample_id,
                    "status": "SUCCEEDED",
                    "state_updated": True,
                    "selections": selections,
                }
            )

    deterministic_payload = {
        "schema_version": 1,
        "fixture_sha256": sha256_file(fixture),
        "seed_rule": "SHA256(campaign_seed|sample_id|stage|attempt[|condition_id])",
        "plans": [plan.to_dict() for plan in plans],
        "samples": sample_results,
        "output_hashes": dict(sorted(output_hashes.items())),
        "usage_states": {
            condition: states[condition].snapshot() for condition in CONDITIONS
        },
        "legacy_contract": {
            "artifact_assignments": legacy_contract["artifact_assignments"],
            "effective_artifact_penalty_weight": legacy_contract[
                "effective_artifact_penalty_weight"
            ],
            "effective_artifact_line": legacy_contract["effective_artifact_line"],
            "mode_config_line": legacy_contract["mode_config_line"],
            "mode_weights": legacy_contract["mode_weights"],
        },
        "actual_legacy_renderer_executed": False,
        "actual_legacy_renderer_blocker": "NumPy/OpenCV are not installed in the base Python environment",
    }
    payload_bytes = (canonical_json(deterministic_payload) + "\n").encode("utf-8")
    return {
        "status": "VERIFIED"
        if all(row["status"] == "SUCCEEDED" for row in sample_results)
        else "PARTIAL_FAILURE",
        "workers": workers,
        "single_shared_process_pool": True,
        "deterministic_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "deterministic_payload": deterministic_payload,
    }
