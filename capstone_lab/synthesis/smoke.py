from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import os
import random
import shutil
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from capstone_lab.config import canonical_json, sha256_file
from capstone_lab.errors import ManifestError

from .contract import (
    CONDITIONS,
    CandidateFeature,
    CandidatePlan,
    derive_seed,
    verify_legacy_contract,
)
from .executor import _inside, _worker_init, validate_fixture_output
from .scoring import UsageState, select_candidate


SMOKE_SCHEMA_VERSION = 1
SOURCE_ALGORITHM = "s4_smoke_source_sha256_rank_v1"
SCALE_RANGES = {
    "05to10": (0.05, 0.10),
    "10to15": (0.10, 0.15),
    "15to20": (0.15, 0.20),
}
LEGACY_FEATURE_FUNCTIONS = (
    "load_image",
    "resize_background_long_side",
    "binarize_mask",
    "crop_to_mask_bbox",
    "resize_object_to_target_scale",
    "apply_object_texture_diversity",
    "extract_background_patch",
    "compute_margin_score",
    "get_effective_center_y_range",
    "compute_object_background_match_score",
)
LEGACY_RENDER_FUNCTIONS = (
    "shrink_mask_for_blending",
    "apply_tone_matching",
    "build_alpha_from_mask",
    "blend_object",
    "apply_post_blend_edge_smoothing",
    "build_transformed_binary_mask",
    "mask_to_polygon_lines",
)


class InjectedInterruption(RuntimeError):
    """Test-only interruption used to exercise coordinator resume."""


@dataclass(frozen=True)
class SmokeConfig:
    campaign_seed: int = 7
    sample_count: int = 20
    candidates_per_sample: int = 3
    positions_per_candidate: int = 2
    target_long_side: int = 384
    modes: tuple[str, ...] = ("natural", "semi")
    difficulties: tuple[str, ...] = ("easy", "medium", "hard", "extreme")
    scale_buckets: tuple[str, ...] = tuple(SCALE_RANGES)

    def validate(self, *, enforce_smoke_minimum: bool = True) -> None:
        minimum = 20 if enforce_smoke_minimum else 1
        if self.sample_count < minimum or self.sample_count > 50:
            raise ManifestError(
                f"S4 sample_count must be between {minimum} and 50 per condition"
            )
        if self.candidates_per_sample < 2:
            raise ManifestError("S4 needs at least two foreground candidates")
        if self.positions_per_candidate < 2:
            raise ManifestError("S4 needs at least two position candidates")
        if self.target_long_side < 96 or self.target_long_side > 1280:
            raise ManifestError("S4 target_long_side must be between 96 and 1280")
        if not self.modes or any(mode not in {"natural", "semi"} for mode in self.modes):
            raise ManifestError("S4 smoke modes must be natural/semi")
        if not self.difficulties:
            raise ManifestError("S4 difficulties must not be empty")
        if not self.scale_buckets or any(
            bucket not in SCALE_RANGES for bucket in self.scale_buckets
        ):
            raise ManifestError("S4 contains an unsupported scale bucket")

    def deterministic_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("modes", "difficulties", "scale_buckets"):
            value[key] = list(value[key])
        return value


@dataclass(frozen=True)
class ActualFeatureTask:
    legacy_source_path: str
    project_root: str
    plan: dict[str, Any]
    background: dict[str, Any]
    foreground: dict[str, Any]
    position_id: str
    target_long_side: int
    fail: bool = False
    terminate: bool = False
    full_policy: bool = False


@dataclass(frozen=True)
class ActualRenderTask:
    feature_task: ActualFeatureTask
    condition_id: str
    selection_metadata: dict[str, Any]
    corrupt_mask: bool = False
    fail: bool = False


_LEGACY_MODULES: dict[str, Any] = {}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes, *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        if path.read_bytes() == content:
            return
        raise ManifestError(f"immutable output already exists with different content: {path}")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        from capstone_lab.campaign.io import atomic_replace
        atomic_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative_file_record(project_root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not _inside(resolved, project_root):
        raise ManifestError(f"smoke source is outside project root: {resolved}")
    return {
        "path": resolved.relative_to(project_root).as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.resolve(strict=True).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"invalid JSONL at {path}:{line_number}") from exc
            source_id = row.get("object_source_id")
            if not isinstance(source_id, str) or not source_id:
                raise ManifestError(f"missing object_source_id at {path}:{line_number}")
            if source_id in seen_ids:
                raise ManifestError(f"duplicate object_source_id in resolution: {source_id}")
            seen_ids.add(source_id)
            rows.append(row)
    return rows


def _validate_resolution(project_root: Path, resolution_path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(resolution_path)
    for row in rows:
        if row.get("status") != "RESOLVED":
            raise ManifestError("S4 requires every environment override row to be RESOLVED")
        environment = row.get("chosen_environment")
        if environment not in {"snow", "non_snow"}:
            raise ManifestError("invalid chosen_environment in resolution manifest")
        for field, hash_field, kind in (
            ("image", "image_sha256", "images"),
            ("mask", "mask_sha256", "masks"),
        ):
            source = (project_root / str(row.get(field, ""))).resolve(strict=True)
            expected_parent = (
                project_root / "Dataset" / "object_pool" / environment / kind
            ).resolve(strict=True)
            if not _inside(source, expected_parent):
                raise ManifestError(
                    f"resolution path disagrees with disk environment for {row['object_source_id']}"
                )
            if sha256_file(source) != row.get(hash_field):
                raise ManifestError(
                    f"resolution source hash changed for {row['object_source_id']}:{field}"
                )
    return rows


def build_smoke_source_manifest(
    *,
    project_root: Path,
    output_path: Path,
    resolution_manifest_path: Path,
    max_foregrounds_per_environment: int = 48,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    if max_foregrounds_per_environment < 8:
        raise ManifestError("S4 smoke manifest needs at least eight foregrounds per environment")
    resolution = resolution_manifest_path.resolve(strict=True)
    resolution_rows = _validate_resolution(root, resolution)

    background_root = root / "Dataset" / "synthesis_demo" / "BackGround"
    object_root = root / "Dataset" / "object_pool"
    if not background_root.is_dir() or not object_root.is_dir():
        raise ManifestError("S4 source roots are missing")

    backgrounds: list[dict[str, Any]] = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp"}
    for domain_dir in sorted(background_root.iterdir(), key=lambda item: item.name):
        if not domain_dir.is_dir():
            continue
        environment = "snow" if domain_dir.name == "snow" else "non_snow"
        for image_path in sorted(domain_dir.iterdir(), key=lambda item: item.name):
            if not image_path.is_file() or image_path.suffix.lower() not in valid_extensions:
                continue
            record = _relative_file_record(root, image_path)
            record.update(
                {
                    "background_id": f"{domain_dir.name}/{image_path.stem}",
                    "domain": domain_dir.name,
                    "environment": environment,
                }
            )
            backgrounds.append(record)
    if not backgrounds:
        raise ManifestError("S4 smoke background pool is empty")

    foregrounds: list[dict[str, Any]] = []
    for environment in ("non_snow", "snow"):
        image_dir = object_root / environment / "images"
        mask_dir = object_root / environment / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            continue
        masks = {
            item.stem: item
            for item in mask_dir.iterdir()
            if item.is_file() and item.suffix.lower() in valid_extensions
        }
        pairs = []
        for image_path in image_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in valid_extensions:
                continue
            mask_path = masks.get(image_path.stem)
            if mask_path is None:
                continue
            rank = hashlib.sha256(
                f"{SOURCE_ALGORITHM}|{environment}|{image_path.stem}".encode("utf-8")
            ).hexdigest()
            pairs.append((rank, image_path.stem, image_path, mask_path))
        for rank, source_id, image_path, mask_path in sorted(pairs)[
            :max_foregrounds_per_environment
        ]:
            image_record = _relative_file_record(root, image_path)
            mask_record = _relative_file_record(root, mask_path)
            foregrounds.append(
                {
                    "foreground_id": source_id,
                    "environment": environment,
                    "rank_sha256": rank,
                    "image_path": image_record["path"],
                    "image_sha256": image_record["sha256"],
                    "image_bytes": image_record["bytes"],
                    "mask_path": mask_record["path"],
                    "mask_sha256": mask_record["sha256"],
                    "mask_bytes": mask_record["bytes"],
                }
            )
    available_environments = {row["environment"] for row in foregrounds}
    needed_environments = {row["environment"] for row in backgrounds}
    if not needed_environments.issubset(available_environments):
        raise ManifestError("a smoke background environment has no foreground source")

    payload = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "algorithm": SOURCE_ALGORITHM,
        "project_relative_only": True,
        "environment_decision_rule": "disk_directory_is_authoritative",
        "resolution_manifest": resolution.relative_to(root).as_posix(),
        "resolution_manifest_sha256": sha256_file(resolution),
        "resolution_rows_verified": len(resolution_rows),
        "backgrounds": backgrounds,
        "foregrounds": foregrounds,
    }
    content = _json_bytes(payload)
    output = output_path.resolve(strict=False)
    if not _inside(output, root) or _inside(output, (root / "Dataset").resolve()):
        raise ManifestError("smoke source manifest must be outside Dataset and inside project")
    _atomic_write(output, content, overwrite=False)
    return {
        "status": "VERIFIED",
        "path": str(output),
        "sha256": _sha256_bytes(content),
        "backgrounds": len(backgrounds),
        "foregrounds": len(foregrounds),
        "resolution_rows_verified": len(resolution_rows),
    }


def _load_source_manifest(
    project_root: Path, manifest_path: Path, resolution_path: Path
) -> tuple[dict[str, Any], str]:
    path = manifest_path.resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "algorithm",
        "project_relative_only",
        "environment_decision_rule",
        "resolution_manifest",
        "resolution_manifest_sha256",
        "resolution_rows_verified",
        "backgrounds",
        "foregrounds",
    }
    if set(payload) != required:
        raise ManifestError("S4 source manifest schema keys differ from v1")
    if payload["schema_version"] != SMOKE_SCHEMA_VERSION:
        raise ManifestError("unsupported S4 source manifest schema")
    if payload["algorithm"] != SOURCE_ALGORITHM:
        raise ManifestError("unsupported S4 source selection algorithm")
    if payload["environment_decision_rule"] != "disk_directory_is_authoritative":
        raise ManifestError("S4 source manifest does not use the approved environment rule")
    resolution = resolution_path.resolve(strict=True)
    _validate_resolution(project_root, resolution)
    if sha256_file(resolution) != payload["resolution_manifest_sha256"]:
        raise ManifestError("environment resolution manifest hash changed")

    seen_backgrounds: set[str] = set()
    seen_foregrounds: set[str] = set()
    for row in payload["backgrounds"]:
        source_id = row.get("background_id")
        if source_id in seen_backgrounds:
            raise ManifestError(f"duplicate background ID: {source_id}")
        seen_backgrounds.add(source_id)
        source = (project_root / row["path"]).resolve(strict=True)
        if not _inside(source, project_root / "Dataset"):
            raise ManifestError("background source escaped Dataset")
        if sha256_file(source) != row["sha256"]:
            raise ManifestError(f"background source hash changed: {source_id}")
    for row in payload["foregrounds"]:
        source_id = row.get("foreground_id")
        if source_id in seen_foregrounds:
            raise ManifestError(f"duplicate foreground ID: {source_id}")
        seen_foregrounds.add(source_id)
        environment = row.get("environment")
        if environment not in {"snow", "non_snow"}:
            raise ManifestError(f"invalid foreground environment: {source_id}")
        for field, hash_field, kind in (
            ("image_path", "image_sha256", "images"),
            ("mask_path", "mask_sha256", "masks"),
        ):
            source = (project_root / row[field]).resolve(strict=True)
            expected = (
                project_root / "Dataset" / "object_pool" / environment / kind
            ).resolve(strict=True)
            if not _inside(source, expected):
                raise ManifestError(f"foreground disk environment mismatch: {source_id}")
            if sha256_file(source) != row[hash_field]:
                raise ManifestError(f"foreground source hash changed: {source_id}:{field}")
    if not seen_backgrounds or not seen_foregrounds:
        raise ManifestError("S4 source manifest has an empty pool")
    return payload, sha256_file(path)


def build_actual_candidate_plans(
    config: SmokeConfig, source_manifest: dict[str, Any]
) -> list[CandidatePlan]:
    backgrounds = sorted(source_manifest["backgrounds"], key=lambda row: row["background_id"])
    foregrounds_by_environment: dict[str, list[str]] = {"snow": [], "non_snow": []}
    for row in source_manifest["foregrounds"]:
        foregrounds_by_environment[row["environment"]].append(row["foreground_id"])
    for values in foregrounds_by_environment.values():
        values.sort()

    plans: list[CandidatePlan] = []
    for index in range(1, config.sample_count + 1):
        sample_id = f"s4_sample_{index:06d}"
        candidate_digest, candidate_seed = derive_seed(
            config.campaign_seed, sample_id, "candidate_plan", 0
        )
        rendering_digest, _ = derive_seed(
            config.campaign_seed, sample_id, "rendering", 0
        )
        rng = random.Random(candidate_seed)
        background = backgrounds[rng.randrange(len(backgrounds))]
        foreground_pool = foregrounds_by_environment[background["environment"]]
        if len(foreground_pool) < config.candidates_per_sample:
            raise ManifestError(
                f"not enough foregrounds for environment {background['environment']}"
            )
        foreground_ids = tuple(rng.sample(foreground_pool, config.candidates_per_sample))
        position_ids = tuple(
            f"pos_{position_index:02d}_{rng.randrange(0, 2**32):08x}"
            for position_index in range(config.positions_per_candidate)
        )
        scale_bucket = rng.choice(config.scale_buckets)
        low, high = SCALE_RANGES[scale_bucket]
        plans.append(
            CandidatePlan(
                sample_id=sample_id,
                background_id=background["background_id"],
                mode=rng.choice(config.modes),
                domain=background["domain"],
                difficulty=rng.choice(config.difficulties),
                scale_bucket=scale_bucket,
                scale_ratio=round(rng.uniform(low, high), 12),
                foreground_candidate_ids=foreground_ids,
                position_candidate_ids=position_ids,
                candidate_seed=candidate_digest,
                rendering_seed=rendering_digest,
                attempt=0,
            )
        )
    ids = [plan.sample_id for plan in plans]
    if len(ids) != len(set(ids)):
        raise ManifestError("duplicate S4 sample ID would cause a filename collision")
    return plans


def _candidate_plan_from_dict(value: dict[str, Any]) -> CandidatePlan:
    return CandidatePlan(
        sample_id=value["sample_id"],
        background_id=value["background_id"],
        mode=value["mode"],
        domain=value["domain"],
        difficulty=value["difficulty"],
        scale_bucket=value["scale_bucket"],
        scale_ratio=float(value["scale_ratio"]),
        foreground_candidate_ids=tuple(value["foreground_candidate_ids"]),
        position_candidate_ids=tuple(value["position_candidate_ids"]),
        candidate_seed=value["candidate_seed"],
        rendering_seed=value["rendering_seed"],
        attempt=int(value["attempt"]),
    )


def _legacy_module(source_path: str) -> Any:
    source = str(Path(source_path).resolve(strict=True))
    cached = _LEGACY_MODULES.get(source)
    if cached is not None:
        return cached
    module_name = "capstone_legacy_s4_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load legacy generator module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in LEGACY_FEATURE_FUNCTIONS + LEGACY_RENDER_FUNCTIONS:
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"legacy generator is missing function: {name}")
    _LEGACY_MODULES[source] = module
    return module


def _record_path(task: ActualFeatureTask, record: dict[str, Any], field: str) -> Path:
    root = Path(task.project_root).resolve(strict=True)
    path = (root / record[field]).resolve(strict=True)
    if not (_inside(path, root / "Dataset") or
            (task.full_policy and field == 'path' and _inside(path, root / 'artifacts/s8_full_backgrounds/v1/images'))):
        raise RuntimeError(f"source path escaped Dataset: {field}")
    if task.full_policy:
        prefix={'path':'path','image_path':'image','mask_path':'mask'}[field]
        size_key='_'+prefix+'_size';mtime_key='_'+prefix+'_mtime_ns'
        if size_key in record and mtime_key in record:
            stat=path.stat()
            if stat.st_size!=record[size_key] or stat.st_mtime_ns!=record[mtime_key]:
                raise RuntimeError(f"formal source stat changed: {field}")
    return path


def _prepare_actual_candidate(task: ActualFeatureTask) -> dict[str, Any]:
    legacy = _legacy_module(task.legacy_source_path)
    plan = _candidate_plan_from_dict(task.plan)
    background_path = _record_path(task, task.background, "path")
    foreground_path = _record_path(task, task.foreground, "image_path")
    mask_path = _record_path(task, task.foreground, "mask_path")
    background = legacy.load_image(background_path, legacy.cv2.IMREAD_COLOR)
    object_bgr = legacy.load_image(foreground_path, legacy.cv2.IMREAD_COLOR)
    object_mask = legacy.load_image(mask_path, legacy.cv2.IMREAD_GRAYSCALE)
    if background is None or object_bgr is None or object_mask is None:
        raise RuntimeError("legacy image decode failed")
    background = legacy.resize_background_long_side(background, task.target_long_side)
    background_records = []
    if task.full_policy:
        background_seed = hashlib.sha256((plan.candidate_seed + '|background').encode()).hexdigest()
        legacy.GLOBAL_STATE['rng'] = random.Random(int(background_seed, 16))
        background, first = legacy.apply_anti_memorization_background_transform(background, plan.domain, plan.difficulty)
        background, second = legacy.apply_domain_aware_background_augmentation(background, plan.domain, plan.difficulty)
        background_records = list(first) + list(second)
    object_mask = legacy.binarize_mask(object_mask)
    cropped_bgr, cropped_mask, _ = legacy.crop_to_mask_bbox(object_bgr, object_mask)
    if background is None or cropped_bgr is None or cropped_mask is None:
        raise RuntimeError("legacy crop/resize preparation failed")
    bg_h, bg_w = background.shape[:2]
    resized_bgr, resized_mask = legacy.resize_object_to_target_scale(
        cropped_bgr, cropped_mask, bg_h, bg_w, plan.scale_ratio
    )
    if resized_bgr is None or resized_mask is None:
        raise RuntimeError("legacy object scale resize failed")

    object_seed = hashlib.sha256(
        f"{plan.candidate_seed}|{task.foreground['foreground_id']}|object_augmentation".encode(
            "utf-8"
        )
    ).hexdigest()
    legacy.GLOBAL_STATE["rng"] = random.Random(int(object_seed, 16))
    augmented_bgr, augmentation_records = legacy.apply_object_texture_diversity(
        resized_bgr, resized_mask, plan.difficulty, plan.domain
    )

    obj_h, obj_w = resized_mask.shape[:2]
    if obj_h >= bg_h or obj_w >= bg_w:
        raise RuntimeError("resized object does not fit background")
    position_digest = hashlib.sha256(
        f"{plan.candidate_seed}|{task.foreground['foreground_id']}|{task.position_id}".encode(
            "utf-8"
        )
    ).digest()
    x_unit = int.from_bytes(position_digest[:8], "big") / float(2**64 - 1)
    y_unit = int.from_bytes(position_digest[8:16], "big") / float(2**64 - 1)
    allowed_min_y, allowed_max_y = legacy.get_effective_center_y_range(
        plan.scale_bucket, plan.domain
    )
    center_x_ratio = 0.15 + 0.70 * x_unit
    center_y_ratio = allowed_min_y + (allowed_max_y - allowed_min_y) * y_unit
    x = int(round(center_x_ratio * bg_w - obj_w * 0.5))
    y = int(round(center_y_ratio * bg_h - obj_h * 0.5))
    x = max(0, min(bg_w - obj_w, x))
    y = max(0, min(bg_h - obj_h, y))
    if task.full_policy:
        position_seed = hashlib.sha256((plan.candidate_seed + '|' + task.foreground['foreground_id'] + '|positions').encode()).hexdigest()
        legacy.GLOBAL_STATE['rng'] = random.Random(int(position_seed,16))
        positions = legacy.sample_candidate_positions(bg_h,bg_w,obj_h,obj_w,16,plan.scale_bucket,plan.domain,plan.mode)
        index = int(task.position_id[1:])
        if index >= len(positions):
            raise RuntimeError('legacy position candidate unavailable')
        x,y = positions[index]['x'],positions[index]['y']
    actual_center_y = (y + obj_h * 0.5) / float(bg_h)
    margin_score, margin_ratio = legacy.compute_margin_score(
        x, y, obj_w, obj_h, bg_w, bg_h
    )
    margin_x = int(round(bg_w * legacy.OUTER_MARGIN_RATIO))
    margin_y = int(round(bg_h * legacy.OUTER_MARGIN_RATIO))
    touches_margin = (
        x < margin_x
        or y < margin_y
        or x + obj_w > bg_w - margin_x
        or y + obj_h > bg_h - margin_y
    )
    position_info = {
        "x": x,
        "y": y,
        "obj_w": obj_w,
        "obj_h": obj_h,
        "center_y_ratio": actual_center_y,
        "allowed_min_center_y": allowed_min_y,
        "allowed_max_center_y": allowed_max_y,
        "margin_score": margin_score,
        "margin_distance_ratio": margin_ratio,
        "touches_margin": touches_margin,
    }
    patch = legacy.extract_background_patch(background, x, y, obj_w, obj_h)
    if patch is None:
        raise RuntimeError("legacy background patch extraction failed")
    raw_score = legacy.compute_object_background_match_score(
        augmented_bgr,
        resized_mask,
        patch,
        actual_center_y,
        plan.mode,
        plan.domain,
        plan.scale_bucket,
        position_info,
    )
    if raw_score is None:
        raise RuntimeError("legacy feature extractor rejected boundary band")
    raw_score["placement_score"] = max(
        0.0,
        min(
            1.0,
            float(raw_score["placement_score"]) + (-0.04 if touches_margin else 0.01),
        ),
    )
    return {
        "legacy": legacy,
        "plan": plan,
        "background": background,
        "object_bgr": augmented_bgr,
        "object_mask": resized_mask,
        "patch": patch,
        "x": x,
        "y": y,
        "raw_score": raw_score,
        "augmentation_records": list(augmentation_records),
        "background_records": background_records,
    }


def compute_actual_features(task: ActualFeatureTask) -> CandidateFeature:
    if task.terminate:
        os._exit(91)
    if task.fail:
        raise RuntimeError(f"injected worker feature failure for {task.plan['sample_id']}")
    prepared = _prepare_actual_candidate(task)
    plan = prepared["plan"]
    raw = prepared["raw_score"]
    candidate_id = f"{task.foreground['foreground_id']}@{task.position_id}"
    return CandidateFeature(
        sample_id=plan.sample_id,
        candidate_id=candidate_id,
        foreground_id=task.foreground["foreground_id"],
        position_id=task.position_id,
        tone=float(raw["tone_score"]),
        texture=float(raw["texture_score"]),
        edge=float(raw["edge_score"]),
        placement=float(raw["placement_score"]),
        artifact_penalty=float(raw["artifact_penalty"]),
        eligible=True,
        eligibility_reasons=(
            "legacy_decode_crop_scale_valid",
            "legacy_boundary_band_valid",
            "position_inside_background",
        ),
    )


def render_actual_bundle(task: ActualRenderTask) -> dict[str, Any]:
    if task.fail:
        raise RuntimeError(
            f"injected worker render failure for {task.feature_task.plan['sample_id']}/{task.condition_id}"
        )
    prepared = _prepare_actual_candidate(task.feature_task)
    legacy = prepared["legacy"]
    plan = prepared["plan"]
    legacy.GLOBAL_STATE["rng"] = random.Random(int(plan.rendering_seed, 16))
    blend_source_mask = legacy.shrink_mask_for_blending(
        prepared["object_mask"],
        plan.mode,
        object_bgr=prepared["object_bgr"],
        patch_bgr=prepared["patch"],
    )
    matched_object = legacy.apply_tone_matching(
        prepared["object_bgr"],
        blend_source_mask,
        prepared["patch"],
        plan.mode,
        difficulty_mode=plan.difficulty,
    )
    alpha = legacy.build_alpha_from_mask(
        prepared["object_mask"],
        plan.mode,
        object_bgr=prepared["object_bgr"],
        patch_bgr=prepared["patch"],
        difficulty_mode=plan.difficulty,
    )
    synthetic = legacy.blend_object(
        prepared["background"], matched_object, alpha, prepared["x"], prepared["y"]
    )
    synthetic = legacy.apply_post_blend_edge_smoothing(
        prepared["background"],
        synthetic,
        alpha,
        prepared["x"],
        prepared["y"],
        plan.difficulty,
    )
    mask = legacy.build_transformed_binary_mask(
        prepared["background"].shape[0],
        prepared["background"].shape[1],
        prepared["object_mask"],
        prepared["x"],
        prepared["y"],
    )
    polygon_lines = legacy.mask_to_polygon_lines(mask, legacy.CLASS_ID)
    if not polygon_lines:
        raise RuntimeError("legacy polygon renderer produced no polygons")
    image_ok, image_encoded = legacy.cv2.imencode(
        ".png", synthetic, [int(legacy.cv2.IMWRITE_PNG_COMPRESSION), 3]
    )
    mask_ok, mask_encoded = legacy.cv2.imencode(
        ".png", mask, [int(legacy.cv2.IMWRITE_PNG_COMPRESSION), 3]
    )
    if not image_ok or not mask_ok:
        raise RuntimeError("OpenCV failed to encode the rendered bundle")
    mask_bytes = mask_encoded.tobytes()
    if task.corrupt_mask:
        mask_bytes = b"not-a-mask"
    return {
        "image": image_encoded.tobytes(),
        "mask": mask_bytes,
        "polygon": ("\n".join(polygon_lines) + "\n").encode("utf-8"),
        "width": int(synthetic.shape[1]),
        "height": int(synthetic.shape[0]),
        "x": int(prepared["x"]),
        "y": int(prepared["y"]),
        "object_augmentations": prepared["augmentation_records"],
        **({"background_augmentations": prepared["background_records"],
            "background_signature": legacy.compute_background_variant_signature(prepared["background"]),
            "mask_pixels": int(legacy.np.count_nonzero(mask))}
           if task.feature_task.full_policy else {}),
        "legacy_feature_functions": list(LEGACY_FEATURE_FUNCTIONS),
        "legacy_renderer_functions": list(LEGACY_RENDER_FUNCTIONS),
    }


def _decode_image(content: bytes, *, grayscale: bool) -> Any:
    import cv2
    import numpy as np

    array = np.frombuffer(content, dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imdecode(array, flag)


def validate_rendered_bundle(
    *, image: bytes, mask: bytes, polygon: bytes, metadata: dict[str, Any]
) -> list[str]:
    import numpy as np

    reasons: list[str] = []
    decoded_image = _decode_image(image, grayscale=False)
    decoded_mask = _decode_image(mask, grayscale=True)
    if decoded_image is None:
        reasons.append("image_decode_failed")
    if decoded_mask is None:
        reasons.append("mask_decode_failed")
    if decoded_image is not None and decoded_mask is not None:
        if decoded_image.shape[:2] != decoded_mask.shape[:2]:
            reasons.append("image_mask_size_mismatch")
        unique = set(int(value) for value in np.unique(decoded_mask))
        if not unique.issubset({0, 255}):
            reasons.append("mask_not_binary")
        if int(np.count_nonzero(decoded_mask)) == 0:
            reasons.append("mask_empty")
    try:
        text = polygon.decode("utf-8").strip()
    except UnicodeDecodeError:
        text = ""
        reasons.append("polygon_not_utf8")
    if not text:
        reasons.append("polygon_empty")
    else:
        for line in text.splitlines():
            fields = line.split()
            if len(fields) < 7 or (len(fields) - 1) % 2:
                reasons.append("polygon_invalid_field_count")
                break
            if fields[0] != "0":
                reasons.append("polygon_invalid_class")
                break
            try:
                coordinates = [float(value) for value in fields[1:]]
            except ValueError:
                reasons.append("polygon_non_numeric")
                break
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                reasons.append("polygon_coordinate_out_of_range")
                break
    required_metadata = {
        "sample_id",
        "condition_id",
        "background_id",
        "foreground_id",
        "candidate_seed",
        "rendering_seed",
        "config_sha256",
        "source_manifest_sha256",
        "resolution_manifest_sha256",
        "legacy_source_sha256",
        "code_sha256",
        "selection",
    }
    missing = sorted(required_metadata - set(metadata))
    if missing:
        reasons.append("metadata_missing:" + ",".join(missing))
    return sorted(set(reasons))


def _bundle_paths(bundle_dir: Path) -> dict[str, Path]:
    return {
        "image": bundle_dir / "image.png",
        "mask": bundle_dir / "mask.png",
        "polygon": bundle_dir / "polygon.txt",
        "metadata": bundle_dir / "metadata.json",
        "marker": bundle_dir / "COMMITTED.json",
    }


def _validate_committed_bundle(
    bundle_dir: Path, expected_contract: dict[str, str]
) -> tuple[dict[str, Any], dict[str, str]]:
    paths = _bundle_paths(bundle_dir)
    if not paths["marker"].is_file():
        raise ManifestError(f"bundle has no commit marker: {bundle_dir}")
    marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
    for key, expected in expected_contract.items():
        if marker.get(key) != expected:
            raise ManifestError(f"committed bundle contract changed: {bundle_dir}:{key}")
    hashes: dict[str, str] = {}
    for name in ("image", "mask", "polygon", "metadata"):
        if not paths[name].is_file():
            raise ManifestError(f"committed bundle is missing {name}: {bundle_dir}")
        hashes[name] = sha256_file(paths[name])
        if marker.get("artifact_sha256", {}).get(name) != hashes[name]:
            raise ManifestError(f"committed bundle hash changed: {bundle_dir}:{name}")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    reasons = validate_rendered_bundle(
        image=paths["image"].read_bytes(),
        mask=paths["mask"].read_bytes(),
        polygon=paths["polygon"].read_bytes(),
        metadata=metadata,
    )
    if reasons:
        raise ManifestError(
            f"committed bundle validation failed: {bundle_dir}: {','.join(reasons)}"
        )
    return metadata, hashes


def _quarantine_partial(bundle_dir: Path, output_dir: Path) -> None:
    if not bundle_dir.exists():
        return
    failed_root = output_dir / "_failed" / bundle_dir.parent.name
    failed_root.mkdir(parents=True, exist_ok=True)
    destination = failed_root / f"{bundle_dir.name}.{uuid.uuid4().hex}"
    from capstone_lab.campaign.io import atomic_replace
    atomic_replace(bundle_dir, destination)


def _commit_rendered_bundle(
    *,
    output_dir: Path,
    plan: CandidatePlan,
    condition: str,
    rendered: dict[str, Any],
    metadata: dict[str, Any],
    contract: dict[str, str],
    inject_failure_stage: str | None = None,
) -> dict[str, str]:
    pending = output_dir / ".pending" / plan.sample_id / (
        condition + "." + uuid.uuid4().hex
    )
    pending.mkdir(parents=True, exist_ok=False)
    paths = _bundle_paths(pending)
    try:
        stage_content = (
            ("image", rendered["image"]),
            ("mask", rendered["mask"]),
            ("polygon", rendered["polygon"]),
            ("metadata", _json_bytes(metadata)),
        )
        artifact_hashes: dict[str, str] = {}
        for name, content in stage_content:
            _atomic_write(paths[name], content)
            artifact_hashes[name] = _sha256_bytes(content)
            if inject_failure_stage == f"after_{name}":
                raise InjectedInterruption(f"injected coordinator failure after {name}")
        reasons = validate_rendered_bundle(
            image=rendered["image"],
            mask=rendered["mask"],
            polygon=rendered["polygon"],
            metadata=metadata,
        )
        if reasons:
            raise ManifestError("rendered bundle validation failed: " + ",".join(reasons))
        marker = {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "sample_id": plan.sample_id,
            "condition_id": condition,
            **contract,
            "artifact_sha256": artifact_hashes,
        }
        _atomic_write(paths["marker"], _json_bytes(marker))
        if inject_failure_stage == "after_marker":
            raise InjectedInterruption("injected coordinator failure after marker")

        final_dir = output_dir / "samples" / plan.sample_id / condition
        if final_dir.exists():
            if (final_dir / "COMMITTED.json").is_file():
                existing_metadata, existing_hashes = _validate_committed_bundle(
                    final_dir, contract
                )
                if existing_metadata != metadata or existing_hashes != artifact_hashes:
                    raise ManifestError(f"committed filename collision: {final_dir}")
                shutil.rmtree(pending)
                return existing_hashes
            _quarantine_partial(final_dir, output_dir)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        from capstone_lab.campaign.io import atomic_replace
        atomic_replace(pending, final_dir)
        return artifact_hashes
    except Exception:
        # The pending directory intentionally remains as failure evidence and is never counted.
        raise


def _contract_hashes(
    *,
    project_root: Path,
    config: SmokeConfig,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    resolution_path: Path,
    legacy_path: Path,
    source_manifest: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    config_hash = _sha256_bytes(_json_bytes(config.deterministic_dict()))
    module_root = Path(__file__).resolve(strict=True).parent
    code_files = (
        module_root / "contract.py",
        module_root / "scoring.py",
        module_root / "smoke.py",
    )
    code_records = {
        f"capstone_lab/synthesis/{path.name}": sha256_file(path) for path in code_files
    }
    code_hash = _sha256_bytes(_json_bytes(code_records))
    input_records = []
    for row in source_manifest["backgrounds"]:
        input_records.append((row["path"], row["sha256"]))
    for row in source_manifest["foregrounds"]:
        input_records.extend(
            (
                (row["image_path"], row["image_sha256"]),
                (row["mask_path"], row["mask_sha256"]),
            )
        )
    input_hash = _sha256_bytes(_json_bytes(sorted(input_records)))
    contract = {
        "config_sha256": config_hash,
        "source_manifest_sha256": source_manifest_sha256,
        "resolution_manifest_sha256": sha256_file(resolution_path),
        "legacy_source_sha256": sha256_file(legacy_path),
        "code_sha256": code_hash,
        "input_sha256": input_hash,
    }
    detail = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        **contract,
        "config": config.deterministic_dict(),
        "source_manifest": source_manifest_path.relative_to(project_root).as_posix(),
        "resolution_manifest": resolution_path.relative_to(project_root).as_posix(),
        "legacy_source": legacy_path.relative_to(project_root).as_posix(),
        "code_files": code_records,
    }
    return contract, detail


def _write_or_validate_run_contract(output_dir: Path, detail: dict[str, Any]) -> None:
    path = output_dir / "run_contract.json"
    content = _json_bytes(detail)
    if path.exists() and path.read_bytes() != content:
        raise ManifestError(
            "existing S4 output cannot be reused after input/config/code/manifest hash change"
        )
    _atomic_write(path, content, overwrite=False)


def _output_hashes(
    output_dir: Path, plans: Iterable[CandidatePlan], contract: dict[str, str]
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, int]]:
    hashes: dict[str, str] = {}
    samples: list[dict[str, Any]] = []
    counts = {condition: 0 for condition in CONDITIONS}
    for plan in plans:
        selections: dict[str, Any] = {}
        for condition in CONDITIONS:
            bundle = output_dir / "samples" / plan.sample_id / condition
            if not (bundle / "COMMITTED.json").is_file():
                continue
            metadata, artifact_hashes = _validate_committed_bundle(bundle, contract)
            counts[condition] += 1
            selections[condition] = metadata["selection"]
            for name, digest in artifact_hashes.items():
                relative = (bundle / _bundle_paths(bundle)[name].name).relative_to(
                    output_dir
                )
                hashes[relative.as_posix()] = digest
            hashes[(bundle / "COMMITTED.json").relative_to(output_dir).as_posix()] = (
                sha256_file(bundle / "COMMITTED.json")
            )
        samples.append(
            {
                "sample_id": plan.sample_id,
                "plan": plan.to_dict(),
                "selections": selections,
            }
        )
    return dict(sorted(hashes.items())), samples, counts


def run_actual_smoke_campaign(
    *,
    project_root: Path,
    source_manifest_path: Path,
    resolution_manifest_path: Path,
    output_dir: Path,
    workers: int,
    config: SmokeConfig | None = None,
    enforce_smoke_minimum: bool = True,
    fail_feature_sample_ids: Iterable[str] = (),
    terminate_feature_sample_ids: Iterable[str] = (),
    fail_render_keys: Iterable[str] = (),
    corrupt_render_keys: Iterable[str] = (),
    interrupt_after_commits: int | None = None,
    inject_commit_failure: tuple[str, str] | None = None,
) -> dict[str, Any]:
    if workers not in {1, 2, 6}:
        raise ManifestError("S4 workers must be one of 1, 2, or 6")
    root = project_root.resolve(strict=True)
    legacy_path = (root / "build_demo_synthetic_segmentation.py").resolve(strict=True)
    source_path = source_manifest_path.resolve(strict=True)
    resolution_path = resolution_manifest_path.resolve(strict=True)
    output = validate_fixture_output(
        output_dir,
        allowed_root=root,
        protected_paths=(root / "Dataset", legacy_path, source_path, resolution_path),
    )
    smoke_config = config or SmokeConfig()
    smoke_config.validate(enforce_smoke_minimum=enforce_smoke_minimum)
    verify_legacy_contract(legacy_path)
    source_manifest, source_manifest_hash = _load_source_manifest(
        root, source_path, resolution_path
    )
    plans = build_actual_candidate_plans(smoke_config, source_manifest)
    contract, contract_detail = _contract_hashes(
        project_root=root,
        config=smoke_config,
        source_manifest_path=source_path,
        source_manifest_sha256=source_manifest_hash,
        resolution_path=resolution_path,
        legacy_path=legacy_path,
        source_manifest=source_manifest,
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_or_validate_run_contract(output, contract_detail)

    backgrounds = {row["background_id"]: row for row in source_manifest["backgrounds"]}
    foregrounds = {row["foreground_id"]: row for row in source_manifest["foregrounds"]}
    fail_features = set(fail_feature_sample_ids)
    terminate_features = set(terminate_feature_sample_ids)
    fail_renders = set(fail_render_keys)
    corrupt_renders = set(corrupt_render_keys)
    valid_sample_ids = {plan.sample_id for plan in plans}
    if not fail_features.issubset(valid_sample_ids):
        raise ManifestError("unknown injected feature failure sample")
    if not terminate_features.issubset(valid_sample_ids):
        raise ManifestError("unknown injected feature termination sample")

    states = {
        condition: UsageState(tuple(sorted(foregrounds)))
        for condition in CONDITIONS
    }
    already_committed: dict[tuple[str, str], dict[str, Any]] = {}
    missing_samples: set[str] = set()
    for plan in plans:
        for condition in CONDITIONS:
            bundle = output / "samples" / plan.sample_id / condition
            if (bundle / "COMMITTED.json").is_file():
                metadata, _ = _validate_committed_bundle(bundle, contract)
                if metadata.get("sample_id") != plan.sample_id or metadata.get(
                    "condition_id"
                ) != condition:
                    raise ManifestError(f"commit marker identity mismatch: {bundle}")
                already_committed[(plan.sample_id, condition)] = metadata
            else:
                missing_samples.add(plan.sample_id)

    feature_tasks: list[ActualFeatureTask] = []
    task_lookup: dict[tuple[str, str], ActualFeatureTask] = {}
    for plan in plans:
        if plan.sample_id not in missing_samples:
            continue
        for foreground_id in plan.foreground_candidate_ids:
            for position_id in plan.position_candidate_ids:
                task = ActualFeatureTask(
                    legacy_source_path=str(legacy_path),
                    project_root=str(root),
                    plan=plan.to_dict(),
                    background=backgrounds[plan.background_id],
                    foreground=foregrounds[foreground_id],
                    position_id=position_id,
                    target_long_side=smoke_config.target_long_side,
                    fail=plan.sample_id in fail_features,
                    terminate=plan.sample_id in terminate_features,
                )
                feature_tasks.append(task)
                task_lookup[(plan.sample_id, f"{foreground_id}@{position_id}")] = task

    features_by_sample: dict[str, list[CandidateFeature]] = {
        plan.sample_id: [] for plan in plans
    }
    feature_errors: dict[str, list[str]] = {plan.sample_id: [] for plan in plans}
    failures: list[dict[str, Any]] = []
    committed_this_run = 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=_worker_init
    ) as pool:
        feature_futures = {
            pool.submit(compute_actual_features, task): task for task in feature_tasks
        }
        for future in as_completed(feature_futures):
            task = feature_futures[future]
            try:
                features_by_sample[task.plan["sample_id"]].append(future.result())
            except Exception as exc:
                feature_errors[task.plan["sample_id"]].append(
                    f"{type(exc).__name__}: {exc}"
                )

        for plan in plans:
            for condition in CONDITIONS:
                existing = already_committed.get((plan.sample_id, condition))
                if existing is not None:
                    states[condition].commit(
                        existing["foreground_id"], plan.domain, plan.background_id
                    )
                    continue
                if feature_errors[plan.sample_id]:
                    failures.append(
                        {
                            "sample_id": plan.sample_id,
                            "condition_id": condition,
                            "stage": "feature",
                            "reasons": sorted(set(feature_errors[plan.sample_id])),
                        }
                    )
                    continue
                choice = select_candidate(
                    condition,
                    plan,
                    features_by_sample[plan.sample_id],
                    states[condition],
                    smoke_config.campaign_seed,
                )
                if choice is None:
                    failures.append(
                        {
                            "sample_id": plan.sample_id,
                            "condition_id": condition,
                            "stage": "selection",
                            "reasons": ["no_eligible_candidate"],
                        }
                    )
                    continue
                feature, selection = choice
                key = f"{plan.sample_id}/{condition}"
                render_task = ActualRenderTask(
                    feature_task=task_lookup[(plan.sample_id, feature.candidate_id)],
                    condition_id=condition,
                    selection_metadata=selection,
                    fail=key in fail_renders,
                    corrupt_mask=key in corrupt_renders,
                )
                try:
                    rendered = pool.submit(render_actual_bundle, render_task).result()
                    metadata = {
                        "schema_version": SMOKE_SCHEMA_VERSION,
                        "sample_id": plan.sample_id,
                        "condition_id": condition,
                        "background_id": plan.background_id,
                        "background_path": backgrounds[plan.background_id]["path"],
                        "background_sha256": backgrounds[plan.background_id]["sha256"],
                        "foreground_id": feature.foreground_id,
                        "foreground_environment": foregrounds[feature.foreground_id][
                            "environment"
                        ],
                        "foreground_image_path": foregrounds[feature.foreground_id][
                            "image_path"
                        ],
                        "foreground_image_sha256": foregrounds[feature.foreground_id][
                            "image_sha256"
                        ],
                        "foreground_mask_path": foregrounds[feature.foreground_id][
                            "mask_path"
                        ],
                        "foreground_mask_sha256": foregrounds[feature.foreground_id][
                            "mask_sha256"
                        ],
                        "candidate_seed": plan.candidate_seed,
                        "rendering_seed": plan.rendering_seed,
                        "config_sha256": contract["config_sha256"],
                        "source_manifest_sha256": contract["source_manifest_sha256"],
                        "resolution_manifest_sha256": contract[
                            "resolution_manifest_sha256"
                        ],
                        "legacy_source_sha256": contract["legacy_source_sha256"],
                        "code_sha256": contract["code_sha256"],
                        "input_sha256": contract["input_sha256"],
                        "mode": plan.mode,
                        "domain": plan.domain,
                        "difficulty": plan.difficulty,
                        "scale_bucket": plan.scale_bucket,
                        "scale_ratio": plan.scale_ratio,
                        "position_id": feature.position_id,
                        "selection": selection,
                        "legacy_total_score_ignored": True,
                        "final_ranking_implementation": "capstone_lab.synthesis.scoring.select_candidate",
                        "tone_matching_preserved": True,
                        "alpha_blending_preserved": True,
                        "boundary_renderer_preserved": True,
                        "render": {
                            key_name: value
                            for key_name, value in rendered.items()
                            if key_name not in {"image", "mask", "polygon"}
                        },
                    }
                    failure_stage = None
                    if inject_commit_failure is not None and inject_commit_failure[0] == key:
                        failure_stage = inject_commit_failure[1]
                    _commit_rendered_bundle(
                        output_dir=output,
                        plan=plan,
                        condition=condition,
                        rendered=rendered,
                        metadata=metadata,
                        contract=contract,
                        inject_failure_stage=failure_stage,
                    )
                    states[condition].commit(
                        feature.foreground_id, plan.domain, plan.background_id
                    )
                    committed_this_run += 1
                    if (
                        interrupt_after_commits is not None
                        and committed_this_run >= interrupt_after_commits
                    ):
                        raise InjectedInterruption(
                            f"injected coordinator termination after {committed_this_run} commits"
                        )
                except InjectedInterruption:
                    raise
                except Exception as exc:
                    failures.append(
                        {
                            "sample_id": plan.sample_id,
                            "condition_id": condition,
                            "stage": "render_or_commit",
                            "reasons": [f"{type(exc).__name__}: {exc}"],
                        }
                    )

    hashes, samples, counts = _output_hashes(output, plans, contract)
    deterministic_payload = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "contract": contract,
        "plans": [plan.to_dict() for plan in plans],
        "samples": samples,
        "output_hashes": hashes,
        "condition_counts": counts,
        "legacy_feature_functions": list(LEGACY_FEATURE_FUNCTIONS),
        "legacy_renderer_functions": list(LEGACY_RENDER_FUNCTIONS),
        "final_ranking_implementation": "capstone_lab.synthesis.scoring.select_candidate",
        "legacy_duplicate_final_score_bypassed": True,
    }
    verified = not failures and all(
        count == smoke_config.sample_count for count in counts.values()
    )
    summary = {
        "status": "VERIFIED" if verified else "PARTIAL_FAILURE",
        "workers": workers,
        "single_shared_process_pool": True,
        "sample_count_per_condition": smoke_config.sample_count,
        "condition_counts": counts,
        "committed_this_run": committed_this_run,
        "skipped_existing": len(already_committed),
        "failures": failures,
        "deterministic_sha256": _sha256_bytes(_json_bytes(deterministic_payload)),
        "deterministic_payload": deterministic_payload,
    }
    _atomic_write(output / "summary.json", _json_bytes(summary))
    return summary
