from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .config import canonical_json, sha256_file
from .errors import ManifestError

ALGORITHM_ID = "low187_sha256_rank_v1"
DEFAULT_SEED = 42
SUBSET_ID = "low187_v1"
EXPECTED_SPLITS = {"train": 748, "val": 100, "test": 200}
CONTENT_FILENAMES = {
    "observed": "real_split_observed_v1.jsonl",
    "subset": "low187_v1.jsonl",
    "review": "object_environment_review_v1.jsonl",
}


@dataclass(frozen=True)
class InventoryRow:
    split: str
    image: str
    image_sha256: str
    label: str
    label_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_relative_path(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ManifestError("inventory path must be a non-empty string")
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ManifestError(f"unsafe project-relative path: {raw}")
    if ":" in path.parts[0]:
        raise ManifestError(f"drive-qualified path is not allowed: {raw}")
    return path.as_posix()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_source(project_root: Path, relative: str) -> Path:
    root = project_root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relative).parts).resolve(strict=False)
    if not _inside(candidate, root):
        raise ManifestError(f"source path escapes project root: {relative}")
    return candidate


def validate_output_dir(project_root: Path, output_dir: Path, protected: Iterable[Path]) -> Path:
    root = project_root.resolve(strict=True)
    output = output_dir.resolve(strict=False)
    if output == root or not _inside(output, root):
        raise ManifestError("manifest output must be a non-root directory inside the project")
    for source in protected:
        source = source.resolve(strict=False)
        if _inside(output, source) or _inside(source, output):
            raise ManifestError(f"manifest output overlaps protected source: {source}")
    return output


def load_inventory(path: Path) -> tuple[list[InventoryRow], str]:
    source = path.resolve(strict=True)
    digest = sha256_file(source)
    rows: list[InventoryRow] = []
    seen_images: set[str] = set()
    seen_labels: set[str] = set()
    valid_hash = set("0123456789abcdef")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "image", "image_sha256", "label", "label_sha256"}
        if not reader.fieldnames or set(reader.fieldnames) != required:
            raise ManifestError(
                f"inventory columns must be exactly {sorted(required)}, got {reader.fieldnames}"
            )
        for line_number, raw in enumerate(reader, start=2):
            split = raw["split"].strip()
            if split not in EXPECTED_SPLITS:
                raise ManifestError(f"line {line_number}: unsupported split {split}")
            image = normalize_relative_path(raw["image"])
            label = normalize_relative_path(raw["label"])
            image_hash = raw["image_sha256"].strip().lower()
            label_hash = raw["label_sha256"].strip().lower()
            if (
                len(image_hash) != 64
                or len(label_hash) != 64
                or set(image_hash) - valid_hash
                or set(label_hash) - valid_hash
            ):
                raise ManifestError(f"line {line_number}: invalid SHA-256")
            if image in seen_images or label in seen_labels:
                raise ManifestError(f"line {line_number}: duplicate image or label path")
            seen_images.add(image)
            seen_labels.add(label)
            rows.append(InventoryRow(split, image, image_hash, label, label_hash))
    counts = Counter(row.split for row in rows)
    if dict(counts) != EXPECTED_SPLITS:
        raise ManifestError(f"split counts differ from required 748/100/200: {dict(counts)}")
    return rows, digest


def _verify_row(project_root: Path, row: InventoryRow) -> list[dict[str, str]]:
    mismatches: list[dict[str, str]] = []
    for kind, relative, expected in (
        ("image", row.image, row.image_sha256),
        ("label", row.label, row.label_sha256),
    ):
        path = resolve_source(project_root, relative)
        if not path.is_file():
            mismatches.append(
                {"kind": kind, "path": relative, "expected": expected, "actual": "MISSING"}
            )
            continue
        actual = sha256_file(path)
        if actual != expected:
            mismatches.append(
                {"kind": kind, "path": relative, "expected": expected, "actual": actual}
            )
    return mismatches


def verify_inventory_files(
    project_root: Path, rows: list[InventoryRow], *, workers: int
) -> dict[str, Any]:
    if workers < 1 or workers > 16:
        raise ManifestError("workers must be between 1 and 16")
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="manifest-hash") as pool:
        results = list(pool.map(lambda row: _verify_row(project_root, row), rows))
    mismatches = [item for group in results for item in group]
    if mismatches:
        preview = mismatches[:10]
        raise ManifestError(
            f"{len(mismatches)} source files differ from inventory; first entries: {preview}"
        )

    split_paths: dict[str, set[str]] = defaultdict(set)
    split_hashes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_paths[row.split].add(row.image)
        split_hashes[row.split].add(row.image_sha256)
    path_intersections: dict[str, int] = {}
    hash_intersections: dict[str, int] = {}
    splits = ("train", "val", "test")
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            key = f"{left}:{right}"
            path_intersections[key] = len(split_paths[left] & split_paths[right])
            hash_intersections[key] = len(split_hashes[left] & split_hashes[right])
    if any(path_intersections.values()) or any(hash_intersections.values()):
        raise ManifestError(
            f"exact split leakage detected: paths={path_intersections}, hashes={hash_intersections}"
        )
    return {
        "verified_rows": len(rows),
        "verified_files": len(rows) * 2,
        "path_intersections": path_intersections,
        "image_hash_intersections": hash_intersections,
    }


def _fingerprint_one(root: Path, path: Path) -> tuple[str, int, str]:
    relative = path.relative_to(root).as_posix()
    return relative, path.stat().st_size, sha256_file(path)


def tree_fingerprint(root: Path, *, workers: int) -> dict[str, Any]:
    source = root.resolve(strict=True)
    if not source.is_dir():
        raise ManifestError(f"dataset root is not a directory: {source}")
    files = sorted((path for path in source.rglob("*") if path.is_file()), key=lambda p: p.relative_to(source).as_posix())
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tree-hash") as pool:
        records = list(pool.map(lambda path: _fingerprint_one(source, path), files))
    digest = hashlib.sha256()
    total_size = 0
    for relative, size, file_hash in records:
        total_size += size
        digest.update(f"{relative}\t{size}\t{file_hash}\n".encode("utf-8"))
    return {
        "algorithm": "tree_sha256_v1",
        "root": str(source),
        "file_count": len(records),
        "total_size": total_size,
        "sha256": digest.hexdigest(),
    }


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _write_immutable(path: Path, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ManifestError(f"refusing to overwrite different existing manifest: {path}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
    return digest


def _rank(relative_image: str, seed: int) -> str:
    value = f"{ALGORITHM_ID}|{seed}|{relative_image}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _content_rows(
    rows: list[InventoryRow], inventory_hash: str, source_hash: str, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    order = {"train": 0, "val": 1, "test": 2}
    observed = [
        {
            "record_id": row.image,
            "original_split": row.split,
            "image": row.image,
            "image_sha256": row.image_sha256,
            "label": row.label,
            "label_sha256": row.label_sha256,
            "inventory_sha256": inventory_hash,
            "generator_source_sha256": source_hash,
            "raw_source_mapping_status": "UNRESOLVED",
        }
        for row in sorted(rows, key=lambda item: (order[item.split], item.image))
    ]
    ranked = sorted(
        ((_rank(row.image, seed), row.image, row) for row in rows if row.split == "train"),
        key=lambda item: (item[0], item[1]),
    )
    subset = [
        {
            "subset_id": SUBSET_ID,
            "selection_algorithm": ALGORITHM_ID,
            "selection_seed": seed,
            "rank": index,
            "rank_digest": digest,
            "record_id": row.image,
            "original_split": row.split,
            "image": row.image,
            "image_sha256": row.image_sha256,
            "label": row.label,
            "label_sha256": row.label_sha256,
            "inventory_sha256": inventory_hash,
            "generator_source_sha256": source_hash,
            "raw_source_mapping_status": "UNRESOLVED",
        }
        for index, (digest, _, row) in enumerate(ranked[:187], start=1)
    ]
    return observed, subset


def _review_rows(verification: dict[str, Any], verification_hash: str) -> list[dict[str, Any]]:
    mismatches = verification.get("object_metadata_mismatch")
    if not isinstance(mismatches, list) or len(mismatches) != 52:
        raise ManifestError(
            f"expected 52 object metadata mismatches, got {len(mismatches) if isinstance(mismatches, list) else 'invalid'}"
        )
    rows: list[dict[str, Any]] = []
    for item in mismatches:
        object_id = item.get("object")
        csv_environment = item.get("metadata")
        actual = item.get("actual")
        if not isinstance(object_id, str) or not isinstance(csv_environment, str) or not isinstance(actual, list):
            raise ManifestError("object metadata mismatch schema is invalid")
        rows.append(
            {
                "object_source_id": object_id,
                "disk_derived_environment": sorted(str(value) for value in actual),
                "csv_environment": csv_environment,
                "status": "NEEDS_REVIEW",
                "chosen_environment": "UNDECIDED",
                "evidence_source": "docs/recovery_audit/verification.json#object_metadata_mismatch",
                "verification_sha256": verification_hash,
            }
        )
    rows.sort(key=lambda item: item["object_source_id"])
    if len({row["object_source_id"] for row in rows}) != 52:
        raise ManifestError("object metadata mismatch IDs are not unique")
    return rows


def _validate_subset(subset: list[dict[str, Any]], rows: list[InventoryRow]) -> dict[str, Any]:
    selected_ids = {row["record_id"] for row in subset}
    selected_hashes = {row["image_sha256"] for row in subset}
    val_test = [row for row in rows if row.split in {"val", "test"}]
    val_test_ids = {row.image for row in val_test}
    val_test_hashes = {row.image_sha256 for row in val_test}
    checks = {
        "selected_count": len(subset),
        "unique_id_count": len(selected_ids),
        "all_train": all(row["original_split"] == "train" for row in subset),
        "val_test_id_intersection": len(selected_ids & val_test_ids),
        "val_test_image_hash_intersection": len(selected_hashes & val_test_hashes),
    }
    if checks != {
        "selected_count": 187,
        "unique_id_count": 187,
        "all_train": True,
        "val_test_id_intersection": 0,
        "val_test_image_hash_intersection": 0,
    }:
        raise ManifestError(f"low187 validation failed: {checks}")
    return checks


def build_s2_manifests(
    *,
    project_root: Path,
    inventory_path: Path,
    verification_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    workers: int = 4,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    if workers < 1 or workers > 16:
        raise ManifestError("workers must be between 1 and 16")
    inventory = inventory_path.resolve(strict=True)
    verification_file = verification_path.resolve(strict=True)
    dataset_root = (root / "Dataset").resolve(strict=True)
    output = validate_output_dir(root, output_dir, (dataset_root, inventory, verification_file))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ManifestError("selection seed must be an integer")

    inventory_before = sha256_file(inventory)
    dataset_before = tree_fingerprint(dataset_root, workers=workers)
    rows, inventory_hash = load_inventory(inventory)
    if inventory_hash != inventory_before:
        raise ManifestError("inventory changed while it was being read")
    file_verification = verify_inventory_files(root, rows, workers=workers)
    verification_hash = sha256_file(verification_file)
    verification = json.loads(verification_file.read_text(encoding="utf-8"))
    source_hash = sha256_file(Path(__file__).resolve())
    observed, subset = _content_rows(rows, inventory_hash, source_hash, seed)
    subset_checks = _validate_subset(subset, rows)
    review = _review_rows(verification, verification_hash)
    orphan_labels = verification.get("synthetic", {}).get("orphan_labels", [])
    if not isinstance(orphan_labels, list) or len(orphan_labels) != 187:
        raise ManifestError("historical synthetic orphan label count is not 187")

    inventory_after = sha256_file(inventory)
    dataset_after = tree_fingerprint(dataset_root, workers=workers)
    if inventory_before != inventory_after:
        raise ManifestError("inventory changed during manifest generation")
    if dataset_before != dataset_after:
        raise ManifestError("Dataset tree changed during read-only manifest generation")

    output.mkdir(parents=True, exist_ok=True)
    content = {
        "observed": _jsonl_bytes(observed),
        "subset": _jsonl_bytes(subset),
        "review": _jsonl_bytes(review),
    }
    paths = {key: output / filename for key, filename in CONTENT_FILENAMES.items()}
    hashes = {key: _write_immutable(paths[key], content[key]) for key in content}
    meta = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "status": "OBSERVED_NOT_FREEZE_APPROVAL",
        "selection": {
            "subset_id": SUBSET_ID,
            "algorithm": ALGORITHM_ID,
            "seed": seed,
            "rank_expression": f'SHA256("{ALGORITHM_ID}|{seed}|" + normalized_relative_image_path)',
            "count": 187,
        },
        "inputs": {
            "inventory": str(inventory),
            "inventory_sha256_before": inventory_before,
            "inventory_sha256_after": inventory_after,
            "verification": str(verification_file),
            "verification_sha256": verification_hash,
            "generator_source": str(Path(__file__).resolve()),
            "generator_source_sha256": source_hash,
        },
        "dataset_before": dataset_before,
        "dataset_after": dataset_after,
        "file_verification": file_verification,
        "subset_checks": subset_checks,
        "outputs": {
            key: {"path": str(paths[key]), "sha256": hashes[key], "rows": len(value)}
            for key, value in (("observed", observed), ("subset", subset), ("review", review))
        },
        "known_unresolved": {
            "raw_source_mapping": "UNRESOLVED",
            "object_environment_review_count": len(review),
            "historical_synthetic_orphan_label_count": len(orphan_labels),
            "shared_low187_for_main": "UNDECIDED",
        },
        "limitations": [
            "this is an observed split manifest and candidate low-data subset, not freeze approval",
            "zero exact hash overlap does not prove absence of near-duplicate scene leakage",
            "historical orphan labels are reported only and are not part of the low-data subset",
        ],
    }
    meta_path = output / "low187_v1.meta.json"
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        stable_keys = ("selection", "inputs", "dataset_before", "dataset_after", "outputs")
        if any(existing.get(key) != meta.get(key) for key in stable_keys):
            raise ManifestError(f"refusing to replace incompatible metadata: {meta_path}")
        meta = existing
    else:
        _write_immutable(meta_path, (canonical_json(meta) + "\n").encode("utf-8"))
    return {
        "status": "VERIFIED",
        "output_dir": str(output),
        "content_hashes": hashes,
        "meta_path": str(meta_path),
        "inventory_sha256": inventory_hash,
        "dataset_tree_sha256": dataset_before["sha256"],
        "dataset_file_count": dataset_before["file_count"],
        "dataset_total_size": dataset_before["total_size"],
        "verified_source_files": file_verification["verified_files"],
        "subset_checks": subset_checks,
        "review_count": len(review),
        "historical_orphan_label_count": len(orphan_labels),
    }
