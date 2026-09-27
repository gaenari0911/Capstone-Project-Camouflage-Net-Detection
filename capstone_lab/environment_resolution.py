from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import canonical_json, sha256_file
from .errors import ManifestError

ALLOWED_ENVIRONMENTS = ("non_snow", "snow")
DECISION_RULE = "disk_directory_is_authoritative"
OUTPUT_FILENAME = "object_environment_resolution_v1.jsonl"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _write_immutable(path: Path, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ManifestError(f"refusing to overwrite different resolution manifest: {path}")
        return digest
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
    return digest


def resolve_object_environments_from_disk(
    *,
    project_root: Path,
    review_path: Path,
    object_pool_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    review = review_path.resolve(strict=True)
    pool = object_pool_root.resolve(strict=True)
    output = output_path.resolve(strict=False)
    if not _inside(review, root) or not _inside(pool, root) or not _inside(output, root):
        raise ManifestError("review, object pool, and output must stay inside project root")
    if output == review or _inside(output, pool) or _inside(pool, output):
        raise ManifestError("resolution output overlaps a protected source")

    review_hash = sha256_file(review)
    input_rows = [
        json.loads(line)
        for line in review.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(input_rows) != 52:
        raise ManifestError(f"expected 52 review rows, got {len(input_rows)}")

    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in input_rows:
        object_id = row.get("object_source_id")
        disk_values = row.get("disk_derived_environment")
        if not isinstance(object_id, str) or not object_id or object_id in seen:
            raise ManifestError("review manifest has an invalid or duplicate object_source_id")
        seen.add(object_id)
        if (
            not isinstance(disk_values, list)
            or len(disk_values) != 1
            or disk_values[0] not in ALLOWED_ENVIRONMENTS
        ):
            raise ManifestError(f"{object_id}: disk environment must be exactly one known value")
        chosen = disk_values[0]
        image = pool / chosen / "images" / f"{object_id}.jpg"
        mask = pool / chosen / "masks" / f"{object_id}.png"
        if not image.is_file() or not mask.is_file():
            raise ManifestError(f"{object_id}: disk image/mask pair is missing")
        other = "snow" if chosen == "non_snow" else "non_snow"
        other_image = pool / other / "images" / f"{object_id}.jpg"
        other_mask = pool / other / "masks" / f"{object_id}.png"
        if other_image.exists() or other_mask.exists():
            raise ManifestError(f"{object_id}: object also exists in the opposite environment")
        resolved.append(
            {
                "object_source_id": object_id,
                "status": "RESOLVED",
                "decision_rule": DECISION_RULE,
                "chosen_environment": chosen,
                "csv_environment": row.get("csv_environment"),
                "disk_derived_environment": disk_values,
                "image": image.relative_to(root).as_posix(),
                "image_sha256": sha256_file(image),
                "mask": mask.relative_to(root).as_posix(),
                "mask_sha256": sha256_file(mask),
                "source_review_sha256": review_hash,
            }
        )
    resolved.sort(key=lambda item: item["object_source_id"])
    content = "".join(canonical_json(row) + "\n" for row in resolved).encode("utf-8")
    output_hash = _write_immutable(output, content)
    counts = {
        environment: sum(
            row["chosen_environment"] == environment for row in resolved
        )
        for environment in ALLOWED_ENVIRONMENTS
    }
    return {
        "status": "VERIFIED",
        "decision_rule": DECISION_RULE,
        "rows": len(resolved),
        "counts": counts,
        "source_review_sha256": review_hash,
        "output": str(output),
        "output_sha256": output_hash,
    }
