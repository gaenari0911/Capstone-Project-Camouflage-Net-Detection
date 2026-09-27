from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import sha256_file


def _find_project_root(start: Path) -> Path | None:
    candidate = start if start.is_dir() else start.parent
    for root in (candidate, *candidate.parents):
        if (root / "docs" / "recovery_audit").is_dir():
            return root
    return None


def inspect_recovery(start: Path) -> dict[str, Any]:
    """Summarize existing audit evidence without touching source datasets."""
    project_root = _find_project_root(start)
    if project_root is None:
        return {"available": False, "reason": "docs/recovery_audit not found"}
    audit_root = project_root / "docs" / "recovery_audit"
    split_path = audit_root / "current_split_inventory.csv"
    verification_path = audit_root / "verification.json"
    missing = [str(path) for path in (split_path, verification_path) if not path.is_file()]
    if missing:
        return {"available": False, "reason": "required audit files missing", "missing": missing}

    split_counts: Counter[str] = Counter()
    hashes: dict[str, set[str]] = defaultdict(set)
    with split_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "image", "image_sha256", "label", "label_sha256"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            return {"available": False, "reason": "split inventory schema mismatch"}
        for row in reader:
            split = row["split"].strip()
            split_counts[split] += 1
            hashes[row["image_sha256"].strip()].add(split)
    cross_split_hashes = sum(1 for splits in hashes.values() if len(splits) > 1)

    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    metadata_mismatches = len(verification.get("object_metadata_mismatch", []))
    synthetic = verification.get("synthetic", {})
    orphan_labels = len(synthetic.get("orphan_labels", []))
    issues = [
        {
            "code": "OBJECT_METADATA_MISMATCH",
            "count": metadata_mismatches,
            "action": "review via a separate override manifest; do not mutate the source CSV",
        },
        {
            "code": "HISTORICAL_SYNTHETIC_ORPHAN_LABEL",
            "count": orphan_labels,
            "action": "keep historical files separate; do not merge into new datasets automatically",
        },
    ]
    issues = [issue for issue in issues if issue["count"]]
    expected = {"train": 748, "val": 100, "test": 200}
    return {
        "available": True,
        "status": "ATTENTION" if issues else "PASS",
        "project_root": str(project_root),
        "split_inventory_sha256": sha256_file(split_path),
        "verification_sha256": sha256_file(verification_path),
        "split_counts": dict(sorted(split_counts.items())),
        "expected_split_counts": expected,
        "split_counts_match_expected": dict(split_counts) == expected,
        "cross_split_exact_image_hashes": cross_split_hashes,
        "outer_crc_test": verification.get("outer_crc_test"),
        "extracted_size_mismatch_count": len(verification.get("extracted_size_mismatches", [])),
        "real_archive_crc_mismatch_count": len(
            verification.get("real_archive_vs_extracted_crc_mismatches", [])
        ),
        "historical_synthetic_counts": {
            "images": synthetic.get("images"),
            "masks": synthetic.get("masks"),
            "labels": synthetic.get("labels"),
        },
        "known_issues": issues,
        "limitations": [
            "manifest hashes are prior observations, not a fresh hash of every source file",
            "zero exact cross-split hashes does not prove absence of near-duplicate scene leakage",
            "background ZIP decode/count adjudication is not represented in these two audit files",
        ],
    }
