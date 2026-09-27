from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from capstone_lab.config import sha256_file
from capstone_lab.errors import ArtifactError
from capstone_lab.manifests import tree_fingerprint
from .contracts import inside
from .io import atomic_json

DATASET_SHA256 = "8ff188ae4788961b00221e0452a5e38da0aa345d2ea1ffed740cfa001902b197"


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_sources(root, output, *, foreground_subset='low187'):
    cv2.setNumThreads(1)
    before = tree_fingerprint(root / "Dataset", workers=1)
    if before["sha256"] != DATASET_SHA256:
        raise ArtifactError("protected Dataset differs from S7")
    observed = jsonl(root / "manifests/real_split_observed_v1.jsonl")
    low = jsonl(root / "manifests/low187_v1.jsonl")
    train = {r["image"]: r for r in observed if r["original_split"] == "train"}
    low_ids = {r["image"] for r in low}
    if len(train) != 748 or len(low_ids) != 187 or not low_ids <= train.keys():
        raise ArtifactError("low187 membership/count changed")
    if foreground_subset not in {'low187', 'train748'}:
        raise ArtifactError('unknown foreground subset')
    approved_ids = low_ids
    if foreground_subset == 'train748':
        from .contracts import read_json
        approval = read_json(root / 'configs/approvals/s8_foreground_amendment_v1.json')
        if approval['status'] != 'APPROVED_BY_USER' or approval['foreground_subsets'] != {
                'A': 'train748', 'M': 'train748', 'H': 'train748', 'L': 'low187'}:
            raise ArtifactError('split foreground policy not approved')
        approved_ids = set(train)
    # Read-only observed Test integrity is not inference/tuning or loading Test pixels.
    for record in observed:
        for key in ("image", "label"):
            if sha256_file(inside(root, record[key])) != record[key + "_sha256"]:
                raise ArtifactError(f"observed source changed: {record[key]}")
    source = root / "Build_Object_Pool_From_Txt_ObjectBased.py"
    spec = importlib.util.spec_from_file_location("s8_readonly_object_reference", source)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)  # main() is guarded; call only pure crop/polygon helpers.
    metadata_path = root / "object_pool_metadata.csv"
    metadata = list(csv.DictReader(metadata_path.open(encoding="utf-8-sig", newline="")))
    by_name = {row["object_name"]: row for row in metadata}
    if len(by_name) != len(metadata):
        raise ArtifactError("duplicate object metadata")
    evidence, rejected = [], []
    for environment in ("non_snow", "snow"):
        image_dir = root / "Dataset/object_pool" / environment / "images"
        for object_image in sorted(image_dir.glob("*.jpg")):
            row = by_name.get(object_image.stem)
            if row is None:
                rejected.append({"object": object_image.name, "reason": "missing explicit metadata"})
                continue
            source_relative = "Dataset/images/train/" + row["source_image"]
            if source_relative not in train:
                rejected.append({"object": object_image.name, "reason": "metadata source not in train"})
                continue
            if source_relative not in approved_ids:
                continue
            record = train[source_relative]
            image = cv2.imread(str(root / source_relative))
            polygon = legacy.load_polygon_from_txt(root / record["label"], image.shape[1], image.shape[0])
            mask = legacy.polygon_to_mask(polygon, image.shape[1], image.shape[0])
            crop = legacy.extract_object(image, mask)
            object_mask = object_image.parent.parent / "masks" / (object_image.stem + ".png")
            stored_mask = cv2.imread(str(object_mask), cv2.IMREAD_GRAYSCALE)
            ok, encoded = cv2.imencode(".jpg", crop["crop_image"])
            image_matches = ok and hashlib.sha256(encoded.tobytes()).hexdigest() == sha256_file(object_image)
            bbox_matches = all(int(row[k]) == crop[k] for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
            mask_matches = stored_mask is not None and np.array_equal(stored_mask, crop["crop_mask"])
            if not (image_matches and mask_matches and bbox_matches):
                rejected.append({"object": object_image.name, "source": source_relative,
                                 "image_exact": bool(image_matches), "mask_exact": bool(mask_matches),
                                 "bbox_exact": bbox_matches, "reason": "crop reconstruction mismatch"})
                continue
            evidence.append({"source_image": source_relative, "source_sha256": record["image_sha256"],
                             "label": record["label"], "label_sha256": record["label_sha256"],
                             "object_image": object_image.relative_to(root).as_posix(),
                             "object_image_sha256": sha256_file(object_image),
                             "object_mask": object_mask.relative_to(root).as_posix(),
                             "object_mask_sha256": sha256_file(object_mask),
                             "environment": environment, "proof": "metadata_and_exact_reconstructed_jpeg_mask_bbox"})
    after = tree_fingerprint(root / "Dataset", workers=1)
    if before != after:
        raise ArtifactError("Dataset changed during source audit")
    unique_sources = {r["source_image"] for r in evidence}
    status = "VERIFIED" if unique_sources == approved_ids and len(evidence) == len(approved_ids) and not rejected else "UNRESOLVED"
    filename = foreground_subset + '_object_provenance.json'
    atomic_json(output / filename,
                {"status": status, "records": evidence, "rejected": rejected,
                 "object_builder_sha256": sha256_file(source),
                 "metadata_sha256": sha256_file(metadata_path),
                 "low187_sha256": sha256_file(root / "manifests/low187_v1.jsonl")}, immutable=True)
    return {"audit_completed": True, "source_membership_status": status,
            "verified_objects": len(evidence), "verified_sources": len(unique_sources),
            "rejected_count": len(rejected), "dataset_before": before, "dataset_after": after,
            "raw_imageN_mapping": "NOT_NEEDED_FOR_CURRENT_TRAIN_CROP_PROOF_NOT_CLAIMED",
            "output_files": [filename]}


def audit_split_sources(root, output):
    from .contracts import read_json
    output = output.resolve()
    if not output.is_relative_to((root / 'artifacts/s8_source_audit').resolve()) or output == (root / 'artifacts/s8_source_audit').resolve():
        raise ArtifactError('new source audit must be in its own s8_source_audit run')
    approval_path = root / 'configs/approvals/s8_foreground_amendment_v1.json'
    approval = read_json(approval_path)
    if approval['foreground_subsets'] != {'A': 'train748', 'M': 'train748', 'H': 'train748', 'L': 'low187'}:
        raise ArtifactError('unexpected source budget mapping')
    reports = {subset: audit_sources(root, output, foreground_subset=subset)
               for subset in ('train748', 'low187')}
    if any(r['source_membership_status'] != 'VERIFIED' for r in reports.values()):
        raise ArtifactError('source provenance verification failed')
    from collections import Counter
    pools = {}
    for subset in reports:
        path = output / (subset + '_object_provenance.json')
        records = read_json(path)['records']
        pools[subset] = {'path': path.relative_to(root).as_posix(), 'sha256': sha256_file(path),
                         'objects': len(records), 'environments': dict(Counter(r['environment'] for r in records))}
    full = {r['object_image']: r for r in read_json(output / 'train748_object_provenance.json')['records']}
    low = read_json(output / 'low187_object_provenance.json')['records']
    if any(full.get(r['object_image']) != r for r in low):
        raise ArtifactError('low pool not exact subset of verified full pool')
    summary = {'status': 'VERIFIED', 'approval_sha256': sha256_file(approval_path),
               'source_policy': approval['foreground_subsets'], 'pools': pools,
               'low_is_exact_subset': True, 'dataset_sha256': DATASET_SHA256,
               'generation_started': False, 'training_started': False,
               'reports': reports}
    atomic_json(output / 'split_source_summary.json', summary, immutable=True)
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_split_sources(Path.cwd().resolve(), args.output), indent=2))
