from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from capstone_lab.config import sha256_file
from capstone_lab.errors import ManifestError
from capstone_lab.manifests import (
    ALGORITHM_ID,
    build_s2_manifests,
    normalize_relative_path,
    tree_fingerprint,
)


class S2Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "Dataset"
        self.audit = self.root / "docs" / "recovery_audit"
        self.audit.mkdir(parents=True)
        rows: list[str] = ["split,image,image_sha256,label,label_sha256"]
        split_counts = (("train", 748), ("val", 100), ("test", 200))
        for split, count in split_counts:
            image_dir = self.dataset / "images" / split
            label_dir = self.dataset / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for index in range(1, count + 1):
                image = image_dir / f"{split}{index}.jpg"
                label = label_dir / f"{split}{index}.txt"
                image.write_bytes(f"image:{split}:{index}".encode())
                label.write_text(f"0 0.1 0.1 {index}\n", encoding="utf-8")
                rows.append(
                    ",".join(
                        (
                            split,
                            image.relative_to(self.root).as_posix(),
                            sha256_file(image),
                            label.relative_to(self.root).as_posix(),
                            sha256_file(label),
                        )
                    )
                )
        self.inventory = self.audit / "current_split_inventory.csv"
        self.inventory.write_text("\n".join(rows) + "\n", encoding="utf-8")
        verification = {
            "outer_crc_test": "PASS",
            "extracted_size_mismatches": [],
            "real_archive_vs_extracted_crc_mismatches": [],
            "object_metadata_mismatch": [
                {
                    "object": f"object_{index:03d}",
                    "metadata": "non_snow",
                    "actual": ["snow"],
                }
                for index in range(52)
            ],
            "synthetic": {
                "images": 2934,
                "masks": 2934,
                "labels": 3121,
                "orphan_labels": [f"orphan_{index:03d}" for index in range(187)],
            },
        }
        self.verification = self.audit / "verification.json"
        self.verification.write_text(json.dumps(verification), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build(self, output: Path, seed: int = 42) -> dict:
        return build_s2_manifests(
            project_root=self.root,
            inventory_path=self.inventory,
            verification_path=self.verification,
            output_dir=output,
            seed=seed,
            workers=4,
        )

    def test_build_is_deterministic_complete_and_read_only(self) -> None:
        inventory_before = sha256_file(self.inventory)
        dataset_before = tree_fingerprint(self.dataset, workers=4)
        output = self.root / "manifests"
        first = self.build(output)
        content_before = {
            name: (output / name).read_bytes()
            for name in (
                "real_split_observed_v1.jsonl",
                "low187_v1.jsonl",
                "object_environment_review_v1.jsonl",
                "low187_v1.meta.json",
            )
        }
        second = self.build(output)
        content_after = {name: (output / name).read_bytes() for name in content_before}

        self.assertEqual(first["status"], "VERIFIED")
        self.assertEqual(first["content_hashes"], second["content_hashes"])
        self.assertEqual(content_before, content_after)
        self.assertEqual(sha256_file(self.inventory), inventory_before)
        self.assertEqual(tree_fingerprint(self.dataset, workers=4), dataset_before)
        self.assertEqual(first["dataset_tree_sha256"], dataset_before["sha256"])
        self.assertEqual(first["verified_source_files"], 2096)
        self.assertEqual(first["review_count"], 52)
        self.assertEqual(first["historical_orphan_label_count"], 187)

        subset = [
            json.loads(line)
            for line in content_before["low187_v1.jsonl"].decode().splitlines()
        ]
        self.assertEqual(len(subset), 187)
        self.assertEqual(len({row["record_id"] for row in subset}), 187)
        self.assertTrue(all(row["original_split"] == "train" for row in subset))
        self.assertTrue(all(row["raw_source_mapping_status"] == "UNRESOLVED" for row in subset))
        self.assertNotIn("created_at", subset[0])
        for row in subset:
            expected = hashlib.sha256(
                f"{ALGORITHM_ID}|42|{row['record_id']}".encode()
            ).hexdigest()
            self.assertEqual(row["rank_digest"], expected)

        review = [
            json.loads(line)
            for line in content_before[
                "object_environment_review_v1.jsonl"
            ].decode().splitlines()
        ]
        self.assertEqual(len(review), 52)
        self.assertTrue(all(row["status"] == "NEEDS_REVIEW" for row in review))
        self.assertTrue(all(row["chosen_environment"] == "UNDECIDED" for row in review))

        with self.assertRaises(ManifestError):
            self.build(output, seed=43)

        target = self.dataset / "images" / "train" / "train1.jpg"
        target.write_bytes(b"changed")
        with self.assertRaises(ManifestError):
            self.build(self.root / "other_manifests")

    def test_unsafe_paths_and_output_overlap_are_blocked(self) -> None:
        with self.assertRaises(ManifestError):
            normalize_relative_path("../outside.jpg")
        with self.assertRaises(ManifestError):
            normalize_relative_path("D:/outside.jpg")
        with self.assertRaises(ManifestError):
            self.build(self.dataset / "generated")


if __name__ == "__main__":
    unittest.main()
