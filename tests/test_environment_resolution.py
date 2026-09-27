from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from capstone_lab.config import sha256_file
from capstone_lab.environment_resolution import (
    DECISION_RULE,
    resolve_object_environments_from_disk,
)
from capstone_lab.errors import ManifestError


class EnvironmentResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pool = self.root / "Dataset" / "object_pool"
        self.review = self.root / "manifests" / "review.jsonl"
        self.review.parent.mkdir(parents=True)
        rows = []
        for index in range(52):
            chosen = "snow" if index < 50 else "non_snow"
            csv_environment = "non_snow" if chosen == "snow" else "snow"
            object_id = f"object_{index:03d}"
            image = self.pool / chosen / "images" / f"{object_id}.jpg"
            mask = self.pool / chosen / "masks" / f"{object_id}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"image:{object_id}".encode())
            mask.write_bytes(f"mask:{object_id}".encode())
            rows.append(
                {
                    "object_source_id": object_id,
                    "disk_derived_environment": [chosen],
                    "csv_environment": csv_environment,
                    "status": "NEEDS_REVIEW",
                    "chosen_environment": "UNDECIDED",
                }
            )
        self.review.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_disk_directory_decision_is_verified_immutable_and_read_only(self) -> None:
        output = self.root / "manifests" / "resolution.jsonl"
        review_before = sha256_file(self.review)
        first = resolve_object_environments_from_disk(
            project_root=self.root,
            review_path=self.review,
            object_pool_root=self.pool,
            output_path=output,
        )
        second = resolve_object_environments_from_disk(
            project_root=self.root,
            review_path=self.review,
            object_pool_root=self.pool,
            output_path=output,
        )
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "VERIFIED")
        self.assertEqual(first["decision_rule"], DECISION_RULE)
        self.assertEqual(first["counts"], {"non_snow": 2, "snow": 50})
        self.assertEqual(len(rows), 52)
        self.assertTrue(all(row["status"] == "RESOLVED" for row in rows))
        self.assertTrue(
            all(
                row["chosen_environment"] == row["disk_derived_environment"][0]
                for row in rows
            )
        )
        self.assertEqual(sha256_file(self.review), review_before)

    def test_missing_pair_and_protected_output_are_rejected(self) -> None:
        missing = self.pool / "snow" / "masks" / "object_000.png"
        missing.unlink()
        with self.assertRaises(ManifestError):
            resolve_object_environments_from_disk(
                project_root=self.root,
                review_path=self.review,
                object_pool_root=self.pool,
                output_path=self.root / "manifests" / "resolution.jsonl",
            )
        with self.assertRaises(ManifestError):
            resolve_object_environments_from_disk(
                project_root=self.root,
                review_path=self.review,
                object_pool_root=self.pool,
                output_path=self.pool / "bad.jsonl",
            )


if __name__ == "__main__":
    unittest.main()
