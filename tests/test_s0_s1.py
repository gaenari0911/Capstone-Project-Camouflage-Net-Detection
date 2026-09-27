from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from capstone_lab.audit import inspect_recovery
from capstone_lab.config import load_config
from capstone_lab.errors import ConfigError, StateError
from capstone_lab.scheduler import Scheduler
from capstone_lab.state import StateStore


def campaign_dict(root: Path, *, name: str = "test_campaign") -> dict:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.txt"
    source.write_text("immutable input", encoding="utf-8")
    protected = root / "protected"
    protected.mkdir()
    return {
        "schema_version": 1,
        "status": "frozen",
        "campaign_name": name,
        "paths": {
            "artifact_root": str(root / "artifacts"),
            "protected_roots": [str(protected), str(source)],
            "input_files": [str(source)],
        },
        "resources": {"cpu_slots": 6, "gpu_mib": 12000},
        "jobs": [
            {
                "job_id": "prepare",
                "dependencies": [],
                "resources": {"cpu": 2, "gpu_mib": 0},
                "duration_seconds": 0.03,
                "max_attempts": 1,
                "fail_attempts": [],
            },
            {
                "job_id": "gpu",
                "dependencies": ["prepare"],
                "resources": {"cpu": 2, "gpu_mib": 8000},
                "duration_seconds": 0.05,
                "max_attempts": 2,
                "fail_attempts": [1],
            },
            {
                "job_id": "cpu",
                "dependencies": ["prepare"],
                "resources": {"cpu": 5, "gpu_mib": 0},
                "duration_seconds": 0.05,
                "max_attempts": 1,
                "fail_attempts": [],
            },
            {
                "job_id": "final",
                "dependencies": ["gpu", "cpu"],
                "resources": {"cpu": 1, "gpu_mib": 0},
                "duration_seconds": 0.02,
                "max_attempts": 1,
                "fail_attempts": [],
            },
        ],
    }


class TempCampaign(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, value: dict, name: str = "campaign.json") -> Path:
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path


class ConfigTests(TempCampaign):
    def test_draft_and_unresolved_are_blocked(self) -> None:
        raw = campaign_dict(self.root)
        raw["status"] = "draft"
        path = self.write(raw)
        self.assertEqual(load_config(path).status, "draft")
        with self.assertRaises(ConfigError):
            load_config(path, require_runnable=True)

        raw["status"] = "frozen"
        raw["paths"]["protected_roots"].append("UNDECIDED")
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(path, require_runnable=True)

        raw = campaign_dict(self.root / "unknown")
        raw["unsupported"] = True
        with self.assertRaises(ConfigError):
            load_config(self.write(raw, "unknown.json"))

    def test_protected_output_overlap_is_blocked(self) -> None:
        raw = campaign_dict(self.root)
        raw["paths"]["artifact_root"] = raw["paths"]["protected_roots"][0]
        with self.assertRaises(ConfigError):
            load_config(self.write(raw), require_runnable=True)

    def test_cycle_and_oversized_resource_are_blocked(self) -> None:
        raw = campaign_dict(self.root)
        raw["jobs"][0]["dependencies"] = ["final"]
        with self.assertRaises(ConfigError):
            load_config(self.write(raw), require_runnable=True)


class AuditTests(TempCampaign):
    def test_known_recovery_mismatches_are_reported_read_only(self) -> None:
        audit_root = self.root / "docs" / "recovery_audit"
        audit_root.mkdir(parents=True)
        inventory = audit_root / "current_split_inventory.csv"
        inventory.write_text(
            "split,image,image_sha256,label,label_sha256\n"
            "train,a.jpg,aaa,a.txt,la\n"
            "val,b.jpg,bbb,b.txt,lb\n"
            "test,c.jpg,ccc,c.txt,lc\n",
            encoding="utf-8",
        )
        verification = {
            "outer_crc_test": "PASS",
            "extracted_size_mismatches": [],
            "real_archive_vs_extracted_crc_mismatches": [],
            "object_metadata_mismatch": [{}, {}],
            "synthetic": {
                "images": 2,
                "masks": 2,
                "labels": 3,
                "orphan_labels": ["orphan"],
            },
        }
        (audit_root / "verification.json").write_text(
            json.dumps(verification), encoding="utf-8"
        )
        report = inspect_recovery(self.root)
        self.assertTrue(report["available"])
        self.assertEqual(report["status"], "ATTENTION")
        self.assertEqual(report["cross_split_exact_image_hashes"], 0)
        self.assertEqual(
            {issue["code"]: issue["count"] for issue in report["known_issues"]},
            {
                "OBJECT_METADATA_MISMATCH": 2,
                "HISTORICAL_SYNTHETIC_ORPHAN_LABEL": 1,
            },
        )
        raw = campaign_dict(self.root / "second")
        raw["jobs"][0]["resources"]["cpu"] = 7
        with self.assertRaises(ConfigError):
            load_config(self.write(raw, "other.json"), require_runnable=True)


class SchedulerTests(TempCampaign):
    def run_campaign(self, raw: dict) -> tuple[dict, StateStore, object]:
        path = self.write(raw)
        config = load_config(path, require_runnable=True)
        store = StateStore(config.artifact_root / "_state" / "campaigns.sqlite3")
        result = Scheduler(config, store, poll_seconds=0.005).run()
        return result, store, config

    def test_four_jobs_retry_dependencies_and_resource_budget(self) -> None:
        raw = campaign_dict(self.root)
        for job in raw["jobs"]:
            job["max_attempts"] = max(2, job["max_attempts"])
        result, store, config = self.run_campaign(raw)
        try:
            self.assertEqual(result["status"], "SUCCEEDED")
            rows = {row["job_id"]: row for row in store.rows(config.campaign_name)}
            self.assertEqual(set(rows), {"prepare", "gpu", "cpu", "final"})
            self.assertTrue(all(row["state"] == "SUCCEEDED" for row in rows.values()))
            self.assertEqual(rows["gpu"]["attempt_count"], 2)
            events = store.events(config.campaign_name)
            starts = [event for event in events if event["event_type"] == "JOB_STARTED"]
            waits = [
                event for event in events if event["event_type"] == "JOB_WAITING_RESOURCES"
            ]
            self.assertTrue(all(event["detail"]["cpu_used"] <= 6 for event in starts))
            self.assertTrue(all(event["detail"]["gpu_mib_used"] <= 12000 for event in starts))
            self.assertTrue(waits)
            start_order = [event["job_id"] for event in starts]
            self.assertLess(start_order.index("prepare"), start_order.index("gpu"))
            self.assertLess(start_order.index("prepare"), start_order.index("cpu"))
            self.assertGreater(start_order.index("final"), start_order.index("gpu"))
            self.assertGreater(start_order.index("final"), start_order.index("cpu"))
        finally:
            store.close()

    def test_permanent_failure_blocks_only_dependent_branch(self) -> None:
        raw = campaign_dict(self.root)
        raw["jobs"][1]["max_attempts"] = 2
        raw["jobs"][1]["fail_attempts"] = [1, 2]
        result, store, config = self.run_campaign(raw)
        try:
            rows = {row["job_id"]: row for row in store.rows(config.campaign_name)}
            self.assertEqual(result["status"], "PARTIAL_FAILURE")
            self.assertEqual(rows["gpu"]["state"], "FAILED")
            self.assertEqual(rows["gpu"]["attempt_count"], 2)
            self.assertEqual(rows["final"]["state"], "BLOCKED_DEPENDENCY")
            self.assertEqual(rows["cpu"]["state"], "SUCCEEDED")
        finally:
            store.close()

    def test_resume_skips_valid_success_and_reruns_corrupt_branch(self) -> None:
        raw = campaign_dict(self.root)
        for job in raw["jobs"]:
            job["max_attempts"] = 3
        result, store, config = self.run_campaign(raw)
        try:
            self.assertEqual(result["status"], "SUCCEEDED")
            attempts_before = {
                row["job_id"]: row["attempt_count"] for row in store.rows(config.campaign_name)
            }
            Scheduler(config, store, poll_seconds=0.005).run()
            attempts_after = {
                row["job_id"]: row["attempt_count"] for row in store.rows(config.campaign_name)
            }
            self.assertEqual(attempts_before, attempts_after)

            prepare = store.row(config.campaign_name, "prepare")
            Path(prepare["artifact_path"]).write_text("corrupt", encoding="utf-8")
            Scheduler(config, store, poll_seconds=0.005).run()
            rerun = {
                row["job_id"]: row["attempt_count"] for row in store.rows(config.campaign_name)
            }
            self.assertGreater(rerun["prepare"], attempts_after["prepare"])
            self.assertGreater(rerun["gpu"], attempts_after["gpu"])
            self.assertGreater(rerun["cpu"], attempts_after["cpu"])
            self.assertGreater(rerun["final"], attempts_after["final"])
        finally:
            store.close()

    def test_input_or_config_change_cannot_reuse_campaign(self) -> None:
        raw = campaign_dict(self.root)
        path = self.write(raw)
        config = load_config(path, require_runnable=True)
        store = StateStore(config.artifact_root / "_state" / "campaigns.sqlite3")
        Scheduler(config, store, poll_seconds=0.005).run()
        Path(raw["paths"]["input_files"][0]).write_text("changed", encoding="utf-8")
        changed_input = load_config(path, require_runnable=True)
        with self.assertRaises(StateError):
            Scheduler(changed_input, store, poll_seconds=0.005).run()
        Path(raw["paths"]["input_files"][0]).write_text("immutable input", encoding="utf-8")
        raw["jobs"][3]["duration_seconds"] = 0.03
        path.write_text(json.dumps(raw), encoding="utf-8")
        changed_config = load_config(path, require_runnable=True)
        with self.assertRaises(StateError):
            Scheduler(changed_config, store, poll_seconds=0.005).run()
        store.close()

    def test_stale_running_job_is_recovered_within_attempt_budget(self) -> None:
        raw = campaign_dict(self.root)
        for job in raw["jobs"]:
            job["max_attempts"] = max(2, job["max_attempts"])
        path = self.write(raw)
        config = load_config(path, require_runnable=True)
        store = StateStore(config.artifact_root / "_state" / "campaigns.sqlite3")
        try:
            store.initialize(config)
            fake_log = config.artifact_root / "stale.log"
            store.begin_attempt(
                config.campaign_name,
                config.jobs[0],
                999999,
                fake_log,
                fake_log,
            )
            result = Scheduler(config, store, poll_seconds=0.005).run()
            self.assertEqual(result["status"], "SUCCEEDED")
            prepare = store.row(config.campaign_name, "prepare")
            self.assertEqual(prepare["attempt_count"], 2)
            self.assertTrue(
                any(
                    event["event_type"] == "RECOVER_INTERRUPTED"
                    for event in store.events(config.campaign_name)
                )
            )
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
