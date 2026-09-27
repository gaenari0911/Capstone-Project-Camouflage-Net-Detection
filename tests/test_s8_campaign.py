from __future__ import annotations

import copy
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from capstone_lab.campaign.contracts import approved_plan, contract, output_path, readiness_plan, validate_config
from capstone_lab.campaign.io import RunLock, atomic_json
from capstone_lab.campaign.supervisor import directory_bytes, launch, owned_process, snapshot, stop_owned
from capstone_lab.errors import ArtifactError, ConfigError, StateError

ROOT = Path(__file__).resolve().parents[1]


def fixture_config(output, duration=.05):
    def job(name, deps=(), fail=(), partial=()):
        return dict(id=name, kind="fixture", dependencies=list(deps), cpu=1, max_attempts=3,
                    duration_seconds=duration, fail_attempts=list(fail), partial_attempts=list(partial))
    return dict(schema_version=1, mode="s8_preflight", output=output, cpu_slots=2, max_parallel=2,
                inputs={}, disk_budget_gib=1, min_free_gib=20, active_seconds_max=600,
                jobs=[job("prepare", partial=(1,)), job("train1", ["prepare"]),
                      job("train2", ["train1"]), job("side", fail=(1,)), job("report", ["train2", "side"])])


def await_terminal(run, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = snapshot(run)
        if state["status"] in {"SUCCEEDED", "PARTIAL_FAILURE", "PAUSED", "FAILED_SUPERVISOR", "BLOCKED_RESOURCES"} and not state.get("supervisor_alive"):
            return state
        time.sleep(.1)
    raise AssertionError(f"campaign timeout: {snapshot(run)}")


class S8Contracts(unittest.TestCase):
    def test_atomic_json_retries_transient_windows_access_denial(self):
        from unittest.mock import patch
        from capstone_lab.campaign import io
        base = ROOT / "artifacts/s8_tests"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            path = Path(directory) / "retry.json"
            real_replace = io.os.replace
            calls = []
            def transient(source, destination):
                calls.append((source, destination))
                if len(calls) < 3:
                    raise PermissionError(5, "transient destination sharing violation")
                return real_replace(source, destination)
            with patch.object(io.os, "replace", side_effect=transient):
                atomic_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            self.assertEqual(len(calls), 3)

    def test_budget_scan_tolerates_atomic_temp_rename(self):
        from unittest.mock import Mock, patch
        transient = Mock()
        transient.is_file.return_value = True
        transient.stat.side_effect = FileNotFoundError("worker committed temporary file")
        with patch.object(Path, "rglob", return_value=[transient]):
            self.assertEqual(directory_bytes(ROOT), 0)

    def test_approval_and_s7_immutability(self):
        _, plan, _ = approved_plan(ROOT)
        self.assertEqual(plan["status"], "VERIFIED")
        result = readiness_plan(ROOT)
        self.assertEqual(result["training_jobs"], 34)
        nodes = {r["node_id"]: r for r in result["nodes"]}
        self.assertIn("validate_anydoor_pilot100", nodes["gate_generative_pilot"]["dependencies"])
        self.assertIn("M0_seed0", nodes["M0_seed1"]["dependencies"])
        self.assertIn("verify_train748_sources", nodes["generate_A1_3000"]["dependencies"])
        self.assertIn("verify_low187_sources", nodes["generate_L_A1_3000"]["dependencies"])

    def test_schema_scope_paths_and_dag_guards(self):
        config = fixture_config("artifacts/s8_tests/example")
        validate_config(ROOT, config)
        for mutation in ("shell", "test", "cycle", "duplicate", "escape", "budget"):
            data = copy.deepcopy(config)
            if mutation == "shell":
                data["jobs"][0]["command"] = "anything"
            elif mutation == "test":
                data["jobs"][0]["kind"] = "final_test"
            elif mutation == "cycle":
                data["jobs"][0]["dependencies"] = ["report"]
            elif mutation == "duplicate":
                data["jobs"].append(data["jobs"][0])
            elif mutation == "escape":
                data["output"] = "Dataset/changed"
            elif mutation == "budget":
                data["disk_budget_gib"] = 101
            with self.assertRaises(ConfigError, msg=mutation):
                validate_config(ROOT, data)

    def test_process_ownership_refuses_foreign_pid(self):
        import os
        import psutil
        record = dict(pid=os.getpid(), created=psutil.Process().create_time(),
                      token="NOT_OUR_TOKEN", module="capstone_lab.campaign")
        self.assertIsNone(owned_process(record))
        with self.assertRaises(StateError):
            stop_owned(record)

    def test_run_lock(self):
        base = ROOT / "artifacts/s8_tests"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            path = Path(directory) / "lock"
            with RunLock(path):
                with self.assertRaises(StateError):
                    with RunLock(path):
                        pass


class S8Integration(unittest.TestCase):
    def setUp(self):
        base = ROOT / "artifacts/s8_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(dir=base))
        self.run = self.directory / "run"
        self.path = self.directory / "config.json"

    def tearDown(self):
        # Only terminate explicit campaign identities from this test-owned directory.
        if (self.run / "owner.json").exists():
            owner = json.loads((self.run / "owner.json").read_text())
            if owned_process(owner):
                stop_owned(owner)
        state = snapshot(self.run)
        for job in state.get("jobs", []):
            if job.get("owner") and owned_process(job["owner"]):
                stop_owned(job["owner"])
        # Preserve test evidence, including failures. No recursive cleanup races with workers.

    def write(self, duration=.05):
        data = fixture_config(self.run.relative_to(ROOT).as_posix(), duration)
        atomic_json(self.path, data)
        return data

    def test_detach_retry_partial_resume_and_hash_guard(self):
        self.write()
        launch(ROOT, self.path)
        state = await_terminal(self.run)
        self.assertEqual(state["status"], "SUCCEEDED", state)
        counts = {j["id"]: j["attempts"] for j in state["jobs"]}
        self.assertEqual(counts["prepare"], 2)
        self.assertEqual(counts["side"], 2)
        launch(ROOT, self.path)
        repeated = await_terminal(self.run)
        self.assertEqual(counts, {j["id"]: j["attempts"] for j in repeated["jobs"]})
        completed = self.run / repeated["jobs"][0]["result"]
        payload = json.loads(completed.read_text())
        payload["status"] = "TAMPERED"
        atomic_json(completed, payload)
        with self.assertRaises(ArtifactError):
            launch(ROOT, self.path)
        data = json.loads(self.path.read_text())
        data["jobs"][0]["duration_seconds"] = .06
        atomic_json(self.path, data)
        with self.assertRaises(StateError):
            launch(ROOT, self.path)

    def test_worker_termination_retries_only_owned_job(self):
        data = self.write(duration=1)
        data["jobs"][0]["partial_attempts"] = []
        atomic_json(self.path, data)
        launch(ROOT, self.path)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            state = snapshot(self.run)
            running = [j for j in state["jobs"] if j["state"] == "RUNNING" and j["id"] == "prepare"]
            if running:
                stop_owned(running[0]["owner"])
                break
            time.sleep(.05)
        self.assertTrue(running)
        final = await_terminal(self.run)
        self.assertEqual(final["status"], "SUCCEEDED")
        self.assertEqual(next(j["attempts"] for j in final["jobs"] if j["id"] == "prepare"), 2)

    def test_failure_isolation_and_bounded_retry(self):
        data = self.write()
        data["jobs"][0]["fail_attempts"] = [1, 2, 3]
        data["jobs"][0]["partial_attempts"] = []
        atomic_json(self.path, data)
        launch(ROOT, self.path)
        state = await_terminal(self.run)
        by_id = {j["id"]: j for j in state["jobs"]}
        self.assertEqual(state["status"], "PARTIAL_FAILURE")
        self.assertEqual(by_id["prepare"]["attempts"], 3)
        self.assertEqual(by_id["side"]["state"], "SUCCEEDED")
        self.assertEqual(by_id["train1"]["state"], "BLOCKED_DEPENDENCY")

    def test_pause_and_coordinator_death_adopts_worker(self):
        data = self.write(duration=1)
        data["jobs"][0]["partial_attempts"] = []
        data["jobs"][3]["fail_attempts"] = []
        atomic_json(self.path, data)
        started = launch(ROOT, self.path)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            state = snapshot(self.run)
            if state["counts"].get("RUNNING", 0) == 2:
                break
            time.sleep(.05)
        running = {j["id"]: j["attempts"] for j in state["jobs"] if j["state"] == "RUNNING"}
        self.assertEqual(len(running), 2)
        stop_owned(started["owner"])
        launch(ROOT, self.path)
        token = json.loads((self.run / "token.json").read_text())["token"]
        atomic_json(self.run / "control.json", {"token": token, "action": "pause"})
        paused = await_terminal(self.run)
        self.assertEqual(paused["status"], "PAUSED")
        for row in paused["jobs"]:
            if row["id"] in running:
                self.assertEqual(row["attempts"], running[row["id"]])
        launch(ROOT, self.path)
        self.assertEqual(await_terminal(self.run)["status"], "SUCCEEDED")


if __name__ == "__main__":
    unittest.main()
