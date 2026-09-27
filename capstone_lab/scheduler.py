from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .config import CampaignConfig, JobSpec, sha256_file
from .errors import ArtifactError
from .state import StateStore, TERMINAL_STATES


@dataclass
class ActiveJob:
    spec: JobSpec
    process: subprocess.Popen[bytes]
    attempt: int
    output_path: Path
    stdout: BinaryIO
    stderr: BinaryIO


class Scheduler:
    def __init__(self, config: CampaignConfig, store: StateStore, *, poll_seconds: float = 0.02):
        self.config = config
        self.store = store
        self.poll_seconds = poll_seconds
        self.by_id = {job.job_id: job for job in config.jobs}
        self.active: dict[str, ActiveJob] = {}

    def _descendants(self, roots: set[str]) -> set[str]:
        result = set(roots)
        changed = True
        while changed:
            changed = False
            for job in self.config.jobs:
                if job.job_id not in result and set(job.dependencies) & result:
                    result.add(job.job_id)
                    changed = True
        return result

    def _validate_artifact(self, path: Path, job: JobSpec, attempt: int) -> str:
        if not path.is_file():
            raise ArtifactError(f"missing artifact: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"unreadable artifact: {exc}") from exc
        expected = {
            "schema_version": 1,
            "campaign": self.config.campaign_name,
            "job_id": job.job_id,
            "attempt": attempt,
            "config_hash": self.config.config_hash,
            "input_hash": self.config.input_hash,
            "kind": "fake",
            "validated": True,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ArtifactError(f"artifact field {key} does not match")
        return sha256_file(path)

    def reconcile(self) -> None:
        invalid: set[str] = set()
        for row in self.store.rows(self.config.campaign_name):
            if row["state"] == "SUCCEEDED":
                try:
                    path = Path(row["artifact_path"])
                    if sha256_file(path) != row["artifact_sha256"]:
                        raise ArtifactError("artifact hash changed")
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if (
                        payload.get("config_hash") != self.config.config_hash
                        or payload.get("input_hash") != self.config.input_hash
                        or payload.get("job_id") != row["job_id"]
                    ):
                        raise ArtifactError("artifact identity changed")
                except (OSError, TypeError, json.JSONDecodeError, ArtifactError) as exc:
                    invalid.add(row["job_id"])
                    self.store.event(
                        self.config.campaign_name,
                        "RECONCILE_INVALID",
                        {"error": str(exc)},
                        row["job_id"],
                    )
            elif row["state"] in {"RUNNING", "VALIDATING"}:
                invalid.add(row["job_id"])
                self.store.event(
                    self.config.campaign_name,
                    "RECOVER_INTERRUPTED",
                    {"previous_state": row["state"]},
                    row["job_id"],
                )
        if invalid:
            self.store.clear_for_rerun(
                self.config.campaign_name,
                self._descendants(invalid),
                "upstream or local artifact requires rerun",
            )

    def _usage(self) -> tuple[int, int]:
        return (
            sum(item.spec.resources.cpu for item in self.active.values()),
            sum(item.spec.resources.gpu_mib for item in self.active.values()),
        )

    def _fits(self, job: JobSpec) -> bool:
        cpu, gpu = self._usage()
        return (
            cpu + job.resources.cpu <= self.config.limits.cpu
            and gpu + job.resources.gpu_mib <= self.config.limits.gpu_mib
        )

    def _launch(self, job: JobSpec) -> None:
        row = self.store.row(self.config.campaign_name, job.job_id)
        attempt = int(row["attempt_count"]) + 1
        attempt_dir = (
            self.config.artifact_root
            / self.config.campaign_name
            / "jobs"
            / job.job_id
            / f"attempt_{attempt:03d}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=True)
        output = attempt_dir / "result.json"
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        stdout = stdout_path.open("wb")
        stderr = stderr_path.open("wb")
        command = [
            sys.executable,
            "-m",
            "capstone_lab.fake_job",
            "--campaign",
            self.config.campaign_name,
            "--job",
            job.job_id,
            "--attempt",
            str(attempt),
            "--duration",
            str(job.duration_seconds),
            "--output",
            str(output),
            "--config-hash",
            self.config.config_hash,
            "--input-hash",
            self.config.input_hash,
        ]
        if attempt in job.fail_attempts:
            command.append("--fail")
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OPENCV_FOR_THREADS_NUM": "1",
            }
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=stdout,
                stderr=stderr,
                env=environment,
                shell=False,
                creationflags=creationflags,
            )
        except Exception:
            stdout.close()
            stderr.close()
            raise
        stored_attempt = self.store.begin_attempt(
            self.config.campaign_name, job, process.pid, stdout_path, stderr_path
        )
        if stored_attempt != attempt:
            process.terminate()
            raise RuntimeError("attempt counter changed during launch")
        self.active[job.job_id] = ActiveJob(job, process, attempt, output, stdout, stderr)
        cpu, gpu = self._usage()
        self.store.event(
            self.config.campaign_name,
            "JOB_STARTED",
            {
                "attempt": attempt,
                "pid": process.pid,
                "cpu_used": cpu,
                "gpu_mib_used": gpu,
                "cpu_limit": self.config.limits.cpu,
                "gpu_mib_limit": self.config.limits.gpu_mib,
            },
            job.job_id,
        )

    def _reap(self) -> None:
        for job_id, active in list(self.active.items()):
            return_code = active.process.poll()
            if return_code is None:
                continue
            active.stdout.close()
            active.stderr.close()
            error: str | None = None
            if return_code == 0:
                try:
                    digest = self._validate_artifact(active.output_path, active.spec, active.attempt)
                    self.store.finish_attempt(
                        self.config.campaign_name, job_id, active.attempt, "SUCCEEDED", return_code, None
                    )
                    self.store.set_job_state(
                        self.config.campaign_name,
                        job_id,
                        "SUCCEEDED",
                        artifact_path=str(active.output_path),
                        artifact_sha256=digest,
                    )
                    self.store.event(
                        self.config.campaign_name,
                        "JOB_SUCCEEDED",
                        {"attempt": active.attempt, "artifact_sha256": digest},
                        job_id,
                    )
                except ArtifactError as exc:
                    error = str(exc)
            else:
                error = f"fake job exited with code {return_code}"
            if error is not None:
                self.store.finish_attempt(
                    self.config.campaign_name, job_id, active.attempt, "FAILED", return_code, error
                )
                state = "RETRY_WAIT" if active.attempt < active.spec.max_attempts else "FAILED"
                self.store.set_job_state(self.config.campaign_name, job_id, state, error=error)
                self.store.event(
                    self.config.campaign_name,
                    "JOB_RETRY" if state == "RETRY_WAIT" else "JOB_FAILED",
                    {"attempt": active.attempt, "error": error},
                    job_id,
                )
            del self.active[job_id]

    def _block_failed_dependencies(self) -> bool:
        changed = False
        states = {row["job_id"]: row["state"] for row in self.store.rows(self.config.campaign_name)}
        for job in self.config.jobs:
            if states[job.job_id] not in {"PLANNED", "READY", "RETRY_WAIT"}:
                continue
            failed = [
                dep
                for dep in job.dependencies
                if states[dep] in {"FAILED", "BLOCKED_DEPENDENCY", "CANCELLED"}
            ]
            if failed:
                self.store.set_job_state(
                    self.config.campaign_name,
                    job.job_id,
                    "BLOCKED_DEPENDENCY",
                    error=f"failed dependencies: {failed}",
                )
                self.store.event(
                    self.config.campaign_name,
                    "JOB_BLOCKED",
                    {"failed_dependencies": failed},
                    job.job_id,
                )
                changed = True
        return changed

    def run(self) -> dict[str, object]:
        self.store.initialize(self.config)
        self.reconcile()
        self.store.set_campaign_status(self.config.campaign_name, "RUNNING")
        while True:
            self._reap()
            progressed = self._block_failed_dependencies()
            rows = self.store.rows(self.config.campaign_name)
            states = {row["job_id"]: row["state"] for row in rows}
            dispatched = False
            for job in self.config.jobs:
                if job.job_id in self.active or states[job.job_id] not in {
                    "PLANNED",
                    "READY",
                    "RETRY_WAIT",
                }:
                    continue
                if all(states[dependency] == "SUCCEEDED" for dependency in job.dependencies):
                    row = next(item for item in rows if item["job_id"] == job.job_id)
                    if int(row["attempt_count"]) >= job.max_attempts:
                        self.store.set_job_state(
                            self.config.campaign_name,
                            job.job_id,
                            "FAILED",
                            error="attempt budget exhausted before dispatch",
                        )
                        self.store.event(
                            self.config.campaign_name,
                            "JOB_FAILED",
                            {"error": "attempt budget exhausted before dispatch"},
                            job.job_id,
                        )
                        states[job.job_id] = "FAILED"
                        progressed = True
                        continue
                    if self._fits(job):
                        self._launch(job)
                        states[job.job_id] = "RUNNING"
                        dispatched = True
                    elif states[job.job_id] != "READY":
                        self.store.set_job_state(self.config.campaign_name, job.job_id, "READY")
                        self.store.event(
                            self.config.campaign_name,
                            "JOB_WAITING_RESOURCES",
                            {"cpu": job.resources.cpu, "gpu_mib": job.resources.gpu_mib},
                            job.job_id,
                        )
            rows = self.store.rows(self.config.campaign_name)
            if not self.active and all(row["state"] in TERMINAL_STATES for row in rows):
                break
            if not self.active and not dispatched and not progressed:
                nonterminal = [row["job_id"] for row in rows if row["state"] not in TERMINAL_STATES]
                raise RuntimeError(f"scheduler made no progress: {nonterminal}")
            time.sleep(self.poll_seconds)
        rows = self.store.rows(self.config.campaign_name)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["state"]] = counts.get(row["state"], 0) + 1
        status = "SUCCEEDED" if counts.get("SUCCEEDED") == len(rows) else "PARTIAL_FAILURE"
        self.store.set_campaign_status(self.config.campaign_name, status)
        self.store.event(self.config.campaign_name, "CAMPAIGN_FINISHED", {"status": status, "counts": counts})
        return {"campaign": self.config.campaign_name, "status": status, "counts": counts}
