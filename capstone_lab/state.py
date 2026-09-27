from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import CampaignConfig, JobSpec, canonical_json
from .errors import StateError

TERMINAL_STATES = {"SUCCEEDED", "FAILED", "BLOCKED_DEPENDENCY", "CANCELLED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS campaigns (
                campaign_name TEXT PRIMARY KEY,
                config_hash TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                resolved_json TEXT NOT NULL,
                artifact_root TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                campaign_name TEXT NOT NULL,
                job_id TEXT NOT NULL,
                state TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL,
                spec_json TEXT NOT NULL,
                artifact_path TEXT,
                artifact_sha256 TEXT,
                last_error TEXT,
                started_at TEXT,
                finished_at TEXT,
                PRIMARY KEY (campaign_name, job_id),
                FOREIGN KEY (campaign_name) REFERENCES campaigns(campaign_name)
            );
            CREATE TABLE IF NOT EXISTS attempts (
                campaign_name TEXT NOT NULL,
                job_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                state TEXT NOT NULL,
                pid INTEGER,
                stdout_path TEXT,
                stderr_path TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                return_code INTEGER,
                error TEXT,
                PRIMARY KEY (campaign_name, job_id, attempt)
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_name TEXT NOT NULL,
                job_id TEXT,
                event_type TEXT NOT NULL,
                detail_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def initialize(self, config: CampaignConfig) -> None:
        now = utc_now()
        row = self.connection.execute(
            "SELECT * FROM campaigns WHERE campaign_name=?", (config.campaign_name,)
        ).fetchone()
        if row:
            if row["config_hash"] != config.config_hash:
                raise StateError(
                    f"campaign {config.campaign_name} already exists with a different config hash"
                )
            if row["input_hash"] != config.input_hash:
                raise StateError(
                    f"campaign {config.campaign_name} input fingerprint changed; use a new campaign name"
                )
        else:
            self.connection.execute(
                """
                INSERT INTO campaigns
                (campaign_name, config_hash, input_hash, resolved_json, artifact_root, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'PLANNED', ?, ?)
                """,
                (
                    config.campaign_name,
                    config.config_hash,
                    config.input_hash,
                    canonical_json(config.resolved),
                    str(config.artifact_root),
                    now,
                    now,
                ),
            )
        for job in config.jobs:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO jobs
                (campaign_name, job_id, state, max_attempts, spec_json)
                VALUES (?, ?, 'PLANNED', ?, ?)
                """,
                (config.campaign_name, job.job_id, job.max_attempts, canonical_json(job.to_dict())),
            )
        self.connection.commit()

    def event(self, campaign: str, event_type: str, detail: dict[str, Any], job_id: str | None = None) -> None:
        self.connection.execute(
            "INSERT INTO events(campaign_name, job_id, event_type, detail_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (campaign, job_id, event_type, canonical_json(detail), utc_now()),
        )
        self.connection.commit()

    def rows(self, campaign: str) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                "SELECT * FROM jobs WHERE campaign_name=? ORDER BY rowid", (campaign,)
            )
        )

    def row(self, campaign: str, job_id: str) -> sqlite3.Row:
        result = self.connection.execute(
            "SELECT * FROM jobs WHERE campaign_name=? AND job_id=?", (campaign, job_id)
        ).fetchone()
        if result is None:
            raise StateError(f"unknown job {job_id}")
        return result

    def set_campaign_status(self, campaign: str, status: str) -> None:
        self.connection.execute(
            "UPDATE campaigns SET status=?, updated_at=? WHERE campaign_name=?",
            (status, utc_now(), campaign),
        )
        self.connection.commit()

    def set_job_state(
        self,
        campaign: str,
        job_id: str,
        state: str,
        *,
        error: str | None = None,
        artifact_path: str | None = None,
        artifact_sha256: str | None = None,
    ) -> None:
        values: list[Any] = [state, error]
        assignments = ["state=?", "last_error=?"]
        if state == "RUNNING":
            assignments.append("started_at=?")
            values.append(utc_now())
        if state in TERMINAL_STATES:
            assignments.append("finished_at=?")
            values.append(utc_now())
        if artifact_path is not None:
            assignments.extend(["artifact_path=?", "artifact_sha256=?"])
            values.extend([artifact_path, artifact_sha256])
        values.extend([campaign, job_id])
        self.connection.execute(
            f"UPDATE jobs SET {', '.join(assignments)} WHERE campaign_name=? AND job_id=?",
            tuple(values),
        )
        self.connection.commit()

    def clear_for_rerun(self, campaign: str, job_ids: Iterable[str], reason: str) -> None:
        for job_id in job_ids:
            self.connection.execute(
                """
                UPDATE jobs SET state='PLANNED', artifact_path=NULL, artifact_sha256=NULL,
                    last_error=?, started_at=NULL, finished_at=NULL
                WHERE campaign_name=? AND job_id=?
                """,
                (reason, campaign, job_id),
            )
            self.event(campaign, "ARTIFACT_INVALIDATED", {"reason": reason}, job_id)
        self.connection.commit()

    def begin_attempt(
        self, campaign: str, job: JobSpec, pid: int, stdout_path: Path, stderr_path: Path
    ) -> int:
        row = self.row(campaign, job.job_id)
        attempt = int(row["attempt_count"]) + 1
        self.connection.execute(
            "UPDATE jobs SET state='RUNNING', attempt_count=?, last_error=NULL, started_at=? WHERE campaign_name=? AND job_id=?",
            (attempt, utc_now(), campaign, job.job_id),
        )
        self.connection.execute(
            """
            INSERT INTO attempts
            (campaign_name, job_id, attempt, state, pid, stdout_path, stderr_path, started_at)
            VALUES (?, ?, ?, 'RUNNING', ?, ?, ?, ?)
            """,
            (campaign, job.job_id, attempt, pid, str(stdout_path), str(stderr_path), utc_now()),
        )
        self.connection.commit()
        return attempt

    def finish_attempt(
        self, campaign: str, job_id: str, attempt: int, state: str, return_code: int, error: str | None
    ) -> None:
        self.connection.execute(
            """
            UPDATE attempts SET state=?, return_code=?, error=?, finished_at=?
            WHERE campaign_name=? AND job_id=? AND attempt=?
            """,
            (state, return_code, error, utc_now(), campaign, job_id, attempt),
        )
        self.connection.commit()

    def events(self, campaign: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM events WHERE campaign_name=? ORDER BY id", (campaign,)
        )
        return [
            {
                "id": row["id"],
                "job_id": row["job_id"],
                "event_type": row["event_type"],
                "detail": json.loads(row["detail_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
