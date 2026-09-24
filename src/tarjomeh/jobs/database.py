"""SQLite-backed job database and checkpoint manager for Tarjomeh.

Saves job configurations, chunk translation progress, and serialized memory state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from tarjomeh.chunking.chunker import Chunk

logger = logging.getLogger(__name__)

# Job ids reach the filesystem (upload cleanup, OCR directories). Restrict
# them to characters that cannot act as glob or path metacharacters.
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


_REDACTED_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("llm", "openrouter", "api_keys"),
    ("llm", "critic", "api_keys"),
)


def _scrub_config_secrets(config: dict[str, Any]) -> dict[str, Any]:
    """Blank credentials in a persisted job config, in place.

    Defence in depth: rows written by earlier releases still contain live
    keys, and no config should ever leave this module carrying one.
    """
    for path in _REDACTED_CONFIG_PATHS:
        node: Any = config
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict) and node.get(path[-1]):
            node[path[-1]] = []
    return config


def _decode_payload_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    try:
        item["payload"] = json.loads(item.get("payload") or "{}")
    except json.JSONDecodeError:
        item["payload"] = {}
    item["candidate_accepted"] = (
        bool(item["candidate_accepted"])
        if "candidate_accepted" in item else None
    )
    return item


def _checkpoint_state(
    conn: sqlite3.Connection,
    job_id: str,
) -> dict[str, Any]:
    """Load a backward-compatible chapter-checkpoint artifact."""
    row = conn.execute(
        "SELECT payload FROM job_artifacts "
        "WHERE job_id=? AND artifact_key='chapter_checkpoints'",
        (job_id,),
    ).fetchone()
    try:
        state = json.loads(row["payload"]) if row else {}
    except (json.JSONDecodeError, TypeError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    reached: list[int] = []
    for value in state.get("reached_positions", []):
        try:
            reached.append(int(value))
        except (TypeError, ValueError):
            continue
    state["version"] = 3
    state["reached_positions"] = sorted(set(reached))
    return state


def _write_checkpoint_state(
    conn: sqlite3.Connection,
    job_id: str,
    state: dict[str, Any],
    timestamp: str,
) -> None:
    conn.execute(
        """
        INSERT INTO job_artifacts (job_id, artifact_key, payload, updated_at)
        VALUES (?, 'chapter_checkpoints', ?, ?)
        ON CONFLICT(job_id, artifact_key) DO UPDATE SET
            payload=excluded.payload,
            updated_at=excluded.updated_at
        """,
        (
            job_id,
            json.dumps(state, ensure_ascii=False, sort_keys=True),
            timestamp,
        ),
    )


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    PAUSING = "pausing"
    COMPLETED = "completed"
    PAUSED = "paused"
    PAUSED_ERROR = "paused_error"
    FAILED = "failed"


class ChunkStatus:
    PENDING = "pending"
    TRANSLATING = "translating"
    TRANSLATED = "translated"
    CRITIQUED = "critiqued"
    REFINED = "refined"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    ERROR = "error"


class JobDatabase:
    """SQLite database for managing job status and translation checkpoints."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_dir = Path("jobs")
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "jobs.db"
        else:
            db_path = Path(db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create schema tables if they do not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    input_path TEXT NOT NULL,
                    output_path TEXT,
                    status TEXT NOT NULL,
                    config TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_workers (
                    job_id TEXT PRIMARY KEY,
                    worker_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    stage TEXT,
                    chunk_index INTEGER,
                    acquired_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    pause_requested_at TEXT,
                    released_at TEXT,
                    release_reason TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    job_id TEXT,
                    chunk_index INTEGER,
                    text TEXT,
                    translation TEXT,
                    metadata TEXT,
                    status TEXT NOT NULL,
                    PRIMARY KEY (job_id, chunk_index)
                )
            """)
            chunk_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
            }
            if "metadata" not in chunk_columns:
                conn.execute("ALTER TABLE chunks ADD COLUMN metadata TEXT")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_state (
                    job_id TEXT PRIMARY KEY,
                    state_data TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_log (
                    job_id TEXT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_events (
                    job_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_artifacts (
                    job_id TEXT NOT NULL,
                    artifact_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, artifact_key)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id TEXT PRIMARY KEY,
                    baseline_job_id TEXT NOT NULL,
                    candidate_job_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_chunks (
                    evaluation_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    result TEXT NOT NULL,
                    PRIMARY KEY (evaluation_id, chunk_index)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_preferences (
                    evaluation_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    preference TEXT NOT NULL,
                    edited_translation TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (evaluation_id, chunk_index)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS approved_benchmark (
                    source_hash TEXT PRIMARY KEY,
                    source_text TEXT NOT NULL,
                    approved_translation TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS qa_issues (
                    job_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    critique_iteration INTEGER NOT NULL,
                    issue_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_quote TEXT NOT NULL,
                    current_persian_quote TEXT NOT NULL,
                    suggested_correction TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (
                        job_id, chunk_index, critique_iteration, issue_id
                    )
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS issue_decisions (
                    job_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    refinement_iteration INTEGER NOT NULL,
                    issue_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    resulting_span TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    candidate_accepted INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (
                        job_id, chunk_index, refinement_iteration, issue_id
                    )
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_events_job_chunk
                ON chunk_events (job_id, chunk_index, timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_qa_issues_job_chunk
                ON qa_issues (job_id, chunk_index, critique_iteration)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_issue_decisions_job_chunk
                ON issue_decisions (job_id, chunk_index, refinement_iteration)
            """)
            # One-time migration: releases before the credential fix stored
            # live API keys in jobs.config. The LIKE pattern matches only a
            # NON-empty list, so this is a no-op once every row is clean.
            legacy = conn.execute(
                """SELECT id, config FROM jobs
                   WHERE config LIKE '%\"api_keys\": [\"%'"""
            ).fetchall()
            for row in legacy:
                try:
                    parsed = json.loads(row["config"] or "{}")
                except json.JSONDecodeError:
                    continue
                conn.execute(
                    "UPDATE jobs SET config = ? WHERE id = ?",
                    (json.dumps(_scrub_config_secrets(parsed)), row["id"]),
                )
            if legacy:
                logger.warning(
                    "Removed persisted API keys from %d legacy job record(s). "
                    "Rotate any key that was stored in jobs.db.",
                    len(legacy),
                )

            conn.commit()

    def create_job(self, job_id: str, input_path: str | Path, config_dict: dict[str, Any]) -> None:
        """Register a new job in the database."""
        if not _JOB_ID_RE.match(job_id or ""):
            raise ValueError(
                f"Invalid job id {job_id!r}: use only letters, digits, "
                "'-' and '_' (max 64 characters)."
            )
        created_at = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO jobs (id, input_path, status, config, created_at) VALUES (?, ?, ?, ?, ?)",
                (job_id, str(input_path), JobStatus.PENDING, json.dumps(config_dict), created_at)
            )
            conn.commit()

    def save_evaluation(
        self,
        baseline_job_id: str,
        candidate_job_id: str,
        summary: dict[str, Any],
        chunks: list[dict[str, Any]],
        evaluation_id: str | None = None,
    ) -> str:
        """Persist one immutable comparison run and its per-chunk evidence."""
        evaluation_id = evaluation_id or uuid.uuid4().hex[:12]
        created_at = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO evaluation_runs
                    (id, baseline_job_id, candidate_job_id, status, summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id, baseline_job_id, candidate_job_id, "completed",
                    json.dumps(summary, ensure_ascii=False, sort_keys=True), created_at,
                ),
            )
            conn.executemany(
                """
                INSERT INTO evaluation_chunks (evaluation_id, chunk_index, result)
                VALUES (?, ?, ?)
                """,
                [
                    (evaluation_id, int(chunk["chunk_index"]),
                     json.dumps(chunk, ensure_ascii=False, sort_keys=True))
                    for chunk in chunks
                ],
            )
            conn.commit()
        return evaluation_id

    def get_evaluation(self, evaluation_id: str) -> dict[str, Any] | None:
        """Return a persisted evaluation with chunks and human decisions."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM evaluation_runs WHERE id = ?", (evaluation_id,)
            ).fetchone()
            if not row:
                return None
            chunk_rows = conn.execute(
                "SELECT result FROM evaluation_chunks WHERE evaluation_id = ? ORDER BY chunk_index",
                (evaluation_id,),
            ).fetchall()
            preference_rows = conn.execute(
                "SELECT * FROM evaluation_preferences WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchall()
        result = dict(row)
        try:
            result["summary"] = json.loads(result["summary"])
        except json.JSONDecodeError:
            result["summary"] = {}
        preferences = {int(item["chunk_index"]): dict(item) for item in preference_rows}
        result["chunks"] = []
        for chunk_row in chunk_rows:
            try:
                chunk = json.loads(chunk_row["result"])
            except json.JSONDecodeError:
                continue
            chunk["preference"] = preferences.get(int(chunk["chunk_index"]))
            result["chunks"].append(chunk)
        return result

    def save_evaluation_preference(
        self,
        evaluation_id: str,
        chunk_index: int,
        preference: str,
        edited_translation: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Save or replace a human decision for one blind comparison."""
        created_at = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO evaluation_preferences
                    (evaluation_id, chunk_index, preference, edited_translation, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_id, chunk_index) DO UPDATE SET
                    preference = excluded.preference,
                    edited_translation = excluded.edited_translation,
                    notes = excluded.notes,
                    created_at = excluded.created_at
                """,
                (evaluation_id, chunk_index, preference, edited_translation, notes, created_at),
            )
            conn.commit()

    def save_approved_benchmark(
        self,
        source_hash: str,
        source_text: str,
        approved_translation: str,
        metadata: dict[str, Any],
    ) -> None:
        """Add a human-approved pair to the private regression corpus."""
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO approved_benchmark
                    (source_hash, source_text, approved_translation, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_hash) DO UPDATE SET
                    approved_translation = excluded.approved_translation,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    source_hash, source_text, approved_translation,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    timestamp, timestamp,
                ),
            )
            conn.commit()

    def list_approved_benchmark(self) -> list[dict[str, Any]]:
        """Return the private human-approved regression corpus."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM approved_benchmark ORDER BY updated_at DESC"
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item["metadata"])
            except json.JSONDecodeError:
                item["metadata"] = {}
            items.append(item)
        return items

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve full details of a job."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            job = dict(row)
            try:
                parsed_config = json.loads(job["config"]) if job["config"] else {}
            except json.JSONDecodeError:
                parsed_config = {}
            job["config"] = _scrub_config_secrets(parsed_config)
            
            # Add progress estimation
            chunks = self.get_chunk_summary(job_id)
            total = chunks["total"]
            completed = chunks["completed"]
            job["progress"] = (completed / total * 100) if total > 0 else 0.0
            
            # Map properties for frontend compatibility
            raw_status = job["status"]
            job["raw_status"] = raw_status
            job["filename"] = Path(job["input_path"]).name
            job["mode"] = job["config"].get("translation", {}).get("mode", "academic")
            job["output_format"] = job["config"].get("output", {}).get("format", "")
            job["output_filename"] = (
                Path(job["output_path"]).name
                if job.get("output_path") else ""
            )
            job["pct"] = job["progress"] / 100.0
            if raw_status in ("running", "pending"):
                job["status"] = "processing"
                
            return job

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all translation jobs with progress estimation."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
            jobs = []
            for row in rows:
                job = dict(row)
                chunks = self.get_chunk_summary(job["id"])
                total = chunks["total"]
                completed = chunks["completed"]
                job["progress"] = (completed / total * 100) if total > 0 else 0.0
                
                try:
                    config_dict = json.loads(job["config"]) if job["config"] else {}
                except Exception:
                    config_dict = {}
                # The raw column used to be returned verbatim, leaking any
                # key it still held. Return the scrubbed structure instead.
                config_dict = _scrub_config_secrets(config_dict)
                job["config"] = config_dict
                
                # Map properties for frontend compatibility
                raw_status = job["status"]
                job["raw_status"] = raw_status
                job["filename"] = Path(job["input_path"]).name
                job["mode"] = config_dict.get("translation", {}).get("mode", "academic")
                job["output_format"] = config_dict.get("output", {}).get("format", "")
                job["output_filename"] = (
                    Path(job["output_path"]).name
                    if job.get("output_path") else ""
                )
                job["pct"] = job["progress"] / 100.0
                if raw_status in ("running", "pending"):
                    job["status"] = "processing"
                    
                jobs.append(job)
            return jobs

    def update_job_status(self, job_id: str, status: str, error_message: str | None = None, output_path: str | Path | None = None) -> None:
        """Update job status, error details, or output path."""
        with self._get_connection() as conn:
            if output_path is not None:
                conn.execute(
                    "UPDATE jobs SET status = ?, error_message = ?, output_path = ? WHERE id = ?",
                    (status, error_message, str(output_path), job_id)
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status = ?, error_message = ? WHERE id = ?",
                    (status, error_message, job_id)
                )
            conn.commit()

    @staticmethod
    def _log_worker_lifecycle(
        conn: sqlite3.Connection,
        job_id: str,
        timestamp: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Append durable worker evidence inside the lease transaction."""
        conn.execute(
            """
            INSERT INTO chunk_events
                (job_id, chunk_index, timestamp, event_type, payload)
            VALUES (?, -1, ?, ?, ?)
            """,
            (
                job_id,
                timestamp,
                event_type,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )

    def claim_worker(
        self,
        job_id: str,
        worker_id: str,
        *,
        stale_after_seconds: float = 180.0,
    ) -> dict[str, Any]:
        """Atomically claim a durable job-worker lease across processes."""
        now = datetime.utcnow()
        timestamp = now.isoformat()
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM job_workers WHERE job_id=?", (job_id,)
            ).fetchone()
            previous = dict(row) if row else {}
            active = bool(
                row and str(row["state"]) in {"active", "pausing"}
            )
            stale = False
            if active:
                try:
                    heartbeat = datetime.fromisoformat(str(row["heartbeat_at"]))
                    stale = (now - heartbeat).total_seconds() > stale_after_seconds
                except (TypeError, ValueError):
                    stale = True
            if active and str(row["worker_id"]) != worker_id and not stale:
                self._log_worker_lifecycle(
                    conn,
                    job_id,
                    timestamp,
                    "worker_claim_rejected",
                    {
                        "worker_id": worker_id,
                        "reason": "active_worker",
                        "owner_worker_id": str(row["worker_id"]),
                        "owner_stage": row["stage"],
                        "owner_chunk_index": row["chunk_index"],
                        "owner_heartbeat_at": row["heartbeat_at"],
                    },
                )
                conn.commit()
                return {
                    "acquired": False,
                    "reason": "active_worker",
                    "worker": previous,
                }

            reclaimed = bool(
                active and str(row["worker_id"]) != worker_id and stale
            )
            renewed = bool(
                active and str(row["worker_id"]) == worker_id
            )
            acquired_at = (
                str(row["acquired_at"])
                if row and str(row["worker_id"]) == worker_id
                else timestamp
            )
            conn.execute(
                """
                INSERT INTO job_workers (
                    job_id,worker_id,state,stage,chunk_index,acquired_at,
                    heartbeat_at,pause_requested_at,released_at,release_reason
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET
                    worker_id=excluded.worker_id,
                    state=excluded.state,
                    stage=excluded.stage,
                    chunk_index=excluded.chunk_index,
                    acquired_at=excluded.acquired_at,
                    heartbeat_at=excluded.heartbeat_at,
                    pause_requested_at=NULL,
                    released_at=NULL,
                    release_reason=NULL
                """,
                (
                    job_id, worker_id, "active", "claimed", None,
                    acquired_at, timestamp, None, None, None,
                ),
            )
            self._log_worker_lifecycle(
                conn,
                job_id,
                timestamp,
                "worker_lease_reclaimed"
                if reclaimed else "worker_lease_renewed"
                if renewed else "worker_lease_acquired",
                {
                    "worker_id": worker_id,
                    "reclaimed": reclaimed,
                    "renewed": renewed,
                    "previous_worker_id": (
                        str(previous.get("worker_id", "")) if reclaimed else ""
                    ),
                    "previous_stage": previous.get("stage") if reclaimed else None,
                    "previous_chunk_index": (
                        previous.get("chunk_index") if reclaimed else None
                    ),
                    "previous_heartbeat_at": (
                        previous.get("heartbeat_at") if reclaimed else None
                    ),
                    "reason": (
                        "stale_lease" if reclaimed
                        else "existing_generation" if renewed
                        else "new_lease"
                    ),
                },
            )
            conn.commit()
        return {
            "acquired": True,
            "reclaimed": reclaimed,
            "renewed": renewed,
            "previous_worker": previous if reclaimed else {},
            "worker_id": worker_id,
        }

    def heartbeat_worker(
        self,
        job_id: str,
        worker_id: str,
        *,
        stage: str = "",
        chunk_index: int | None = None,
    ) -> bool:
        """Refresh a lease only when the caller still owns it."""
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE job_workers
                SET heartbeat_at=?, stage=?, chunk_index=?
                WHERE job_id=? AND worker_id=? AND state IN ('active','pausing')
                """,
                (timestamp, stage, chunk_index, job_id, worker_id),
            )
            conn.commit()
            return bool(cursor.rowcount)

    def request_job_pause(self, job_id: str) -> dict[str, Any]:
        """Record a pause request, distinguishing request from acknowledgement."""
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM job_workers WHERE job_id=?", (job_id,)
            ).fetchone()
            active = bool(
                row and str(row["state"]) in {"active", "pausing"}
            )
            status = JobStatus.PAUSING if active else JobStatus.PAUSED
            conn.execute(
                "UPDATE jobs SET status=?, error_message=NULL WHERE id=?",
                (status, job_id),
            )
            if active:
                conn.execute(
                    """
                    UPDATE job_workers
                    SET state='pausing', pause_requested_at=?, heartbeat_at=?
                    WHERE job_id=? AND worker_id=?
                    """,
                    (timestamp, timestamp, job_id, str(row["worker_id"])),
                )
            self._log_worker_lifecycle(
                conn,
                job_id,
                timestamp,
                "worker_pause_requested",
                {
                    "worker_id": str(row["worker_id"]) if active else None,
                    "active_worker": active,
                    "resulting_status": status,
                    "stage": row["stage"] if active else None,
                    "chunk_index": row["chunk_index"] if active else None,
                },
            )
            conn.commit()
        return {
            "status": status,
            "active_worker": active,
            "worker_id": str(row["worker_id"]) if active else None,
        }

    def acknowledge_job_pause(
        self,
        job_id: str,
        worker_id: str,
        *,
        stage: str = "paused",
        chunk_index: int | None = None,
    ) -> bool:
        """Let the owning worker acknowledge pause after its atomic checkpoint."""
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE job_workers
                SET state='paused', stage=?, chunk_index=?, heartbeat_at=?
                WHERE job_id=? AND worker_id=? AND state IN ('active','pausing')
                """,
                (stage, chunk_index, timestamp, job_id, worker_id),
            )
            if cursor.rowcount:
                conn.execute(
                    "UPDATE jobs SET status=?, error_message=NULL WHERE id=?",
                    (JobStatus.PAUSED, job_id),
                )
                self._log_worker_lifecycle(
                    conn,
                    job_id,
                    timestamp,
                    "worker_pause_persisted",
                    {
                        "worker_id": worker_id,
                        "stage": stage,
                        "chunk_index": chunk_index,
                    },
                )
            conn.commit()
            return bool(cursor.rowcount)

    def release_worker(
        self,
        job_id: str,
        worker_id: str,
        *,
        reason: str = "finished",
    ) -> bool:
        """Release only the lease owned by this worker generation."""
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM job_workers WHERE job_id=? AND worker_id=?",
                (job_id, worker_id),
            ).fetchone()
            cursor = conn.execute(
                """
                UPDATE job_workers
                SET state='released', released_at=?, heartbeat_at=?,
                    release_reason=?
                WHERE job_id=? AND worker_id=? AND state!='released'
                """,
                (timestamp, timestamp, reason[:200], job_id, worker_id),
            )
            if cursor.rowcount:
                self._log_worker_lifecycle(
                    conn,
                    job_id,
                    timestamp,
                    "worker_lease_released",
                    {
                        "worker_id": worker_id,
                        "reason": reason[:200],
                        "previous_state": row["state"] if row else None,
                        "stage": row["stage"] if row else None,
                        "chunk_index": row["chunk_index"] if row else None,
                    },
                )
            conn.commit()
            return bool(cursor.rowcount)

    def get_worker_lease(self, job_id: str) -> dict[str, Any] | None:
        """Return the persisted worker generation for diagnostics."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM job_workers WHERE job_id=?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def log_event(self, job_id: str, level: str, message: str) -> None:
        """Append a log event to the job log table."""
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO job_log (job_id, timestamp, level, message) VALUES (?, ?, ?, ?)",
                (job_id, timestamp, level, message)
            )
            conn.commit()

    def save_job_artifact(
        self,
        job_id: str,
        artifact_key: str,
        payload: dict[str, Any],
    ) -> None:
        """Persist job-scoped structured metadata such as research results."""
        timestamp = datetime.utcnow().isoformat()
        payload_str = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO job_artifacts (job_id, artifact_key, payload, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id, artifact_key) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (job_id, artifact_key, payload_str, timestamp),
            )
            conn.commit()

    def get_job_artifact(self, job_id: str, artifact_key: str) -> dict[str, Any] | None:
        """Return one structured job artifact, or None when absent."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT payload FROM job_artifacts WHERE job_id = ? AND artifact_key = ?",
                (job_id, artifact_key),
            ).fetchone()
        if not row:
            return None
        try:
            value = json.loads(row["payload"])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def merge_job_artifact_entry(
        self,
        job_id: str,
        artifact_key: str,
        collection_key: str,
        entry_key: str,
        entry: dict[str, Any],
        *,
        version: int = 1,
    ) -> None:
        """Merge one JSON artifact entry without losing concurrent writers."""
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT payload FROM job_artifacts "
                "WHERE job_id=? AND artifact_key=?",
                (job_id, artifact_key),
            ).fetchone()
            try:
                payload = json.loads(row["payload"]) if row else {}
            except (json.JSONDecodeError, TypeError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            collection = payload.get(collection_key, {})
            if not isinstance(collection, dict):
                collection = {}
            collection[entry_key] = entry
            payload["version"] = int(version)
            payload[collection_key] = collection
            conn.execute(
                """
                INSERT INTO job_artifacts (job_id, artifact_key, payload, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id, artifact_key) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    job_id,
                    artifact_key,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    timestamp,
                ),
            )
            conn.commit()

    def update_chunk_with_event(
        self,
        job_id: str,
        chunk_index: int,
        status: str,
        translation: str,
        event_type: str,
        payload: dict[str, Any],
        additional_events: list[tuple[str, dict[str, Any]]] | None = None,
    ) -> None:
        """Update canonical chunk text and its audit event atomically."""
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE chunks SET status=?, translation=? "
                "WHERE job_id=? AND chunk_index=?",
                (status, translation, job_id, chunk_index),
            )
            conn.execute(
                """
                INSERT INTO chunk_events
                    (job_id, chunk_index, timestamp, event_type, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    chunk_index,
                    timestamp,
                    event_type,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )
            for additional_type, additional_payload in additional_events or []:
                conn.execute(
                    """
                    INSERT INTO chunk_events
                        (job_id, chunk_index, timestamp, event_type, payload)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        chunk_index,
                        timestamp,
                        additional_type,
                        json.dumps(
                            additional_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                )
            conn.commit()

    def log_chunk_event(
        self,
        job_id: str,
        chunk_index: int,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Append structured per-chunk QA/progress evidence."""
        timestamp = datetime.utcnow().isoformat()
        payload_str = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chunk_events
                    (job_id, chunk_index, timestamp, event_type, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, chunk_index, timestamp, event_type, payload_str),
            )
            conn.commit()

    def get_chunk_events(self, job_id: str, chunk_index: int | None = None) -> list[dict[str, Any]]:
        """Retrieve structured per-chunk event records for a job."""
        with self._get_connection() as conn:
            if chunk_index is None:
                rows = conn.execute(
                    """
                    SELECT * FROM chunk_events
                    WHERE job_id = ?
                    ORDER BY chunk_index ASC, timestamp ASC
                    """,
                    (job_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM chunk_events
                    WHERE job_id = ? AND chunk_index = ?
                    ORDER BY timestamp ASC
                    """,
                    (job_id, chunk_index),
                ).fetchall()

        events: list[dict[str, Any]] = []
        for row in rows:
            event = dict(row)
            try:
                event["payload"] = json.loads(event["payload"])
            except json.JSONDecodeError:
                event["payload"] = {}
            events.append(event)
        return events

    def save_qa_issues(
        self,
        job_id: str,
        chunk_index: int,
        critique_iteration: int,
        issues: list[dict[str, Any]],
    ) -> None:
        """Persist one validated MQM issue set without changing existing events."""
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM qa_issues WHERE job_id=? AND chunk_index=? "
                "AND critique_iteration=?",
                (job_id, chunk_index, critique_iteration),
            )
            for issue in issues:
                issue_id = str(issue.get("issue_id", "")).strip()
                if not issue_id:
                    continue
                payload = dict(issue)
                conn.execute(
                    """
                    INSERT INTO qa_issues (
                        job_id, chunk_index, critique_iteration, issue_id,
                        category, severity, confidence, source_quote,
                        current_persian_quote, suggested_correction, rationale,
                        status, payload, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id, chunk_index, critique_iteration, issue_id,
                        str(issue.get("category", "")),
                        str(issue.get("severity", "")),
                        float(issue.get("confidence", 0.0) or 0.0),
                        str(issue.get("source_quote", "")),
                        str(issue.get("current_persian_quote", "")),
                        str(issue.get("suggested_correction", "")),
                        str(issue.get("rationale", "")),
                        "open",
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        timestamp,
                    ),
                )
            conn.commit()

    def clear_chunk_qa_records(self, job_id: str, chunk_index: int) -> None:
        """Discard superseded structured QA rows before a chunk is rerun."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM qa_issues WHERE job_id=? AND chunk_index=?",
                (job_id, chunk_index),
            )
            conn.execute(
                "DELETE FROM issue_decisions WHERE job_id=? AND chunk_index=?",
                (job_id, chunk_index),
            )
            conn.commit()

    def save_issue_decisions(
        self,
        job_id: str,
        chunk_index: int,
        refinement_iteration: int,
        critique_iteration: int,
        decisions: list[dict[str, Any]],
        *,
        candidate_accepted: bool,
    ) -> None:
        """Persist the translator's decision for each validated MQM issue."""
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM issue_decisions WHERE job_id=? AND chunk_index=? "
                "AND refinement_iteration=?",
                (job_id, chunk_index, refinement_iteration),
            )
            for decision in decisions:
                issue_id = str(decision.get("issue_id", "")).strip()
                if not issue_id:
                    continue
                payload = dict(decision)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO issue_decisions (
                        job_id, chunk_index, refinement_iteration, issue_id,
                        decision, resulting_span, rationale,
                        candidate_accepted, payload, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id, chunk_index, refinement_iteration, issue_id,
                        str(decision.get("decision", "")),
                        str(decision.get("resulting_span", "")),
                        str(decision.get("rationale", "")),
                        int(candidate_accepted),
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        timestamp,
                    ),
                )
                conn.execute(
                    """
                    UPDATE qa_issues SET status=?
                    WHERE job_id=? AND chunk_index=?
                      AND critique_iteration=? AND issue_id=?
                    """,
                    (
                        (
                            str(decision.get("decision", ""))
                            if str(decision.get("commit_status", "")).startswith(
                                "committed"
                            )
                            else str(decision.get("commit_status", ""))
                            or (
                                str(decision.get("decision", ""))
                                if candidate_accepted else "candidate_rejected"
                            )
                        ),
                        job_id,
                        chunk_index, critique_iteration, issue_id,
                    ),
                )
            conn.commit()

    def get_qa_issues(
        self, job_id: str, chunk_index: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM qa_issues WHERE job_id=?"
        params: tuple[Any, ...] = (job_id,)
        if chunk_index is not None:
            query += " AND chunk_index=?"
            params += (chunk_index,)
        query += " ORDER BY chunk_index, critique_iteration, issue_id"
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_decode_payload_row(row) for row in rows]

    def get_issue_decisions(
        self, job_id: str, chunk_index: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM issue_decisions WHERE job_id=?"
        params: tuple[Any, ...] = (job_id,)
        if chunk_index is not None:
            query += " AND chunk_index=?"
            params += (chunk_index,)
        query += " ORDER BY chunk_index, refinement_iteration, issue_id"
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_decode_payload_row(row) for row in rows]

    def save_chunks(self, job_id: str, chunks: list[Chunk]) -> None:
        """Insert or ignore initial list of chunks for a job."""
        with self._get_connection() as conn:
            for chunk in chunks:
                conn.execute(
                    """INSERT OR IGNORE INTO chunks
                       (job_id, chunk_index, text, metadata, status)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        job_id,
                        chunk.index,
                        chunk.text,
                        json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                        ChunkStatus.PENDING,
                    )
                )
            conn.commit()

    def commit_chunk_checkpoint(
        self,
        job_id: str,
        chunk_index: int,
        status: str,
        translation: str | None,
        memory_state: dict[str, Any],
        search_state: dict[str, Any] | None = None,
        paragraph_identity: dict[str, Any] | None = None,
        candidate_selection: dict[str, Any] | None = None,
        canonical_admission: dict[str, Any] | None = None,
        final_quality_admission: dict[str, Any] | None = None,
        source_obligation_resolution: dict[str, Any] | None = None,
        chapter_checkpoint: dict[str, Any] | None = None,
    ) -> None:
        """Persist chunk completion and its memory snapshot in ONE transaction.

        These were three separate commits, so a crash landing between them left
        a chunk marked COMPLETED whose memory contribution was never saved --
        and resume trusts both. The SQL mirrors update_chunk, save_memory_state
        and save_job_artifact exactly; only the transaction boundary changes.
        """
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            if translation is not None:
                conn.execute(
                    "UPDATE chunks SET status = ?, translation = ? "
                    "WHERE job_id = ? AND chunk_index = ?",
                    (status, translation, job_id, chunk_index)
                )
            else:
                conn.execute(
                    "UPDATE chunks SET status = ? WHERE job_id = ? AND chunk_index = ?",
                    (status, job_id, chunk_index)
                )
            conn.execute(
                "INSERT OR REPLACE INTO memory_state (job_id, state_data) VALUES (?, ?)",
                (job_id, json.dumps(memory_state))
            )
            if search_state is not None:
                conn.execute(
                    """
                    INSERT INTO job_artifacts (job_id, artifact_key, payload, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(job_id, artifact_key) DO UPDATE SET
                        payload = excluded.payload,
                        updated_at = excluded.updated_at
                    """,
                    (
                        job_id,
                        "web_search_state",
                        json.dumps(search_state, ensure_ascii=False, sort_keys=True),
                        timestamp,
                    ),
                )
            if paragraph_identity is not None:
                row = conn.execute(
                    "SELECT payload FROM job_artifacts "
                    "WHERE job_id = ? AND artifact_key = ?",
                    (job_id, "canonical_chunk_paragraphs_v1"),
                ).fetchone()
                try:
                    identity_state = json.loads(row["payload"]) if row else {}
                except (json.JSONDecodeError, TypeError):
                    identity_state = {}
                if not isinstance(identity_state, dict):
                    identity_state = {}
                chunks = identity_state.get("chunks", {})
                if not isinstance(chunks, dict):
                    chunks = {}
                chunks[str(chunk_index)] = paragraph_identity
                identity_state = {"version": 1, "chunks": chunks}
                conn.execute(
                    """
                    INSERT INTO job_artifacts (job_id, artifact_key, payload, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(job_id, artifact_key) DO UPDATE SET
                        payload = excluded.payload,
                        updated_at = excluded.updated_at
                    """,
                    (
                        job_id,
                        "canonical_chunk_paragraphs_v1",
                        json.dumps(
                            identity_state, ensure_ascii=False, sort_keys=True
                        ),
                        timestamp,
                    ),
                )
            if candidate_selection is not None:
                conn.execute(
                    """
                    INSERT INTO chunk_events
                        (job_id, chunk_index, timestamp, event_type, payload)
                    VALUES (?, ?, ?, 'final_candidate_selection', ?)
                    """,
                    (
                        job_id,
                        chunk_index,
                        timestamp,
                        json.dumps(
                            candidate_selection,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                )
            if canonical_admission is not None:
                conn.execute(
                    """
                    INSERT INTO chunk_events
                        (job_id, chunk_index, timestamp, event_type, payload)
                    VALUES (?, ?, ?, 'final_canonical_admission', ?)
                    """,
                    (
                        job_id,
                        chunk_index,
                        timestamp,
                        json.dumps(
                            canonical_admission,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                )
            if final_quality_admission is not None:
                conn.execute(
                    """
                    INSERT INTO chunk_events
                        (job_id, chunk_index, timestamp, event_type, payload)
                    VALUES (?, ?, ?, 'final_quality_admission', ?)
                    """,
                    (
                        job_id,
                        chunk_index,
                        timestamp,
                        json.dumps(
                            final_quality_admission,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                )
            if source_obligation_resolution is not None:
                row = conn.execute(
                    "SELECT payload FROM job_artifacts "
                    "WHERE job_id = ? AND artifact_key = ?",
                    (job_id, "source_obligation_recovery_v1"),
                ).fetchone()
                try:
                    recovery_state = json.loads(row["payload"]) if row else {}
                except (json.JSONDecodeError, TypeError):
                    recovery_state = {}
                if not isinstance(recovery_state, dict):
                    recovery_state = {}
                entries = recovery_state.get("entries", {})
                if not isinstance(entries, dict):
                    entries = {}
                entry = entries.get(str(chunk_index))
                committed_hash = (
                    hashlib.sha256(translation.encode("utf-8")).hexdigest()
                    if translation is not None else ""
                )
                admitted_replacement = bool(
                    isinstance(entry, dict)
                    and entry.get("candidate_sha256") != committed_hash
                    and paragraph_identity is not None
                    and candidate_selection is not None
                    and canonical_admission is not None
                    and final_quality_admission is not None
                    and all(
                        record.get(key) == committed_hash
                        for record, key in (
                            (paragraph_identity, "canonical_target_hash"),
                            (candidate_selection, "canonical_target_hash"),
                            (canonical_admission, "canonical_target_hash"),
                            (final_quality_admission, "candidate_target_hash"),
                        )
                    )
                )
                if (
                    isinstance(entry, dict)
                    and entry.get("status") == "pending"
                    and status in {ChunkStatus.COMPLETED, ChunkStatus.NEEDS_REVIEW}
                    and entry.get("source_sha256")
                    == source_obligation_resolution.get("source_sha256")
                    and committed_hash
                    == source_obligation_resolution.get("candidate_sha256")
                    and (
                        entry.get("candidate_sha256") == committed_hash
                        or admitted_replacement
                    )
                ):
                    resolved_entry = dict(entry)
                    resolved_entry.update({
                        "status": "resolved",
                        "resolved_at": timestamp,
                        "resolved_candidate_sha256": (
                            source_obligation_resolution.get("candidate_sha256")
                        ),
                        "replaced_candidate_sha256": (
                            entry.get("candidate_sha256")
                            if admitted_replacement else None
                        ),
                    })
                    resolved_entry.pop("candidate", None)
                    entries[str(chunk_index)] = resolved_entry
                    recovery_state.update({"version": 1, "entries": entries})
                    conn.execute(
                        """
                        INSERT INTO job_artifacts
                            (job_id, artifact_key, payload, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(job_id, artifact_key) DO UPDATE SET
                            payload = excluded.payload,
                            updated_at = excluded.updated_at
                        """,
                        (
                            job_id,
                            "source_obligation_recovery_v1",
                            json.dumps(
                                recovery_state,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            timestamp,
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO chunk_events
                            (job_id, chunk_index, timestamp, event_type, payload)
                        VALUES (?, ?, ?, 'source_obligation_recovery_resolved', ?)
                        """,
                        (
                            job_id,
                            chunk_index,
                            timestamp,
                            json.dumps(
                                source_obligation_resolution,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        ),
                    )
            if chapter_checkpoint is not None:
                checkpoint_state = _checkpoint_state(conn, job_id)
                position = int(chapter_checkpoint["chapter_position"])
                if position not in checkpoint_state["reached_positions"]:
                    pending = dict(chapter_checkpoint)
                    pending.update({
                        "chapter_position": position,
                        "boundary_chunk_index": int(
                            chapter_checkpoint["boundary_chunk_index"]
                        ),
                        "state": "pending_export",
                        "created_at": str(
                            chapter_checkpoint.get("created_at") or timestamp
                        ),
                    })
                    checkpoint_state["pending"] = pending
                    _write_checkpoint_state(
                        conn, job_id, checkpoint_state, timestamp
                    )
            conn.commit()

    def save_pending_chapter_checkpoint(
        self,
        job_id: str,
        checkpoint: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist a recovered checkpoint intent without changing chunk data."""
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            state = _checkpoint_state(conn, job_id)
            position = int(checkpoint["chapter_position"])
            if position not in state["reached_positions"]:
                pending = dict(checkpoint)
                pending.update({
                    "chapter_position": position,
                    "boundary_chunk_index": int(
                        checkpoint["boundary_chunk_index"]
                    ),
                    "state": "pending_export",
                    "created_at": str(
                        checkpoint.get("created_at") or timestamp
                    ),
                })
                state["pending"] = pending
                _write_checkpoint_state(conn, job_id, state, timestamp)
            conn.commit()
        return state

    def complete_chapter_checkpoint(
        self,
        job_id: str,
        checkpoint: dict[str, Any],
        output_path: str | Path,
        *,
        output_sha256: str,
        output_bytes: int,
    ) -> dict[str, Any]:
        """Publish checkpoint metadata and PAUSED status in one transaction."""
        timestamp = datetime.utcnow().isoformat()
        position = int(checkpoint["chapter_position"])
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            state = _checkpoint_state(conn, job_id)
            reached = set(state["reached_positions"])
            reached.add(position)
            publication = {
                **checkpoint,
                "chapter_position": position,
                "boundary_chunk_index": int(
                    checkpoint["boundary_chunk_index"]
                ),
                "state": "published",
                "published_at": timestamp,
                "output_path": str(output_path),
                "output_sha256": output_sha256,
                "output_bytes": int(output_bytes),
            }
            history: list[dict[str, Any]] = []
            for item in state.get("publications", []):
                if not isinstance(item, dict):
                    continue
                try:
                    item_position = int(
                        item.get("chapter_position", -1) or -1
                    )
                except (TypeError, ValueError):
                    continue
                if item_position != position:
                    history.append(item)
            history.append(publication)
            state.update({
                "version": 3,
                "pending": None,
                "reached_positions": sorted(reached),
                "latest_position": position,
                "latest_title": str(checkpoint.get("chapter_title", "")),
                "latest_publication": publication,
                "publications": history,
            })
            _write_checkpoint_state(conn, job_id, state, timestamp)
            conn.execute(
                "UPDATE jobs SET status=?, error_message=NULL, "
                "output_path=? WHERE id=?",
                (JobStatus.PAUSED, str(output_path), job_id),
            )
            conn.commit()
        return state

    def fail_chapter_checkpoint(
        self,
        job_id: str,
        checkpoint: dict[str, Any],
        error_message: str,
    ) -> dict[str, Any]:
        """Keep checkpoint intent retryable while recording export failure."""
        timestamp = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            state = _checkpoint_state(conn, job_id)
            pending = dict(state.get("pending") or checkpoint)
            pending["state"] = "pending_export"
            pending["last_failure_at"] = timestamp
            pending["last_error"] = error_message
            pending["failure_count"] = int(pending.get("failure_count", 0)) + 1
            state["pending"] = pending
            _write_checkpoint_state(conn, job_id, state, timestamp)
            conn.execute(
                "UPDATE jobs SET status=?, error_message=? WHERE id=?",
                (JobStatus.PAUSED_ERROR, error_message, job_id),
            )
            conn.commit()
        return state

    def get_chunks(self, job_id: str) -> list[dict[str, Any]]:
        """Retrieve all chunk records for a job."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM chunks WHERE job_id = ? ORDER BY chunk_index ASC", (job_id,)).fetchall()
            chunks = []
            for row in rows:
                chunk = dict(row)
                try:
                    chunk["metadata"] = json.loads(chunk.get("metadata") or "{}")
                except json.JSONDecodeError:
                    chunk["metadata"] = {}
                chunks.append(chunk)
            return chunks

    def update_chunk(self, job_id: str, chunk_index: int, status: str, translation: str | None = None) -> None:
        """Update status and translation text for a single chunk."""
        with self._get_connection() as conn:
            if translation is not None:
                conn.execute(
                    "UPDATE chunks SET status = ?, translation = ? WHERE job_id = ? AND chunk_index = ?",
                    (status, translation, job_id, chunk_index)
                )
            else:
                conn.execute(
                    "UPDATE chunks SET status = ? WHERE job_id = ? AND chunk_index = ?",
                    (status, job_id, chunk_index)
                )
            conn.commit()

    def get_chunk_summary(self, job_id: str) -> dict[str, int]:
        """Summarize status counts for all chunks in a job."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM chunks WHERE job_id = ? GROUP BY status",
                (job_id,)
            ).fetchall()
            summary = {"total": 0, "completed": 0, "pending": 0, "errors": 0}
            for row in rows:
                status = row["status"]
                cnt = row["cnt"]
                summary["total"] += cnt
                if status in (ChunkStatus.COMPLETED, ChunkStatus.NEEDS_REVIEW):
                    summary["completed"] += cnt
                elif status == ChunkStatus.ERROR:
                    summary["errors"] += cnt
                else:
                    # pending, translating, translated, critiqued, refined are all "pending" completion
                    summary["pending"] += cnt
            return summary

    def save_memory_state(self, job_id: str, memory_state: dict[str, Any]) -> None:
        """Save serialized memory manager state."""
        state_str = json.dumps(memory_state)
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memory_state (job_id, state_data) VALUES (?, ?)",
                (job_id, state_str)
            )
            conn.commit()

    def get_memory_state(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve serialized memory manager state."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT state_data FROM memory_state WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return None
            try:
                return json.loads(row["state_data"])
            except json.JSONDecodeError:
                return None

    def cleanup_job(self, job_id: str) -> bool:
        """Delete temporary and intermediate artifacts for a job."""
        job = self.get_job(job_id)
        if not job:
            return False

        # Preserving database records and logs. Do NOT run DELETE database statements.
        pass

        # Delete temporary upload files. Compare stems rather than building a
        # glob from the job id, so no id can ever act as a pattern.
        upload_dir = Path("jobs/uploads")
        if upload_dir.is_dir():
            for p in upload_dir.iterdir():
                if p.is_file() and p.stem == job_id:
                    try:
                        p.unlink()
                    except Exception as e:
                        logger.warning(
                            "Failed to delete upload file %s: %s", p, e
                        )

        # OCR output for this job lives under jobs/ocr/<job_id>/.
        ocr_dir = Path("jobs") / "ocr" / job_id
        if ocr_dir.is_dir():
            try:
                shutil.rmtree(ocr_dir)
            except Exception as e:
                logger.warning("Failed to delete OCR dir %s: %s", ocr_dir, e)

        return True
