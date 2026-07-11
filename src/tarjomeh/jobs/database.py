"""SQLite-backed job database and checkpoint manager for Tarjomeh.

Saves job configurations, chunk translation progress, and serialized memory state.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from tarjomeh.chunking.chunker import Chunk

logger = logging.getLogger(__name__)


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
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
                CREATE TABLE IF NOT EXISTS chunks (
                    job_id TEXT,
                    chunk_index INTEGER,
                    text TEXT,
                    translation TEXT,
                    status TEXT NOT NULL,
                    PRIMARY KEY (job_id, chunk_index)
                )
            """)
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
                CREATE INDEX IF NOT EXISTS idx_chunk_events_job_chunk
                ON chunk_events (job_id, chunk_index, timestamp)
            """)
            conn.commit()

    def create_job(self, job_id: str, input_path: str | Path, config_dict: dict[str, Any]) -> None:
        """Register a new job in the database."""
        created_at = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO jobs (id, input_path, status, config, created_at) VALUES (?, ?, ?, ?, ?)",
                (job_id, str(input_path), JobStatus.PENDING, json.dumps(config_dict), created_at)
            )
            conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve full details of a job."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            job = dict(row)
            try:
                job["config"] = json.loads(job["config"]) if job["config"] else {}
            except json.JSONDecodeError:
                job["config"] = {}
            
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

    def save_chunks(self, job_id: str, chunks: list[Chunk]) -> None:
        """Insert or ignore initial list of chunks for a job."""
        with self._get_connection() as conn:
            for chunk in chunks:
                conn.execute(
                    "INSERT OR IGNORE INTO chunks (job_id, chunk_index, text, status) VALUES (?, ?, ?, ?)",
                    (job_id, chunk.index, chunk.text, ChunkStatus.PENDING)
                )
            conn.commit()

    def get_chunks(self, job_id: str) -> list[dict[str, Any]]:
        """Retrieve all chunk records for a job."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM chunks WHERE job_id = ? ORDER BY chunk_index ASC", (job_id,)).fetchall()
            return [dict(row) for row in rows]

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

        # Delete temporary upload files if they exist in the jobs/uploads folder
        upload_dir = Path("jobs/uploads")
        for p in upload_dir.glob(f"{job_id}.*"):
            try:
                p.unlink()
            except Exception as e:
                logger.warning("Failed to delete upload file %s: %s", p, e)

        return True
