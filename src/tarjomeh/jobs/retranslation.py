"""Job-local disposable state for atomic manual retranslation admission."""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from typing import Any

from tarjomeh.jobs.database import JobDatabase

_TABLES = (
    "jobs",
    "chunks",
    "memory_state",
    "job_artifacts",
    "chunk_events",
    "qa_issues",
    "issue_decisions",
    "job_log",
    "job_workers",
)


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, *args: Any) -> bool:
        try:
            return super().__exit__(*args)
        finally:
            self.close()


class RetranslationDatabase(JobDatabase):
    """Stage quality/canonical writes; mirror physical cost evidence immediately."""

    def __init__(self, path: Any, original: JobDatabase, attempt_id: str) -> None:
        self.original = original
        self.attempt_id = attempt_id
        self.admitted = False
        super().__init__(path)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, factory=_ClosingConnection)
        conn.row_factory = sqlite3.Row
        return conn

    def log_chunk_event(
        self, job_id: str, chunk_index: int, event_type: str, payload: dict[str, Any]
    ) -> None:
        payload = {**payload, "manual_retranslation_attempt": self.attempt_id}
        super().log_chunk_event(job_id, chunk_index, event_type, payload)
        if event_type in {"llm_call_started", "llm_call_attempt", "settings_env_difference"}:
            self.original.log_chunk_event(job_id, chunk_index, event_type, payload)

    def heartbeat_worker(self, *args: Any, **kwargs: Any) -> bool:
        return self.original.heartbeat_worker(*args, **kwargs)

    def get_worker_lease(self, job_id: str) -> dict[str, Any] | None:
        return self.original.get_worker_lease(job_id)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        job = super().get_job(job_id)
        live = self.original.get_job(job_id)
        if job and live:
            for key in ("status", "raw_status"):
                job[key] = live[key]
        return job

    def acknowledge_job_pause(self, *args: Any, **kwargs: Any) -> bool:
        return self.original.acknowledge_job_pause(*args, **kwargs)

    def reserve_extra_final_refinement(self, job_id: str, chunk_index: int, **kwargs: Any) -> bool:
        reserved = self.original.reserve_extra_final_refinement(job_id, chunk_index, **kwargs)
        payload = self.original.get_job_artifact(job_id, "extra_final_refinement_v1")
        if payload is not None:
            super().save_job_artifact(job_id, "extra_final_refinement_v1", payload)
        return reserved

    def merge_job_artifact_entry(
        self, job_id: str, artifact_key: str, *args: Any, **kwargs: Any
    ) -> Any:
        if artifact_key == "extra_final_refinement_v1":
            self.original.merge_job_artifact_entry(job_id, artifact_key, *args, **kwargs)
        return super().merge_job_artifact_entry(job_id, artifact_key, *args, **kwargs)

    def save_job_artifact(self, job_id: str, artifact_key: str, payload: dict[str, Any]) -> None:
        if artifact_key == "web_search_state":
            self.original.save_job_artifact(job_id, artifact_key, payload)
        super().save_job_artifact(job_id, artifact_key, payload)


@contextmanager
def staged_retranslation(
    original: JobDatabase,
    job_id: str,
    chunk_index: int,
) -> Iterator[RetranslationDatabase]:
    """Copy only this job, not the full DB, and publish one admitted transaction.

    A crash discards disposable state. Existing canonical chunks, QA and memory
    stay intact; provider attempts are still visible in the production audit.
    """
    attempt_id = uuid.uuid4().hex
    before = original.get_chunk(job_id, chunk_index)
    if not before:
        raise ValueError("Cannot stage an unknown chunk")
    source_sha256 = original.get_job_source_hash(job_id)
    if not source_sha256:
        raise ValueError("Cannot stage a missing source file")
    with tempfile.TemporaryDirectory(
        prefix="retranslation-", dir=original.db_path.parent
    ) as directory:
        stage = RetranslationDatabase(os.path.join(directory, "attempt.db"), original, attempt_id)
        with stage._get_connection() as conn:
            conn.execute("ATTACH DATABASE ? AS saved", (str(original.db_path.resolve()),))
            conn.execute("BEGIN")
            estimated = 0
            for table in _TABLES:
                columns = [row[1] for row in conn.execute(f'PRAGMA saved.table_info("{table}")')]
                predicate = '"id"=?' if table == "jobs" else '"job_id"=?'
                sizes = "+".join(
                    f'coalesce(length(CAST("{column}" AS BLOB)),0)' for column in columns
                )
                estimated += int(
                    conn.execute(
                        f'SELECT coalesce(sum({sizes}),0) FROM saved."{table}" WHERE {predicate}',
                        (job_id,),
                    ).fetchone()[0]
                )
            if shutil.disk_usage(directory).free < estimated * 2 + 32 * 1024 * 1024:
                raise RuntimeError(
                    "Insufficient space for isolated retranslation; no model call made"
                )
            for table in _TABLES:
                predicate = '"id"=?' if table == "jobs" else '"job_id"=?'
                conn.execute(
                    f'INSERT INTO main."{table}" SELECT * FROM saved."{table}" WHERE {predicate}',
                    (job_id,),
                )
            conn.execute(
                "INSERT INTO main.book_term_decisions SELECT * FROM saved.book_term_decisions "
                "WHERE source_sha256=(SELECT source_sha256 FROM main.jobs WHERE id=?)",
                (job_id,),
            )
            baseline = conn.execute("SELECT coalesce(max(rowid),0) FROM chunk_events").fetchone()[0]
            conn.commit()
        conn.close()
        os.chmod(stage.db_path, 0o600)
        try:
            yield stage
        finally:
            # Preserve attempts even when admission or a later exception fails.
            accepted = stage.admitted
            try:
                source_unchanged = original.get_job_source_hash(job_id) == source_sha256
            except ValueError:
                source_unchanged = False
            with closing(original._get_connection()) as conn, conn:
                conn.execute("ATTACH DATABASE ? AS attempt", (str(stage.db_path.resolve()),))
                conn.execute("BEGIN IMMEDIATE")
                current = conn.execute(
                    "SELECT * FROM chunks WHERE job_id=? AND chunk_index=?", (job_id, chunk_index)
                ).fetchone()
                if accepted and (
                    not current
                    or current["text"] != before["text"]
                    or current["translation"] != before["translation"]
                    or current["status"] != before["status"]
                    or current["metadata"] != before.get("metadata")
                    or not source_unchanged
                ):
                    accepted = False
                if accepted:
                    for table in ("chunks", "qa_issues", "issue_decisions"):
                        conn.execute(
                            f'DELETE FROM main."{table}" WHERE job_id=? AND chunk_index=?',
                            (job_id, chunk_index),
                        )
                        conn.execute(
                            f'INSERT INTO main."{table}" SELECT * FROM attempt."{table}" '
                            "WHERE job_id=? AND chunk_index=?",
                            (job_id, chunk_index),
                        )
                    for table in ("memory_state", "job_artifacts"):
                        conn.execute(
                            f'INSERT OR REPLACE INTO main."{table}" '
                            f'SELECT * FROM attempt."{table}" '
                            "WHERE job_id=?",
                            (job_id,),
                        )
                rows = conn.execute(
                    "SELECT * FROM attempt.chunk_events WHERE rowid>? "
                    "AND event_type NOT IN ('llm_call_started','llm_call_attempt','se"
                    "ttings_env_difference') ORDER BY rowid",
                    (baseline,),
                ).fetchall()
                for row in rows:
                    kind = (
                        row["event_type"]
                        if accepted
                        else "aborted_retranslation/" + row["event_type"]
                    )
                    conn.execute(
                        "INSERT INTO main.chunk_events VALUES (?,?,?,?,?)",
                        (row["job_id"], row["chunk_index"], row["timestamp"], kind, row["payload"]),
                    )
                conn.commit()
            conn.close()
            original.log_chunk_event(
                job_id,
                chunk_index,
                "manual_retranslation_result",
                {
                    "attempt_id": attempt_id,
                    "admitted": accepted,
                    "policy": "isolated_job_state_atomic_canonical_admission",
                },
            )
            if stage.admitted and not accepted:
                raise RuntimeError(
                    "Canonical chunk changed during retranslation; candidate not published"
                )
