"""Explicit, zero-provider revision of a paused job's reviewed summary.

Use the host maintenance wrapper for writes. Hash approval binds the operator's
decision, not semantic truth. No normal translation/export stage calls this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import time
from contextlib import closing
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from tarjomeh.memory.summary_revision import (
    document_hash,
    history_key,
    text_hash,
    validate_manifest,
)

PREFIX = "reviewed_summary_revision:"


def _connect(path: Path, *, writable: bool = False) -> sqlite3.Connection:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Existing regular job database required")
    conn = sqlite3.connect(
        path.resolve().as_uri() + ("?mode=rw" if writable else "?mode=ro"), uri=True, timeout=5
    )
    conn.row_factory = sqlite3.Row
    return conn


def _hash_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Existing regular source/backup required")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(conn: sqlite3.Connection, job_id: str) -> tuple[dict, dict]:
    job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    row = conn.execute("SELECT state_data FROM memory_state WHERE job_id=?", (job_id,)).fetchone()
    if not job or not row or job["status"] != "paused":
        raise ValueError("Revision requires an existing paused job and saved memory")
    if (
        conn.execute("SELECT 1 FROM jobs WHERE status IN ('running','pausing') LIMIT 1").fetchone()
        or conn.execute(
            "SELECT 1 FROM job_workers WHERE state IN ('active','pausing') LIMIT 1"
        ).fetchone()
    ):
        raise ValueError("All jobs and durable worker leases must be idle")
    return dict(job), json.loads(row["state_data"])


def _canonical(conn: sqlite3.Connection, job: dict, state: dict, manifest: dict) -> None:
    digest = _hash_file(Path(job["input_path"]))
    if digest != manifest["document_sha256"] or job.get("source_sha256") not in (None, "", digest):
        raise ValueError("Source document has changed")
    refs = manifest["coverage"]
    if state.get("summary_coverage") != refs:
        raise ValueError("Canonical coverage has changed")
    pairs = {
        p["chunk_index"]: p
        for p in state.get("past_translations", [])
        if p.get("active", True) and type(p.get("chunk_index")) is int
    }
    sources = {}
    for ref in refs:
        index = ref["chunk_index"]
        row = conn.execute(
            "SELECT * FROM chunks WHERE job_id=? AND chunk_index=?", (job["id"], index)
        ).fetchone()
        pair = pairs.get(index, {})
        if (
            not row
            or row["status"] not in {"completed", "needs_review"}
            or not isinstance(row["text"], str)
            or not isinstance(row["translation"], str)
            or pair.get("source") != row["text"]
            or pair.get("translation") != row["translation"]
            or text_hash(row["text"]) != ref["source_sha256"]
            or text_hash(row["translation"]) != ref["target_sha256"]
        ):
            raise ValueError("Canonical source/target proof failed")
        sources[index] = row["text"]
    for item in manifest["evidence"]:
        source = sources.get(item["chunk_index"])
        if (
            source is None
            or text_hash(source) != item["source_sha256"]
            or source[item["start"] : item["end"]] != item["quote"]
        ):
            raise ValueError("Source quotation proof failed")


def _verify(conn: sqlite3.Connection, manifest: dict) -> dict:
    validate_manifest(manifest)
    job, state = _read(conn, manifest["job_id"])
    if state.get("quality_policy_version") != 1:
        raise ValueError("Revision requires the existing hash-bound canonical memory policy")
    if document_hash(state) != manifest["expected_memory_sha256"]:
        raise ValueError("stale memory: prepare a new manifest and obtain new approval")
    if state.get("bilingual_summary") != manifest["previous_summary"]:
        raise ValueError("Previous summary has changed")
    _canonical(conn, job, state, manifest)
    known = {history_key(h) for h in state.get("summary_history", []) if isinstance(h, dict)}
    current = history_key({"summary": state["bilingual_summary"], "coverage": manifest["coverage"]})
    known.add(current)
    supersedes = {tuple(k) for k in manifest["supersedes"]}
    if current not in supersedes or not supersedes <= known:
        raise ValueError("Supersession must identify the current and only known history records")
    return state


def prepare(path: Path, proposal: dict) -> dict:
    """Bind a proposed correction to the current complete state, without writes."""
    if set(proposal) != {
        "job_id",
        "document_sha256",
        "previous_summary",
        "candidate_summary",
        "supersedes",
        "evidence",
    }:
        raise ValueError("Invalid proposal fields")
    with closing(_connect(path)) as conn:
        conn.execute("BEGIN")
        _, state = _read(conn, proposal["job_id"])
        manifest = dict(
            deepcopy(proposal),
            schema=1,
            expected_memory_sha256=document_hash(state),
            coverage=deepcopy(state.get("summary_coverage", [])),
        )
        _verify(conn, manifest)
        return manifest


def _backup(path: Path, destination: Path, job_id: str, expected: str) -> dict:
    """Online SQLite snapshot with exclusive creation, integrity and hash receipt."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError("Backup must be a new file in an existing private directory")
    with closing(_connect(path)) as source:
        size = (
            source.execute("PRAGMA page_count").fetchone()[0]
            * source.execute("PRAGMA page_size").fetchone()[0]
        )
        if shutil.disk_usage(destination.parent).free < size + 64 * 1024 * 1024:
            raise ValueError("Insufficient space for a consistent SQLite backup and reserve")
        fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        # Failures retain the partial file for inspection; it is never called verified.
        with closing(sqlite3.connect(destination)) as target:
            deadline = time.monotonic() + 60

            def progress(status, remaining, total):
                if time.monotonic() > deadline:
                    raise TimeoutError("SQLite backup exceeded its time allowance")

            source.backup(target, pages=256, sleep=0.05, progress=progress)
            if target.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("Backup integrity check failed")
            saved = target.execute(
                "SELECT state_data FROM memory_state WHERE job_id=?", (job_id,)
            ).fetchone()
            if not saved or document_hash(json.loads(saved[0])) != expected:
                raise ValueError("stale snapshot: expected memory was not captured")
        with destination.open("r+b") as stream:
            os.fsync(stream.fileno())
    return {"path": str(destination.resolve()), "sha256": _hash_file(destination)}


def _approval(approval: str, expected: str, reviewer: str, locks: bool) -> None:
    if approval != expected or not reviewer.strip() or not locks:
        raise ValueError("Exact hash approval, reviewer and host maintenance locks are required")


def _receipt(conn: sqlite3.Connection, job_id: str, digest: str) -> dict | None:
    row = conn.execute(
        "SELECT payload FROM job_artifacts WHERE job_id=? AND artifact_key=?",
        (job_id, PREFIX + digest),
    ).fetchone()
    return json.loads(row[0]) if row else None


def _write(
    conn: sqlite3.Connection, job_id: str, state: dict, digest: str, receipt: dict, event: str
) -> None:
    timestamp = datetime.now(UTC).isoformat()

    def encode(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)

    conn.execute("UPDATE memory_state SET state_data=? WHERE job_id=?", (encode(state), job_id))
    conn.execute(
        "INSERT INTO job_artifacts VALUES (?,?,?,?) ON CONFLICT(job_id,artifact_key) "
        "DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at",
        (job_id, PREFIX + digest, encode(receipt), timestamp),
    )
    conn.execute(
        "INSERT INTO chunk_events VALUES (?,?,?,?,?)",
        (
            job_id,
            -1,
            timestamp,
            event,
            encode(
                {
                    "revision_id": digest,
                    "reviewer": receipt.get("rollback_reviewer", receipt["reviewer"]),
                    "authority": "explicit_human_review_not_semantic_certification",
                    "state_sha256": document_hash(state),
                }
            ),
        ),
    )


def apply(
    path: Path,
    manifest: dict,
    *,
    approval: str,
    reviewer: str,
    backup: Path,
    operator_locks_held: bool = False,
) -> dict:
    validate_manifest(manifest)
    digest, job_id = document_hash(manifest), manifest["job_id"]
    _approval(approval, digest, reviewer, operator_locks_held)
    with closing(_connect(path)) as conn:
        conn.execute("BEGIN")
        _, state = _read(conn, job_id)
        prior = _receipt(conn, job_id, digest)
        if prior:
            if prior.get("status") != "applied" or document_hash(state) != prior["after_sha256"]:
                raise ValueError("stale or revoked revision receipt")
            job, _ = _read(conn, job_id)
            _canonical(conn, job, state, manifest)
            return dict(prior, status="already_applied")
        _verify(conn, manifest)
    snapshot = _backup(path, backup, job_id, manifest["expected_memory_sha256"])
    with closing(_connect(path, writable=True)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        before = _verify(conn, manifest)
        if _receipt(conn, job_id, digest):
            raise ValueError("Revision was applied concurrently; recheck receipt")
        after = deepcopy(before)
        after["bilingual_summary"] = deepcopy(manifest["candidate_summary"])
        after.setdefault("summary_history", []).append(
            {
                "summary": deepcopy(manifest["candidate_summary"]),
                "coverage": deepcopy(manifest["coverage"]),
                "reviewed_revision": {
                    "schema": 1,
                    "revision_id": digest,
                    "approval_sha256": approval,
                    "reviewer": reviewer,
                    "manifest": deepcopy(manifest),
                },
            }
        )
        receipt = dict(
            status="applied",
            revision_id=digest,
            reviewer=reviewer,
            before_state=before,
            after_sha256=document_hash(after),
            backup=snapshot,
            manifest=deepcopy(manifest),
        )
        _write(conn, job_id, after, digest, receipt, "reviewed_summary_revision")
    return receipt


def rollback(
    path: Path,
    job_id: str,
    digest: str,
    *,
    approval: str,
    reviewer: str,
    backup: Path,
    operator_locks_held: bool = False,
) -> dict:
    def verify(conn):
        job, state = _read(conn, job_id)
        receipt = _receipt(conn, job_id, digest)
        if not receipt:
            raise ValueError("Unknown revision")
        _approval(approval, receipt["after_sha256"], reviewer, operator_locks_held)
        expected = receipt.get("rollback_after_sha256", receipt["after_sha256"])
        if document_hash(state) != expected:
            raise ValueError("stale memory: rollback would overwrite intervening work")
        validate_manifest(receipt["manifest"])
        _canonical(conn, job, state, receipt["manifest"])
        return state, receipt

    with closing(_connect(path)) as conn:
        conn.execute("BEGIN")
        state, receipt = verify(conn)
        if receipt.get("status") == "rolled_back":
            return dict(receipt, status="already_rolled_back")
    snapshot = _backup(path, backup, job_id, receipt["after_sha256"])
    with closing(_connect(path, writable=True)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        state, receipt = verify(conn)
        if receipt.get("status") != "applied":
            raise ValueError("Revision changed concurrently")
        after = deepcopy(state)
        after["bilingual_summary"] = deepcopy(receipt["before_state"]["bilingual_summary"])
        after["summary_history"].append(
            {
                "summary": deepcopy(after["bilingual_summary"]),
                "coverage": deepcopy(after["summary_coverage"]),
                "summary_revision_rollback": {"revision_id": digest},
            }
        )
        receipt.update(
            status="rolled_back",
            rollback_after_sha256=document_hash(after),
            rollback_backup=snapshot,
            rollback_reviewer=reviewer,
        )
        _write(conn, job_id, after, digest, receipt, "summary_revision_rollback")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "check", "apply", "rollback"])
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument(
        "--file", required=True, type=Path, help="Proposal, manifest or rollback receipt"
    )
    parser.add_argument("--approve-sha256", default="")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--operator-locks-held", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.file.read_text(encoding="utf-8"))
    if args.action == "prepare":
        result = prepare(args.db, payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.action == "check":
        with closing(_connect(args.db)) as conn:
            conn.execute("BEGIN")
            _verify(conn, payload)
        print(
            json.dumps(
                {
                    "manifest_sha256": document_hash(payload),
                    "status": "dry_run_only",
                    "semantic_accuracy": "requires_human_review",
                }
            )
        )
        return
    if args.backup is None:
        parser.error("writes require an exclusive new --backup path")
    options = dict(
        approval=args.approve_sha256,
        reviewer=args.reviewer,
        backup=args.backup,
        operator_locks_held=args.operator_locks_held,
    )
    if args.action == "apply":
        result = apply(args.db, payload, **options)
    else:
        result = rollback(args.db, payload["manifest"]["job_id"], payload["revision_id"], **options)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
