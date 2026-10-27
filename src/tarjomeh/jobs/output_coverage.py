"""Read-only, evidence-bound publication coverage; never an admission gate."""
from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Any


def output_coverage(
    chunks: list[dict[str, Any]], identity: dict[str, Any],
    publication: dict[str, Any], actual_sha256: str | None,
) -> dict[str, Any]:
    identity = identity if isinstance(identity, dict) else {}
    publication = publication if isinstance(publication, dict) else {}
    finished = [c for c in chunks if c.get("status") in {"completed", "needs_review"}
                and str(c.get("translation") or "").strip()]
    report: dict[str, Any] = {
        "saved_finished_count": len(finished), "published_finished_count": None,
        "unpublished_finished_count": None, "publication_time": publication.get("published_at"),
        "status": "unknown", "reason": "publication_proof_unavailable",
        "authority": "publication_coverage_only_not_semantic_or_export_permission",
    }
    count = identity.get("persisted_chunk_count")
    boundary = publication.get("boundary_chunk_index")
    expected_file = publication.get("output_sha256")
    indices = [c.get("chunk_index") for c in chunks]
    if (
        type(count) is not int or count <= 0 or type(boundary) is not int
        or boundary + 1 != count or identity.get("lexically_identical") is not True
        or any(type(i) is not int for i in indices) or len(indices) != len(set(indices))
        or not isinstance(expected_file, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_file)
        or not isinstance(identity.get("expected_hash"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", identity["expected_hash"])
        or not actual_sha256
    ):
        return report
    prefix = sorted((c for c in finished if c["chunk_index"] <= boundary), key=lambda c: c["chunk_index"])
    if [c["chunk_index"] for c in prefix] != list(range(count)):
        return report
    canonical = re.sub(r"\s+", " ", "\n\n".join(str(c["translation"]).strip() for c in prefix)).strip()
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_file or digest != identity.get("expected_hash"):
        return dict(report, status="stale", reason="file_or_canonical_scope_changed")
    return dict(report, published_finished_count=count,
                unpublished_finished_count=len(finished) - count,
                status="current" if count == len(finished) else "partial",
                reason="file_hash_and_canonical_prefix_match")


def read_output_coverage(db: Any, job: dict[str, Any]) -> dict[str, Any]:
    """Use existing read APIs; unavailable/active evidence is diagnostic only."""
    unknown = output_coverage([], {}, {}, None)
    try:
        chunks = db.get_chunks(job["id"])
        if not isinstance(chunks, list):
            return unknown
        unknown = output_coverage(chunks, {}, {}, None)
        if job.get("raw_status", job.get("status")) not in {"paused", "paused_error", "completed"}:
            return unknown
        state = db.get_job_artifact(job["id"], "chapter_checkpoints") or {}
        if not isinstance(state, dict):
            return unknown
        publication = state.get("latest_publication") or {}
        identity = db.get_job_artifact(job["id"], "canonical_document_identity") or {}
        if not isinstance(publication, dict) or not isinstance(identity, dict):
            return unknown
        path = Path(str(job.get("output_path") or ""))
        if not publication.get("output_path") or path.is_symlink() or not path.is_file():
            return unknown
        if path.resolve() != Path(publication["output_path"]).resolve():
            return unknown
        before = path.stat()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            return unknown
        return output_coverage(chunks, identity, publication, digest)
    except (OSError, ValueError, TypeError, sqlite3.Error):
        return unknown
