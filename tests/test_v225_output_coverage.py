"""Read-only publication scope, not a new export or quality gate."""
import copy
import hashlib
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from tarjomeh.jobs.output_coverage import output_coverage
from tarjomeh.jobs.output_coverage import read_output_coverage


def evidence():
    chunks = [dict(chunk_index=i, status="needs_review", translation=f"text {i}") for i in range(3)]
    identity = dict(persisted_chunk_count=2, lexically_identical=True,
                    expected_hash=hashlib.sha256(b"text 0 text 1").hexdigest())
    publication = dict(boundary_chunk_index=1, output_sha256="a" * 64,
                       output_path="preview.docx", published_at="2026-01-01")
    return chunks, identity, publication


def test_existing_file_covers_only_published_prefix():
    chunks, identity, publication = evidence()
    before = copy.deepcopy((chunks, identity, publication))
    report = output_coverage(chunks, identity, publication, "a" * 64)
    assert report["saved_finished_count"] == 3
    assert report["published_finished_count"] == 2
    assert report["status"] == "partial"
    assert report["unpublished_finished_count"] == 1
    assert (chunks, identity, publication) == before


def test_missing_identity_does_not_guess_count_or_fail():
    chunks, _, publication = evidence()
    report = output_coverage(chunks, {}, publication, "a" * 64)
    assert report["status"] == "unknown"
    assert report["published_finished_count"] is None


def test_changed_file_or_retranslated_prefix_is_not_current():
    chunks, identity, publication = evidence()
    assert output_coverage(chunks, identity, publication, "b" * 64)["status"] == "stale"
    chunks[0]["translation"] = "different text"
    assert output_coverage(chunks, identity, publication, "a" * 64)["status"] == "stale"


def test_gap_in_published_prefix_or_duplicate_rows_is_unknown():
    chunks, identity, publication = evidence()
    assert output_coverage(chunks[1:], identity, publication, "a" * 64)["status"] == "unknown"
    assert output_coverage(chunks + [chunks[0]], identity, publication, "a" * 64)["status"] == "unknown"


def test_full_current_finished_scope_is_not_whole_book_certification():
    chunks, identity, publication = evidence()
    report = output_coverage(chunks[:2], identity, publication, "a" * 64)
    assert report["status"] == "current"
    assert report["authority"] == "publication_coverage_only_not_semantic_or_export_permission"


def test_reader_is_read_only_and_verifies_actual_file(tmp_path):
    chunks, identity, publication = evidence()
    path = tmp_path / "preview.docx"
    path.write_bytes(b"published document")
    publication.update(output_path=str(path), output_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    db = MagicMock()
    db.get_chunks.return_value = chunks
    artifacts = {"chapter_checkpoints": {"latest_publication": publication},
                 "canonical_document_identity": identity}
    db.get_job_artifact.side_effect = lambda _, key: artifacts[key]
    job = dict(id="abc", status="paused", output_path=str(path))
    before = copy.deepcopy((chunks, identity, publication, job))
    assert read_output_coverage(db, job)["status"] == "partial"
    assert {call[0] for call in db.mock_calls} == {"get_chunks", "get_job_artifact"}
    assert (chunks, identity, publication, job) == before
    path.write_bytes(b"changed")
    assert read_output_coverage(db, job)["status"] == "stale"
    job["status"] = "running"
    db.reset_mock()
    assert read_output_coverage(db, job)["status"] == "unknown"
    db.get_job_artifact.assert_not_called()


def test_reader_database_failure_does_not_create_an_export_gate():
    db = MagicMock()
    db.get_chunks.side_effect = sqlite3.OperationalError("unavailable")
    assert read_output_coverage(db, {"id": "abc"})["status"] == "unknown"


@pytest.mark.parametrize("identity_field,value", [("expected_hash", None), ("persisted_chunk_count", True)])
def test_incomplete_proof_is_unknown(identity_field, value):
    chunks, identity, publication = evidence()
    identity[identity_field] = value
    assert output_coverage(chunks, identity, publication, "a" * 64)["status"] == "unknown"


def test_web_history_detail_and_qa_report_labels_are_read_only(monkeypatch):
    from tarjomeh.web.app import create_app

    monkeypatch.setenv("UI_SECRET_TOKEN", "test-token")
    client = create_app().test_client()
    db = MagicMock()
    db.list_jobs.return_value = [{"id": "abc", "status": "paused"}]
    db.get_job.return_value = {"id": "abc", "status": "paused"}
    db.get_chunks.return_value = []
    db.get_chunk_summary.return_value = {}
    db.get_worker_lease.return_value = None
    db.get_job_artifact.return_value = {}
    headers = {"Authorization": "Bearer test-token"}
    with patch("tarjomeh.jobs.database.JobDatabase", return_value=db):
        response = client.get("/api/jobs", headers=headers)
        assert response.status_code == 200
        assert response.json["jobs"][0]["output_coverage"]["status"] == "unknown"
        response = client.get("/api/jobs/abc", headers=headers)
        assert response.status_code == 200
        assert response.json["output_coverage"]["saved_finished_count"] == 0
    assert not any(call[0].startswith(("save", "update", "delete", "create")) for call in db.mock_calls)
