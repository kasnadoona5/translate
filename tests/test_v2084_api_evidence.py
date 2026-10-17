from unittest.mock import patch

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.glossary.book_review import _sha
from tarjomeh.jobs.database import JobDatabase, JobStatus
from tarjomeh.web.app import create_app


@pytest.mark.parametrize("source,evidence_hash", [("absent", "actual"), ("apparatus", "wrong")])
def test_approval_refuses_unattested_selected_evidence(
    tmp_path, monkeypatch, source, evidence_hash
):
    path = tmp_path / "source.txt"
    text = "The apparatus acts."
    path.write_text(text, encoding="utf-8")
    config = TarjomehConfig()
    config.glossary.path, config.glossary.paths = "", []
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("evidence", path, config.to_dict())
    db.update_job_status("evidence", JobStatus.AWAITING_BOOK_TERM_REVIEW)
    metadata = {
        "paragraph_indices": [0],
        "structural_roles": ["body"],
        "source_paragraph_spans": [[0, len(text)]],
        "source_paragraph_hashes": [_sha(text)],
    }
    db.save_chunks(
        "evidence",
        [Chunk(index=0, text=text, chapter_title="Chapter", section_title="", metadata=metadata)],
    )
    digest = db.get_job_source_hash("evidence")
    db.save_job_artifact(
        "evidence",
        "book_term_body_index_v1",
        {
            "source_sha256": digest,
            "entries": [
                {
                    "chunk_index": 0,
                    "local_paragraph": 0,
                    "paragraph_index": 0,
                    "paragraph_sha256": _sha(text),
                    "chapter_position": 1,
                }
            ],
        },
    )
    review = {
        "phase": "awaiting",
        "review_version": 3,
        "source_sha256": digest,
        "proposals": [
            {
                "source": source,
                "target": "",
                "status": "proposed",
                "source_evidence_sha256": _sha(text) if evidence_hash == "actual" else "0" * 64,
            }
        ],
    }
    db.save_job_artifact("evidence", "book_term_review_v1", review)
    monkeypatch.setenv("UI_SECRET_TOKEN", "test-token")
    client = create_app(config).test_client()
    with patch("tarjomeh.jobs.database.JobDatabase", return_value=db):
        response = client.post(
            "/api/jobs/evidence/book-term-review",
            headers={"Authorization": "Bearer test-token"},
            json={
                "action": "decide",
                "decisions": [
                    {
                        "index": 0,
                        "status": "approved",
                        "target": "دستگاه",
                        "scope_mode": "evidence_paragraph",
                    }
                ],
            },
        )
    assert response.status_code == 409
    assert db.get_job_artifact("evidence", "book_term_review_v1") == review
