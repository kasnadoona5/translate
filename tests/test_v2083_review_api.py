from unittest.mock import patch

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.glossary.book_review import _sha
from tarjomeh.jobs.database import JobDatabase, JobStatus
from tarjomeh.web.app import create_app


@pytest.mark.parametrize(
    "version,scope,expected",
    [
        (3, "book_wide_recommended", 200),
        (2, "book_wide_recommended", 400),
        (2, "evidence_paragraph", 200),
        (2, "all_body", 409),
        (3, "all_body", 409),
        (99, "evidence_paragraph", 409),
    ],
)
def test_versioned_scope_survives_reload_without_v2_all_body_fallback(
    tmp_path,
    monkeypatch,
    version,
    scope,
    expected,
):
    source = tmp_path / "book.txt"
    source.write_text("The apparatus acts.", encoding="utf-8")
    db = JobDatabase(tmp_path / "jobs.db")
    config = TarjomehConfig()
    config.glossary.path, config.glossary.paths = "", []
    db.create_job("job", source, config.to_dict())
    db.update_job_status("job", JobStatus.AWAITING_BOOK_TERM_REVIEW)
    text = source.read_text(encoding="utf-8")
    metadata = {
        "paragraph_indices": [0],
        "structural_roles": ["body"],
        "source_paragraph_spans": [[0, len(text)]],
        "source_paragraph_hashes": [_sha(text)],
    }
    db.save_chunks(
        "job",
        [Chunk(index=0, text=text, chapter_title="Chapter", section_title="", metadata=metadata)],
    )
    digest = db.get_job_source_hash("job")
    db.save_job_artifact(
        "job",
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
    db.save_job_artifact(
        "job",
        "book_term_review_v1",
        {
            "phase": "awaiting",
            "review_version": version,
            "source_sha256": digest,
            "proposals": [
                {
                    "source": "apparatus",
                    "target": "",
                    "status": "proposed",
                    "source_evidence_sha256": _sha(text),
                }
            ],
        },
    )
    monkeypatch.setenv("UI_SECRET_TOKEN", "test-token")
    client = create_app(config).test_client()
    headers = {"Authorization": "Bearer test-token"}
    route = "/api/jobs/job/book-term-review"
    with patch("tarjomeh.jobs.database.JobDatabase", return_value=db):
        fetched = client.get(route, headers=headers)
        assert fetched.status_code == 200
        if version in {2, 3}:
            assert fetched.json["review"]["proposals"][0]["scope_counts"] == {
                "matched": 1,
                "evidence": 1,
                "elsewhere": 0,
            }
        response = client.post(
            route,
            headers=headers,
            json={
                "action": "decide",
                "confirm_bulk": True,
                "decisions": [
                    {
                        "index": 0,
                        "status": "approved",
                        "target": "دستگاه",
                        "scope_mode": scope,
                    }
                ],
            },
        )
        assert response.status_code == expected
        if expected == 200:
            assert (
                client.get(route, headers=headers).json["review"]["proposals"][0]["scope_mode"]
                == scope
            )
            assert client.post(route, headers=headers, json={"action": "start"}).status_code == 200
            assert client.post(route, headers=headers, json={"action": "start"}).status_code == 409
