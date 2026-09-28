"""The opt-in wait is durable and cannot start on an implicit approval."""

import hashlib
from unittest.mock import MagicMock, patch

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import TranslationPipeline
from tarjomeh.jobs.database import JobDatabase, JobStatus
from tarjomeh.web.app import create_app


def test_book_term_review_requires_confirmation_and_starts_once(tmp_path, monkeypatch):
    source = tmp_path / "book.txt"
    source.write_text("The polity and its institutions.", encoding="utf-8")
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("book-job", source, {})
    db.update_job_status("book-job", JobStatus.AWAITING_BOOK_TERM_REVIEW)
    db.save_job_artifact("book-job", "book_term_review_v1", {
        "phase": "awaiting",
        "source_sha256": db.get_job_source_hash("book-job"),
        "proposals": [{
            "source": "polity", "target": "", "status": "proposed",
            "source_evidence": "The polity and its institutions.",
            "source_evidence_sha256": hashlib.sha256(
                b"The polity and its institutions."
            ).hexdigest(),
        }],
    })
    monkeypatch.setenv("UI_SECRET_TOKEN", "test-token")
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    headers = {"Authorization": "Bearer test-token"}
    route = "/api/jobs/book-job/book-term-review"
    with patch("tarjomeh.jobs.database.JobDatabase", return_value=db):
        assert client.get(route, headers=headers).status_code == 200
        decision = {"action": "decide", "decisions": [{
            "index": 0, "status": "approved", "target": "\u0633\u06cc\u0627\u0633\u062a",
        }]}
        assert client.post(route, headers=headers, json=decision).status_code == 409
        decision["confirm_bulk"] = True
        assert client.post(route, headers=headers, json=decision).status_code == 200
        assert db.get_job("book-job")["raw_status"] == JobStatus.AWAITING_BOOK_TERM_REVIEW
        assert client.post(route, headers=headers, json={"action": "start"}).status_code == 200
        assert db.get_job("book-job")["raw_status"] == JobStatus.PAUSED
        assert db.get_job_artifact("book-job", "book_term_review_v1")["phase"] == "started"
        assert client.post(route, headers=headers, json={"action": "start"}).status_code == 409


def test_skip_records_no_approved_terms(tmp_path, monkeypatch):
    source = tmp_path / "book.txt"
    source.write_text("A source.", encoding="utf-8")
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("skip-job", source, {})
    db.update_job_status("skip-job", JobStatus.AWAITING_BOOK_TERM_REVIEW)
    db.save_job_artifact("skip-job", "book_term_review_v1", {
        "phase": "awaiting", "source_sha256": db.get_job_source_hash("skip-job"),
        "proposals": [{"source": "source", "target": "guess", "status": "proposed"}],
    })
    monkeypatch.setenv("UI_SECRET_TOKEN", "test-token")
    app = create_app()
    app.config["TESTING"] = True
    with patch("tarjomeh.jobs.database.JobDatabase", return_value=db):
        response = app.test_client().post(
            "/api/jobs/skip-job/book-term-review",
            headers={"Authorization": "Bearer test-token"}, json={"action": "skip"},
        )
    assert response.status_code == 200
    saved = db.get_job_artifact("skip-job", "book_term_review_v1")
    assert saved["phase"] == "skipped"
    assert saved["proposals"][0]["status"] == "skipped"


def test_fresh_opt_in_waits_before_chunk_one_and_skip_resumes(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text(("The polity and its institutions are examined. " * 4),
                      encoding="utf-8")
    output = tmp_path / "target.txt"
    config = TarjomehConfig()
    config.output.format = "txt"
    config.translation.review_book_terms_before_translating = True
    config.translation.enable_book_research = False
    config.translation.enable_web_context = False
    config.translation.enable_critique = False
    config.translation.enable_back_translation = False
    config.glossary.enable_auto_extraction = False
    config.memory.enable_4layer = False
    pipeline = TranslationPipeline(config)
    pipeline.db = JobDatabase(tmp_path / "jobs.db")
    pipeline.llm_client.complete = MagicMock(return_value='{"terms": []}')
    pipeline._translate_single_chunk = MagicMock(
        side_effect=lambda **kwargs: "FA " + kwargs["chunk"].text
    )

    pipeline.run(source, output, job_id="fresh-review")
    assert pipeline.db.get_job("fresh-review")["raw_status"] == JobStatus.AWAITING_BOOK_TERM_REVIEW
    assert not output.exists()
    assert not pipeline._translate_single_chunk.called
    review = pipeline.db.get_job_artifact("fresh-review", "book_term_review_v1")
    review["phase"] = "skipped"
    for proposal in review["proposals"]:
        proposal["status"] = "skipped"
    assert pipeline.db.complete_book_term_review("fresh-review", review)

    pipeline.run(source, output, job_id="fresh-review")
    assert pipeline.db.get_job("fresh-review")["raw_status"] == JobStatus.COMPLETED
    assert output.exists()
    assert pipeline._translate_single_chunk.called


def test_approved_term_reaches_prompt_and_final_scoped_check(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("The polity and institutions are examined in this argument.",
                      encoding="utf-8")
    output = tmp_path / "target.txt"
    config = TarjomehConfig()
    config.output.format = "txt"
    config.translation.review_book_terms_before_translating = True
    config.translation.enable_book_research = False
    config.translation.enable_web_context = False
    config.translation.enable_critique = False
    config.translation.enable_back_translation = False
    config.translation.enable_integrity_gate = False
    config.glossary.path = ""
    config.glossary.enable_auto_extraction = False
    config.glossary.enable_auto_correction = False
    config.memory.enable_4layer = False
    pipeline = TranslationPipeline(config)
    pipeline.db = JobDatabase(tmp_path / "jobs.db")
    calls = []

    def fake_completion(*, messages, **kwargs):
        calls.append((messages, kwargs))
        if kwargs.get("_operation") == "auto_term_extraction":
            return '{"terms": []}'
        return "\u062f\u0631 \u0627\u06cc\u0646 \u0628\u062d\u062b \u0633\u06cc\u0627\u0633\u062a \u0648 \u0646\u0647\u0627\u062f\u0647\u0627 \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc\u200c\u0634\u0648\u0646\u062f."

    pipeline.llm_client.complete = MagicMock(side_effect=fake_completion)
    pipeline.run(source, output, job_id="approved-review")
    review = pipeline.db.get_job_artifact("approved-review", "book_term_review_v1")
    review["phase"] = "started"
    review["proposals"] = [{
        "source": "polity", "target": "\u0633\u06cc\u0627\u0633\u062a", "status": "approved",
        "scope_mode": "all_body", "keep_original": False,
    }]
    assert pipeline.db.complete_book_term_review("approved-review", review)

    pipeline.run(source, output, job_id="approved-review")
    events = pipeline.db.get_chunk_events("approved-review", 0)
    resolved = [event for event in events if event["event_type"] == "book_term_scope_resolved"]
    final = [event for event in events if event["event_type"] == "book_term_final_compliance"]
    prompt_sizes = [event for event in events if event["event_type"] == "translation_prompt_composition"]
    assert resolved and resolved[-1]["payload"]["matched"]
    assert final and final[-1]["payload"]["compliant"]
    assert prompt_sizes[-1]["payload"]["book_term_review_mode"] == "opt_in"
    assert prompt_sizes[-1]["payload"]["components"]["glossary"]["chars"] > 0
    assert any("Approved book terms" in str(messages)
               for messages, _ in calls)
