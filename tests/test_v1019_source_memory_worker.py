from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from tarjomeh.context.book_researcher import BookResearcher
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    PipelinePausedException,
    TranslationPipeline,
    _best_source_faithful_version,
    _chunk_memory_admission,
    _chunk_style_policy,
)
from tarjomeh.jobs.database import JobDatabase, JobStatus
from tarjomeh.memory.proper_nouns import ProperNouns


def _critique(
    *,
    accuracy: float = 9.0,
    terminology: float = 9.0,
    fluency: float = 9.0,
    register: float = 9.0,
    issues: list[dict[str, object]] | None = None,
) -> SimpleNamespace:
    scores = (accuracy, terminology, fluency, register)
    return SimpleNamespace(
        accuracy=accuracy,
        terminology=terminology,
        fluency=fluency,
        register=register,
        average=sum(scores) / len(scores),
        issue_details=issues or [],
        issues=[],
        valid=True,
    )


def _event_payload(critique: SimpleNamespace) -> dict[str, object]:
    return {
        "valid": True,
        "blocking_issue_count": 0,
        "scores": {
            "accuracy": critique.accuracy,
            "terminology": critique.terminology,
            "fluency": critique.fluency,
            "register": critique.register,
            "average": critique.average,
        },
        "issue_details": critique.issue_details,
    }


class _EventDB:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = events

    def get_chunk_events(self, _job_id: str, _chunk_index: int):
        return self.events


def test_late_omission_restores_earlier_integrity_valid_version() -> None:
    omission = {
        "issue_id": "missing-relation",
        "category": "omission",
        "severity": "major",
        "confidence": 0.95,
        "source_quote": "the long-term dynamics of institutions and society",
        "current_persian_quote": "the long term",
        "suggested_correction": "the long-term dynamics of institutions and society",
    }
    complete = _critique()
    shortened = _critique(accuracy=8.0, issues=[omission])

    selected = _best_source_faithful_version([
        ("complete target relation", complete),
        ("parenthetical label only", shortened),
    ])

    assert selected is not None
    assert selected[0] == 0
    assert selected[1] == "complete target relation"


def test_grounded_minor_semantic_issue_keeps_continuity_but_not_authority() -> None:
    issue = {
        "issue_id": "scope-at-most",
        "category": "accuracy",
        "severity": "minor",
        "confidence": 0.75,
        "source_quote": "at most six perspectives",
        "current_persian_quote": "at the end six perspectives",
        "suggested_correction": "a maximum of six perspectives",
    }
    critique = _critique(accuracy=9.0, fluency=8.0, issues=[issue])
    db = _EventDB([
        {"event_type": "chunk_started", "payload": {}},
        {"event_type": "critique_completed", "payload": _event_payload(critique)},
        {"event_type": "mqm_minor_only_deferred", "payload": {}},
    ])

    memory = _chunk_memory_admission(db, "job", 0)
    style = _chunk_style_policy(db, "job", 0)

    assert memory["long_term_reliable"] is False
    assert memory["short_term_trust"] == "advisory_review"
    assert memory["continuity_retained"] is True
    assert memory["grounded_unresolved_issue_ids"] == ["scope-at-most"]
    assert style["approved"] is False
    assert style["reason"] == "unresolved_grounded_quality_issue"


def test_lowercase_entity_mapping_does_not_override_common_lexical_sense() -> None:
    memory = ProperNouns()
    memory.add_noun(
        "meridian",
        "\u0645\u0631\u06cc\u062f\u06cc\u0646",
        category="organization",
        provenance="accepted_correction",
        evidence_key="front-matter",
        context_independent=True,
    )

    assert memory.applies_to_source("meridian", "Published by meridian press.")
    assert not memory.applies_to_source(
        "meridian", "The argument crosses a conceptual meridian here."
    )


def test_entity_memory_strips_its_redundant_english_annotation() -> None:
    memory = ProperNouns()
    report = memory.add_noun(
        "Mina Scholar",
        "\u0645\u06cc\u0646\u0627 \u0627\u0633\u06a9\u0627\u0644\u0631 (Mina Scholar, 1948-2014)",
        category="person",
        provenance="observed_translation",
        evidence_key="chunk:1",
        context_independent=True,
    )

    assert report["target"] == "\u0645\u06cc\u0646\u0627 \u0627\u0633\u06a9\u0627\u0644\u0631"
    assert report["stripped_target_annotation"] == "(Mina Scholar, 1948-2014)"


def test_book_identity_evidence_is_not_mislabeled_as_term_evidence() -> None:
    researcher = BookResearcher.__new__(BookResearcher)
    researcher.config = TarjomehConfig()
    url = "https://doi.org/10.1000/book"
    result = researcher._normalise_terms(
        [{
            "source": "relational selectivity",
            "target": (
                "\u06af\u0632\u06cc\u0646\u0634\u06af\u0631\u06cc "
                "\u0631\u0627\u0628\u0637\u0647\u0627\u06cc"
            ),
            "source_urls": [url],
        }],
        [{
            "url": url,
            "title": "A catalogue record for the identified book",
            "snippet": "This record confirms the author and ISBN only.",
            "source_authority": "catalogue",
            "identity_evidence": {"identifier_support": True},
        }],
    )[0]

    assert result["identity_supported"] is True
    assert result["term_supported"] is False
    assert result["term_supporting_excerpts"] == []
    assert result["evidence_type"] == "book_context_only"


def test_worker_lease_prevents_duplicates_and_acknowledges_pause(tmp_path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("job", tmp_path / "book.pdf", {})

    first = db.claim_worker("job", "worker-one")
    duplicate = db.claim_worker("job", "worker-two")
    pause = db.request_job_pause("job")

    assert first["acquired"] is True
    assert duplicate["acquired"] is False
    assert pause["status"] == JobStatus.PAUSING

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.db = db
    pipeline.current_job_id = "job"
    pipeline.worker_id = "worker-one"
    pipeline._worker_claimed = True
    with pytest.raises(PipelinePausedException):
        pipeline._pause_if_requested(
            "job", stage="chunk_checkpoint_committed", chunk_index=4
        )

    assert db.get_job("job")["raw_status"] == JobStatus.PAUSED
    lease = db.get_worker_lease("job")
    assert lease is not None
    assert lease["state"] == "paused"
    events = db.get_chunk_events("job", 4)
    assert events[-1]["event_type"] == "worker_pause_acknowledged"


def test_background_heartbeat_keeps_long_operation_lease_live(tmp_path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("job", tmp_path / "book.pdf", {})
    assert db.claim_worker("job", "worker-one")["acquired"] is True
    with db._get_connection() as conn:
        conn.execute(
            "UPDATE job_workers SET heartbeat_at=? WHERE job_id=?",
            ("2000-01-01T00:00:00", "job"),
        )
        conn.commit()
    before = db.get_worker_lease("job")
    assert before is not None

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.db = db
    pipeline.current_job_id = "job"
    pipeline.worker_id = "worker-one"
    pipeline._worker_claimed = True
    pipeline._worker_stage = "provider_call"
    pipeline._worker_chunk_index = 8
    pipeline._start_lease_heartbeat(interval_seconds=0.01)
    try:
        deadline = time.monotonic() + 1.0
        current = before
        while (
            current["heartbeat_at"] == before["heartbeat_at"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            current = db.get_worker_lease("job") or before
        assert current is not None
        assert current["heartbeat_at"] != before["heartbeat_at"]
        duplicate = db.claim_worker(
            "job", "worker-two", stale_after_seconds=0.5
        )
        assert duplicate["acquired"] is False
    finally:
        pipeline._worker_claimed = False
        pipeline._stop_lease_heartbeat()
