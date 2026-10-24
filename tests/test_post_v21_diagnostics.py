"""Diagnostic evidence cannot grant approval or change existing routing."""

import json
from copy import deepcopy
from hashlib import sha256
from unittest.mock import patch

import pytest
from test_v1035_final_quality_authority import _AcceptingGate
from test_v2084_term_evidence import _records

from tarjomeh.core import pipeline
from tarjomeh.glossary.book_review import (
    book_term_review_evidence,
    nested_reviewed_term_evidence,
    resolve_reviewed_book_terms,
)
from tarjomeh.quality.critique import CritiqueResult, TranslationCritique
from tarjomeh.quality.grounding import source_segments


def digest(text):
    return sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("overall,expected", [(9, 9), (None, 6.5)])
def test_score_diagnostics_preserve_score_of_record_and_routing(overall, expected):
    data = {"scores": dict(accuracy=2, fluency=8, terminology=8, register=8), "issues": []}
    if overall is not None:
        data["overall"] = overall
    result = TranslationCritique._parse_response(json.dumps(data))
    before = deepcopy(result.to_dict())
    route = pipeline._critique_requires_refinement(result, 8)
    diagnostic = result.diagnostic_evidence()
    event = pipeline._critique_for_event(result, 8, 1)
    assert diagnostic["model_overall"] == overall
    assert diagnostic["dimension_mean"] == 6.5
    assert diagnostic["score_of_record"] == expected == result.average
    assert result.to_dict() == before
    assert pipeline._critique_requires_refinement(result, 8) == route
    assert event["diagnostic_evidence"] == diagnostic
    assert CritiqueResult.from_dict(result.to_dict()).diagnostic_evidence() == diagnostic


def test_missing_historical_model_overall_is_not_inferred_from_average():
    result = CritiqueResult.from_dict({"average": 9, "accuracy": 8})
    assert result.average == 9
    assert result.diagnostic_evidence()["model_overall_evidence"] == "not_recorded"


def test_ignored_severe_grounding_is_visible_but_not_promoted():
    source = "Smith et al. (2012) describe a measured trend."
    target = "The report describes a trend."
    base = dict(category="accuracy", confidence=0.99, rationale="test evidence",
                current_persian_quote="describes a trend", suggested_correction="describes it")
    result = TranslationCritique._parse_response(json.dumps({
        "scores": dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9),
        "issues": [
            {**base, "severity": "minor", "source_quote": "describe a measured trend."},
            {**base, "severity": "critical", "source_quote": source},
        ],
    }), source, target)
    segments = deepcopy(source_segments(source))
    assert len(segments) == 2  # This patch reports, not redesigns, segmentation.
    assert result.valid and len(result.issue_details) == 1
    before = deepcopy(result.to_dict())
    diagnostic = result.diagnostic_evidence()
    assert diagnostic["ignored_severe_issue_count"] == 1
    assert diagnostic["grounding_incomplete"]
    assert "issue_source_segment_unresolved" in (
        diagnostic["ignored_invalid_issues"][0]["validation_errors"]
    )
    assert result.to_dict() == before and source_segments(source) == segments


def _scope():
    source, metadata, _, _ = _records([
        "Governance and meta-governance govern the system.",
    ])
    terms = [
        {"source": term, "target": target, "status": "approved",
         "scope_mode": "evidence_paragraph", "source_evidence_sha256": digest(source)}
        for term, target in (
            ("governance", "\u062d\u06a9\u0645\u0631\u0627\u0646\u06cc"),
            ("meta-governance", "\u0641\u0631\u0627\u062d\u06a9\u0645\u0631\u0627\u0646\u06cc"),
        )
    ]
    matches, _ = resolve_reviewed_book_terms(source, metadata, 2, terms)
    return source, metadata, matches


def test_nested_source_reporting_does_not_alter_matcher_or_authority():
    source, metadata, matches = _scope()
    before = deepcopy(matches)
    overlap = nested_reviewed_term_evidence(source, metadata, matches)
    assert len(overlap) == 1
    assert overlap[0]["shorter_term"] == "governance"
    start, end = overlap[0]["shorter_span"]
    assert source[start:end] == "governance"
    assert matches == before and matches[0]["source_occurrence_count"] == 2
    stale = [{**match, "source_paragraph_hash": "wrong"} for match in matches]
    assert not nested_reviewed_term_evidence(source, metadata, stale)


def test_same_paragraph_repeated_review_still_blocks_but_evidence_is_saved(monkeypatch):
    source, metadata, matches = _scope()
    word = "\u062d\u06a9\u0645\u0631\u0627\u0646\u06cc"
    typo = "\u0645\u06cc\u200c\u0633\u0627\u0632\u0632"
    fixed = "\u0645\u06cc\u200c\u0633\u0627\u0632\u062f"
    previous = word + " " + "\u0641\u0631\u0627" + word + " " + typo + "."
    candidate = previous.replace(typo, fixed)
    calls = []
    real_admission = pipeline._reviewed_term_candidate_admission

    def observed(*args, **kwargs):
        calls.append((args, kwargs))
        return real_admission(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_reviewed_term_candidate_admission", observed)
    final, decisions, report = pipeline._salvage_local_refinement_edits(
        source=source, previous=previous, proposed=candidate,
        issue_details=[{"issue_id": "typo", "source_segment_id": "p1:s1",
                        "source_quote": "govern", "current_persian_quote": typo}],
        issue_decisions=[{"issue_id": "typo", "decision": "accepted",
                          "resulting_span": fixed}],
        integrity_gate=_AcceptingGate(), protected_terms=[], protect_inline_english=False,
        allowed_inline_originals=[], source_metadata=metadata, reviewed_matches=matches,
    )
    assert final == previous and not report["committed_count"]
    assert len(calls) == 2  # Existing coherent and isolated attempts, not extra admission calls.
    evidence = report["attempts"][0]["term_admission"]
    assert not evidence["accepted"] and evidence["violation_count"] == 0
    assert evidence["blocking_review_count"] == 1
    assert evidence["candidate_target_sha256"] == digest(candidate)
    assert evidence["source_sha256"] == digest(source)
    assert evidence["nested_source_matches"]
    assert decisions[0]["term_admission"] == evidence


def test_report_reconstructs_literal_overlap_without_new_scope_decisions():
    source, metadata, matches = _scope()
    events = [{"chunk_index": 0, "event_type": "book_term_scope_resolved",
               "payload": {"matched": matches}}]
    chunks = [{"chunk_index": 0, "text": source, "metadata": metadata,
               "translation": "target", "status": "completed"}]
    snapshot = deepcopy((events, chunks))
    result = book_term_review_evidence({}, events, chunks)
    assert len(result["nested_source_matches"]) == 1
    assert all(item["authority"] == "mandatory_only_in_proven_source_paragraph"
               for item in result["paragraph_term_presence"])
    assert (events, chunks) == snapshot


def test_qa_report_counts_physical_attempts_not_relogged_reviews(tmp_path, monkeypatch):
    from tarjomeh.chunking.chunker import Chunk
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.jobs.database import JobDatabase
    from tarjomeh.web.app import create_app

    config = TarjomehConfig()
    db = JobDatabase(tmp_path / "jobs.db")
    source = tmp_path / "source.txt"
    source.write_text("A claim.", encoding="utf-8")
    db.create_job("diagnostics", source, config.to_dict())
    db.save_chunks("diagnostics", [Chunk(0, "A claim.", "Chapter", "")])
    db.log_chunk_event("diagnostics", 0, "llm_call_attempt", {
        "operation": "critique", "success": True,
    })
    critic = TranslationCritique._parse_response(json.dumps({
        "scores": dict(accuracy=2, fluency=8, terminology=8, register=8),
        "overall": 9, "issues": [],
    }))
    event = pipeline._critique_for_event(critic, 8, 1)
    for _ in range(2):
        db.log_chunk_event("diagnostics", 0, "critique_completed", event)
    monkeypatch.setenv("UI_SECRET_TOKEN", "test")
    with patch("tarjomeh.jobs.database.JobDatabase", return_value=db):
        response = create_app(config).test_client().get(
            "/api/jobs/diagnostics/qa-report", headers={"Authorization": "Bearer test"},
        )
    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert "Physical attempt totals: active=1 lifetime=1" in text
    assert "Subset of total physical attempts" in text
    assert "cached or re-logged reviews" in text
    assert '"dimension_mean": 6.5' in text
    assert '"model_overall": 9.0' in text
    assert len(db.get_chunk_events("diagnostics")) == 3
