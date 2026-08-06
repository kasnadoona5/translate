from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from tarjomeh.core.pipeline import (
    _critique_passes_quality_gate,
    _critique_requires_refinement,
)
from tarjomeh.jobs.database import JobDatabase
from tarjomeh.quality.critique import CritiqueResult, TranslationCritique
from tarjomeh.quality.refiner import TranslationRefiner


def _critique_json(*, source_quote: str = "the field") -> str:
    return json.dumps({
        "scores": {
            "accuracy": 8.5,
            "fluency": 9,
            "terminology": 8,
            "register": 9,
        },
        "overall": 8.6,
        "issues": [{
            "category": "terminology",
            "severity": "major",
            "confidence": 0.97,
            "source_quote": source_quote,
            "current_persian_quote": "حوزه",
            "suggested_correction": "میدان",
            "rationale": "The glossary requires میدان.",
        }],
    }, ensure_ascii=False)


def test_mqm_issue_is_grounded_canonical_and_stable() -> None:
    first = TranslationCritique._parse_response(
        _critique_json(), "Capital organizes the field.", "سرمایه حوزه را سامان می‌دهد."
    )
    second = TranslationCritique._parse_response(
        _critique_json(), "Capital organizes the field.", "سرمایه حوزه را سامان می‌دهد."
    )

    assert first.valid
    assert first.issue_details[0]["source_quote"] == "the field"
    assert first.issue_details[0]["current_persian_quote"] == "حوزه"
    assert first.issue_details[0]["issue_id"].startswith("mqm-")
    assert first.issue_details[0]["issue_id"] == second.issue_details[0]["issue_id"]


def test_mqm_issue_with_invented_span_is_invalid() -> None:
    result = TranslationCritique._parse_response(
        _critique_json(source_quote="invented source"),
        "Capital organizes the field.",
        "سرمایه حوزه را سامان می‌دهد.",
    )

    assert not result.valid
    assert "issue_source_quote_not_found" in result.validation_errors


def test_minor_only_mqm_issue_does_not_force_refinement() -> None:
    critique = CritiqueResult(
        accuracy=9,
        fluency=8,
        terminology=9,
        register=9,
        average=8.75,
        issues=["[MINOR/fluency] style preference"],
        issue_details=[{
            "issue_id": "mqm-minor",
            "category": "fluency",
            "severity": "minor",
        }],
    )

    assert not _critique_requires_refinement(critique, 9.0)
    assert _critique_passes_quality_gate(critique, 9.0)


def _minor_critique(*, category: str, confidence: float) -> CritiqueResult:
    return CritiqueResult(
        accuracy=9,
        fluency=9,
        terminology=9,
        register=9,
        average=9,
        issues=[f"[MINOR/{category}] grounded concern"],
        issue_details=[{
            "issue_id": f"mqm-minor-{category}",
            "category": category,
            "severity": "minor",
            "confidence": confidence,
            "source_quote": "capital's tendency",
            "current_persian_quote": "تمایل سرمایه",
            "suggested_correction": "گرایش سرمایه",
        }],
    )


def test_high_confidence_semantic_minor_requests_bounded_refinement() -> None:
    for category in ("accuracy", "omission", "terminology"):
        critique = _minor_critique(category=category, confidence=0.85)

        assert _critique_requires_refinement(critique, 9.0)
        assert not _critique_passes_quality_gate(critique, 9.0)


def test_minor_routing_excludes_low_confidence_and_style_preferences() -> None:
    low_confidence = _minor_critique(category="accuracy", confidence=0.84)
    style_preference = _minor_critique(category="fluency", confidence=0.99)

    assert not _critique_requires_refinement(low_confidence, 9.0)
    assert not _critique_requires_refinement(style_preference, 9.0)


def test_refiner_requires_one_balanced_decision_per_issue() -> None:
    issue = {
        "issue_id": "mqm-field",
        "category": "terminology",
        "severity": "major",
        "confidence": 0.97,
        "source_quote": "the field",
        "current_persian_quote": "حوزه",
        "suggested_correction": "میدان",
        "rationale": "Glossary requirement.",
    }

    class FakeLLM:
        async def chat(self, prompt: str) -> str:
            assert '"issue_id":"mqm-field"' in prompt
            return json.dumps({
                "translation": "سرمایه میدان را سامان می‌دهد.",
                "decision": "revised",
                "rationale": "The terminology concern was accepted.",
                "issue_decisions": [{
                    "issue_id": "mqm-field",
                    "decision": "accepted",
                    "resulting_span": "میدان",
                    "rationale": "It is mandatory and accurate.",
                }],
            }, ensure_ascii=False)

    critique = CritiqueResult(
        average=8.6,
        issues=["[MAJOR/terminology] field"],
        issue_details=[issue],
    )
    result = asyncio.run(TranslationRefiner(FakeLLM()).refine_with_decision(
        "Capital organizes the field.", "سرمایه حوزه را سامان می‌دهد.", critique
    ))

    assert result.valid
    assert result.issue_decisions[0]["decision"] == "accepted"
    assert result.issue_decisions[0]["resulting_span"] == "میدان"


def test_missing_refiner_issue_decision_is_rejected() -> None:
    raw = json.dumps({
        "translation": "ترجمه",
        "decision": "preserved",
        "rationale": "Preserved.",
        "issue_decisions": [],
    }, ensure_ascii=False)
    result = TranslationRefiner._parse_response(raw, ["mqm-required"])

    assert not result.valid
    assert "missing_issue_decision:mqm-required" in result.validation_errors


def test_refiner_decision_span_must_exist_in_returned_translation() -> None:
    raw = json.dumps({
        "translation": "ترجمه نهایی معتبر است.",
        "decision": "revised",
        "rationale": "Applied the grounded concern.",
        "issue_decisions": [{
            "issue_id": "mqm-required",
            "decision": "accepted",
            "resulting_span": "عبارتی که برگردانده نشد",
            "rationale": "Claimed to be present.",
        }],
    }, ensure_ascii=False)
    result = TranslationRefiner._parse_response(raw, ["mqm-required"])

    assert not result.valid
    assert "resulting_span_not_in_translation:mqm-required" in (
        result.validation_errors
    )


def test_q3_tables_persist_issues_and_decisions_additively() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        db = JobDatabase(Path(temp_dir) / "jobs.db")
        db.create_job("q3-job", "book.txt", {})
        issue = {
            "issue_id": "mqm-field",
            "category": "terminology",
            "severity": "major",
            "confidence": 0.97,
            "source_quote": "the field",
            "current_persian_quote": "حوزه",
            "suggested_correction": "میدان",
            "rationale": "Glossary requirement.",
        }
        db.save_qa_issues("q3-job", 0, 0, [issue])
        db.save_issue_decisions(
            "q3-job", 0, 1, 0,
            [{
                "issue_id": "mqm-field",
                "decision": "rejected",
                "resulting_span": "حوزه",
                "rationale": "Context supports the existing rendering.",
            }],
            candidate_accepted=True,
        )

        issues = db.get_qa_issues("q3-job", 0)
        decisions = db.get_issue_decisions("q3-job", 0)
        assert issues[0]["status"] == "rejected"
        assert decisions[0]["candidate_accepted"] is True
        assert decisions[0]["payload"]["resulting_span"] == "حوزه"

        db.clear_chunk_qa_records("q3-job", 0)
        assert db.get_qa_issues("q3-job", 0) == []
        assert db.get_issue_decisions("q3-job", 0) == []
