import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tarjomeh.core.pipeline import (
    _canonical_chunk_paragraph_identity,
    _high_confidence_minor_issues,
    _unresolved_grounded_memory_issues,
)
from tarjomeh.glossary.book_recommendations import (
    format_book_recommendations,
    recommendation_evidence,
    recommendation_presence,
    recommendations_from_context,
    resolve_book_recommendations,
)
from tarjomeh.glossary.book_review import (
    _sha,
    check_reviewed_book_terms,
    resolve_reviewed_book_terms,
)
from tarjomeh.quality.critique import CritiqueResult, TranslationCritique
from tarjomeh.quality.grounding import source_segments
from tarjomeh.quality.refiner import TranslationRefiner


@pytest.fixture
def data():
    parts = ["The apparatus acts.", "Another apparatus remains."]
    source = "\n\n".join(parts)
    metadata = {
        "paragraph_indices": [7, 8],
        "structural_roles": ["body", "body"],
        "source_paragraph_hashes": [_sha(value) for value in parts],
        "source_paragraph_spans": [[0, len(parts[0])], [len(parts[0]) + 2, len(source)]],
    }
    term = {
        "source": "apparatus",
        "target": "دستگاه",
        "status": "approved",
        "scope_mode": "book_wide_recommended",
        "source_evidence_sha256": _sha(parts[0]),
        "source_count": 9999,
    }
    review = {"review_version": 3, "phase": "started", "source_sha256": "book", "proposals": [term]}
    body = {
        "source_sha256": "book",
        "entries": [
            {
                "chunk_index": 0,
                "local_paragraph": local,
                "paragraph_index": global_index,
                "paragraph_sha256": _sha(parts[local]),
                "chapter_position": 2,
            }
            for local, global_index in enumerate([7, 8])
        ],
    }
    return source, metadata, review, body


def test_required_here_is_not_whole_book_enforcement(data):
    source, metadata, review, body = data
    mandatory, uncertain = resolve_reviewed_book_terms(source, metadata, 2, review["proposals"])
    assert not uncertain and len(mandatory) == 1 and mandatory[0]["paragraph_index"] == 0
    target = "دستگاه عمل می‌کند.\n\nسازوکار دیگری باقی می‌ماند."
    report, uncertain = check_reviewed_book_terms(target, source, metadata, mandatory)
    assert report.compliant and not uncertain
    scope = resolve_book_recommendations(source, metadata, 0, review, body)
    assert not scope["deferred"] and len(scope["recommendations"]) == 1
    advice = scope["recommendations"]
    assert advice[0]["global_paragraph_index"] == 8
    assert len(advice[0]["occurrences"]) == 1  # not source_count=9999
    assert (
        recommendations_from_context(format_book_recommendations(advice))[0]["target"] == "دستگاه"
    )
    canonical, identity = _canonical_chunk_paragraph_identity(
        SimpleNamespace(text=source, metadata=metadata),
        target,
    )
    result = recommendation_presence(source, canonical, metadata, advice, identity)
    assert result["occurrence_counts"]["preferred_wording_not_found"] == 1
    assert result["locations"][0]["target_excerpt"] == "سازوکار دیگری باقی می‌ماند."
    assert "issues" not in result and "review" not in result
    identity["canonical_target_hash"] = "wrong"
    assert (
        recommendation_presence(source, canonical, metadata, advice, identity)["occurrence_counts"][
            "alignment_unavailable_usage_unknown"
        ]
        == 1
    )


def test_long_paragraph_quote_contains_the_actual_matched_term(data):
    source, metadata, review, body = data
    first, second = source.split("\n\n")
    second = ("Neutral discussion. " * 60) + second
    source = first + "\n\n" + second
    metadata["source_paragraph_hashes"][1] = _sha(second)
    metadata["source_paragraph_spans"][1] = [len(first) + 2, len(source)]
    body["entries"][1]["paragraph_sha256"] = _sha(second)
    advice = resolve_book_recommendations(source, metadata, 0, review, body)["recommendations"]
    assert "apparatus" in advice[0]["source_quote"]
    assert advice[0]["occurrences"][0]["start"] > 500


@pytest.mark.parametrize(
    "change,expected",
    [
        ("skip", 0),
        ("pending", 0),
        ("v2", 0),
        ("heading", 0),
        ("stale", 0),
        ("two_senses", 0),
        ("curated", 0),
        ("keep_original", 0),
        ("chapter", 0),
    ],
)
def test_advice_has_no_authority_when_scope_is_unproved(data, change, expected):
    source, metadata, review, body = data
    terms = []
    if change == "skip":
        review["phase"] = "skipped"
    if change == "pending":
        review["proposals"][0]["status"] = "proposed"
    if change == "v2":
        review["review_version"] = 2
    if change == "heading":
        body["entries"] = body["entries"][:1]
    if change == "stale":
        body["source_sha256"] = "other book"
    if change == "two_senses":
        review["proposals"].append(dict(review["proposals"][0], sense_id="other"))
    if change == "curated":
        terms = [{"source": "apparatus", "target": "چیز دیگر"}]
    if change == "keep_original":
        review["proposals"][0]["keep_original"] = True
    if change == "chapter":
        review["proposals"][0]["chapter_positions"] = [3]
    assert (
        len(
            resolve_book_recommendations(source, metadata, 0, review, body, terms)[
                "recommendations"
            ]
        )
        == expected
    )


@pytest.mark.parametrize("optional", [None, "bad", [{}], [{"recommendation_id": "unknown"}]])
def test_optional_observations_never_invalidate_core_or_create_grounded_issues(optional):
    raw = json.dumps(
        {
            "scores": {name: 9 for name in ("accuracy", "fluency", "terminology", "register")},
            "issues": [],
            "advisory_term_observations": optional,
        }
    )
    result = TranslationCritique._parse_response(raw)
    assert result.valid and result.average == 9 and not result.validation_errors
    assert not result.issue_details and not result.issues
    assert not _unresolved_grounded_memory_issues(result)
    assert not _high_confidence_minor_issues(result)
    assert not CritiqueResult.from_dict(result.to_dict()).advisory_term_observations


def test_exact_advisory_identity_and_normal_accuracy_issue_coexist(data):
    source, metadata, review, body = data
    advice = resolve_book_recommendations(source, metadata, 0, review, body)["recommendations"]
    raw = json.dumps(
        {
            "scores": {name: 9 for name in ("accuracy", "fluency", "terminology", "register")},
            "issues": [
                {
                    "category": "accuracy",
                    "severity": "major",
                    "confidence": 0.95,
                    "source_segment_id": "p2:s1",
                    "source_quote": "Another apparatus remains.",
                    "current_persian_quote": "دستگاه دیگری ناپدید می‌شود.",
                    "suggested_correction": "دستگاه دیگری باقی می‌ماند.",
                    "rationale": "The source asserts persistence, not disappearance.",
                }
            ],
            "advisory_term_observations": [
                {
                    "recommendation_id": advice[0]["recommendation_id"],
                    "source_paragraph_hash": advice[0]["source_paragraph_hash"],
                    "note": "Optional alternative.",
                }
            ],
        }
    )
    result = TranslationCritique._parse_response(
        raw,
        source,
        "دستگاه عمل می‌کند.\n\nدستگاه دیگری ناپدید می‌شود.",
        advisory_recommendations=advice,
    )
    assert result.valid and len(result.advisory_term_observations) == 1
    assert len(result.issue_details) == 1 and _unresolved_grounded_memory_issues(result)
    assert len(CritiqueResult.from_dict(result.to_dict()).advisory_term_observations) == 1


def test_report_drops_old_generation_and_does_not_guess_alignment(data):
    source, metadata, review, body = data
    scope = resolve_book_recommendations(source, metadata, 0, review, body)
    event = {"chunk_index": 0, "event_type": "book_term_recommendations_resolved", "payload": scope}
    chunks = [
        {
            "chunk_index": 0,
            "status": "completed",
            "text": source,
            "translation": "متن",
            "metadata": metadata,
        }
    ]
    result = recommendation_evidence([event], chunks, {})
    assert result["occurrence_counts"]["alignment_unavailable_usage_unknown"] == 1
    assert not recommendation_evidence(
        [event, {"chunk_index": 0, "event_type": "chunk_started"}], chunks, {}
    )["locations"]


@pytest.mark.asyncio
async def test_critic_and_existing_refiner_receive_advice_without_new_advice_call(data):
    source, metadata, review, body = data
    advice = resolve_book_recommendations(source, metadata, 0, review, body)["recommendations"]
    context = format_book_recommendations(advice)
    response = json.dumps(
        {
            "scores": {name: 9 for name in ("accuracy", "fluency", "terminology", "register")},
            "issues": [],
            "source_coverage": {
                "complete": True,
                "checked_source_segment_ids": [
                    item["segment_id"] for item in source_segments(source)
                ],
                "uncovered_source_segment_ids": [],
            },
            "advisory_term_observations": "bad optional field",
        }
    )
    llm = SimpleNamespace(chat=AsyncMock(return_value=response))
    result = await TranslationCritique(llm).critique(source, "ترجمه", terminology=context)
    assert result.valid and result.attempts == 1 and llm.chat.await_count == 1
    assert context in llm.chat.await_args.args[0]
    refiner = TranslationRefiner(llm)
    preserved = await refiner.refine_with_decision(source, "ترجمه", result, terminology=context)
    assert preserved.decision == "preserved" and llm.chat.await_count == 1
    # Only an already justified ordinary issue can reach the existing refiner call.
    result.issues = ["A source-based error"]
    llm.chat.return_value = json.dumps(
        {
            "decision": "preserved",
            "translation": "ترجمه",
            "rationale": "The original is accurate.",
            "issue_decisions": [],
        }
    )
    await refiner.refine_with_decision(source, "ترجمه", result, terminology=context)
    assert context in llm.chat.await_args.args[0]
