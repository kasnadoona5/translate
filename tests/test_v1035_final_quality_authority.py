from __future__ import annotations

from types import SimpleNamespace

from tarjomeh.core.pipeline import (
    TranslationPipeline,
    _research_context_for_memory,
    _salvage_local_refinement_edits,
)
from tarjomeh.memory.manager import (
    _clean_style_sample,
    _source_style_contradiction,
    _style_record_is_authoritative,
)
from tarjomeh.memory.proper_nouns import ProperNouns
from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    restore_source_note_markers,
)


class _AcceptedIntegrity:
    accepted = True

    def to_dict(self) -> dict[str, object]:
        return {"accepted": True, "blocking_count": 0, "findings": []}


class _AcceptingGate:
    def evaluate(self, source: str, candidate: str, **kwargs: object):
        return _AcceptedIntegrity()


def _salvage(
    previous: str,
    proposed: str,
    issues: list[dict[str, str]],
    decisions: list[dict[str, str]],
):
    return _salvage_local_refinement_edits(
        source="The source preserves every proposition.",
        previous=previous,
        proposed=proposed,
        issue_details=issues,
        issue_decisions=decisions,
        integrity_gate=_AcceptingGate(),  # type: ignore[arg-type]
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )


def test_equivalent_overlapping_review_issues_share_one_safe_edit() -> None:
    previous = "این نظریه دولت را می‌سازز و توضیح می‌دهد."
    proposed = "این نظریه دولت را می‌سازد و توضیح می‌دهد."
    current = "می‌سازز"
    result = "می‌سازد"
    issues = [
        {
            "issue_id": issue_id,
            "source_segment_id": "p1:s1",
            "source_quote": "constructs",
            "current_persian_quote": current,
        }
        for issue_id in ("typo-a", "typo-b")
    ]
    decisions = [
        {
            "issue_id": issue_id,
            "decision": "accepted",
            "resulting_span": result,
        }
        for issue_id in ("typo-a", "typo-b")
    ]

    final, enriched, report = _salvage(
        previous, proposed, issues, decisions
    )

    assert final == proposed
    assert report["committed_count"] == 1
    assert enriched[0]["commit_status"] == "committed_local"
    assert enriched[1]["commit_status"] == "committed_local"
    assert enriched[1]["commit_reason"] == "equivalent_duplicate_satisfied"


def test_conflicting_overlapping_review_edits_still_fail_closed() -> None:
    previous = "این نظریه دولت را می‌سازز و توضیح می‌دهد."
    issues = [
        {
            "issue_id": "a",
            "source_segment_id": "p1:s1",
            "current_persian_quote": "می‌سازز",
        },
        {
            "issue_id": "b",
            "source_segment_id": "p1:s1",
            "current_persian_quote": "می‌سازز",
        },
    ]
    decisions = [
        {"issue_id": "a", "decision": "accepted", "resulting_span": "می‌سازد"},
        {"issue_id": "b", "decision": "accepted", "resulting_span": "می‌سازند"},
    ]

    final, enriched, report = _salvage(
        previous,
        previous.replace("می‌سازز", "می‌سازد"),
        issues,
        decisions,
    )

    assert final == previous
    assert report["committed_count"] == 0
    assert {item["commit_reason"] for item in enriched} == {
        "overlapping_local_span"
    }


def test_minimal_unique_diff_can_recover_one_typo_from_oversized_span() -> None:
    prefix = "این بند رابطه میان دولت و جامعه را با دقت بررسی می‌کند و " * 8
    previous = prefix + "در پایان نظریه را می‌سازز."
    proposed = prefix + "در پایان نظریه را می‌سازد."
    issue = {
        "issue_id": "typo",
        "source_segment_id": "p1:s1",
        "source_quote": "constructs the theory",
        "current_persian_quote": previous,
    }
    decision = {
        "issue_id": "typo",
        "decision": "accepted",
        "resulting_span": proposed,
    }

    final, enriched, report = _salvage(
        previous, proposed, [issue], [decision]
    )

    assert final == proposed
    assert report["committed_count"] == 1
    assert len(enriched[0]["resulting_span"]) < len(proposed)


def test_unique_plain_duplicate_note_is_removed_but_ambiguous_surplus_is_not() -> None:
    source = "A source sentence.1"
    repaired, report = restore_source_note_markers(
        source,
        "یک جملهٔ ترجمه‌شده.¹ ۱",
    )
    assert repaired == "یک جملهٔ ترجمه‌شده.¹"
    assert report["surplus"] == []
    assert report["repairs"][-1]["type"] == (
        "unique_plain_duplicate_note_marker_removed"
    )

    ambiguous, ambiguous_report = restore_source_note_markers(
        source,
        "یک جملهٔ ترجمه‌شده.¹ [1]",
    )
    assert ambiguous == "یک جملهٔ ترجمه‌شده.¹ [1]"
    assert ambiguous_report["surplus"] == ["1"]


def test_integrity_blocks_a_new_surplus_note_marker() -> None:
    result = PostEditIntegrityGate().evaluate(
        "A source sentence.1",
        "یک جمله.¹ ۱",
        previous="یک جمله.¹",
    )
    assert not result.accepted
    assert any(
        finding.check_id == "note_markers_surplus"
        for finding in result.findings
    )


def test_style_authority_requires_clean_surface_and_all_dimensions_at_nine() -> None:
    clean = "این تحلیل رابطهٔ میان نهادها را به‌دقت بررسی می‌کند."
    record = {
        "text": clean,
        "representative": True,
        "quality_score": 90.0,
        "final_scores": {
            "accuracy": 9.4,
            "fluency": 8.9,
            "terminology": 9.3,
            "register": 9.2,
        },
    }
    assert not _style_record_is_authoritative(record)
    record["final_scores"]["fluency"] = 9.0
    assert _style_record_is_authoritative(record)
    record["text"] = "این تحلیل دولت‌‌جامعه را بررسی می‌کند."
    assert _clean_style_sample(record["text"]) == ""
    assert not _style_record_is_authoritative(record)


def test_source_count_contradiction_cannot_teach_style() -> None:
    findings = _source_style_contradiction(
        "This chapter addresses two issues.",
        "این فصل به سه مسئله می‌پردازد.",
    )
    assert "announced_count_lexical_mismatch" in findings


def test_only_evidence_bound_unambiguous_research_enters_prompt() -> None:
    context = _research_context_for_memory({
        "status": "completed",
        "book_context": "Verified book context.",
        "terms": [
            {
                "source": "safe term",
                "target": "اصطلاح امن",
                "status": "suggested",
                "identity_supported": True,
                "term_supported": True,
            },
            {
                "source": "weak term",
                "target": "حدس ضعیف",
                "status": "suggested",
                "identity_supported": True,
                "term_supported": False,
            },
            {
                "source": "ambiguous term",
                "target": "الف / ب",
                "status": "suggested",
                "identity_supported": True,
                "term_supported": True,
            },
        ],
    })
    assert "safe term -> اصطلاح امن" in context
    assert "weak term" not in context
    assert "ambiguous term" not in context


def test_unaligned_reviewed_correction_is_migrated_to_context_deferred() -> None:
    memory = ProperNouns()
    memory.deserialize({
        "nouns": {"Clays Ltd": "بریتانیا"},
        "metadata": {"Clays Ltd": {"category": "organization"}},
        "provenance": {
            "Clays Ltd": {
                "origin": "accepted_correction",
                "authority": 80,
                "context_independent": True,
            }
        },
    })
    assert memory.is_context_deferred("Clays Ltd")
    assert "Clays Ltd" not in memory.get_context(
        source_text="Printed by Clays Ltd in the United Kingdom."
    )


def test_exact_final_repair_is_conditional_and_reruns_authority_gates() -> None:
    source = __import__("inspect").getsource(
        TranslationPipeline._translate_single_chunk
    )
    assert '"exact_final_quality_repair"' in source
    assert "_salvage_local_refinement_edits(" in source
    assert 'stage="exact_final_quality_repair"' in source
    assert '"exact_final_quality_repair_validation"' in source
    assert "_actionable_structure_findings(" in source
    assert "_candidate_regression_details(" in source
    accepted_branch = source.split("if final_repair_accepted:", 1)[1]
    assert accepted_branch.index('"critique_completed"') < accepted_branch.index(
        "translation = repaired_candidate"
    )
    assert accepted_branch.index("language_quality = audit_translation_language(") < (
        accepted_branch.index("canonical_candidate_hash = (")
    )
