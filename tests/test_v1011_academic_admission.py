from __future__ import annotations

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import _chunk_style_policy
from tarjomeh.core.prompts import (
    CRITIQUE_PROMPT,
    GLOSSARY_EXTRACT_PROMPT,
    INCREMENTAL_NER_PROMPT,
    TRANSLATE_CHUNK_PROMPT,
)
from tarjomeh.memory.bilingual_summary import BilingualSummary
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.memory.proper_nouns import (
    automatic_terminology_risk_reasons,
    has_minimal_automatic_term_evidence,
    observed_bilingual_target,
)


def test_automatic_terms_reject_context_expansion_and_inflected_clauses() -> None:
    expanded = automatic_terminology_risk_reasons(
        "institutional change",
        "\u062a\u063a\u06cc\u06cc\u0631 \u0646\u0647\u0627\u062f\u06cc \u062f\u0648\u0644\u062a \u0645\u062f\u0631\u0646",
    )
    clausal = automatic_terminology_risk_reasons(
        "institutional change",
        "\u0646\u0647\u0627\u062f\u0647\u0627 \u062a\u063a\u06cc\u06cc\u0631 \u0645\u06cc\u200c\u06a9\u0646\u0646\u062f",
    )

    assert "target_scope_wider_than_source" in expanded
    assert "inflected_or_clausal_target" in clausal
    assert not automatic_terminology_risk_reasons(
        "political field",
        "\u0645\u06cc\u062f\u0627\u0646 \u0633\u06cc\u0627\u0633\u06cc",
    )
    assert not automatic_terminology_risk_reasons(
        "mediation",
        "\u0645\u06cc\u0627\u0646\u062c\u06cc\u200c\u06af\u0631\u06cc",
    )


def test_automatic_term_requires_exact_minimal_span_evidence() -> None:
    valid = {
        "term": "institutional order",
        "suggested_persian": "\u0646\u0638\u0645 \u0646\u0647\u0627\u062f\u06cc",
        "exact_source_span": "institutional order",
        "exact_target_span": "\u0646\u0638\u0645 \u0646\u0647\u0627\u062f\u06cc",
        "context_independent": True,
    }
    expanded = {
        **valid,
        "suggested_persian": "\u0646\u0638\u0645 \u0646\u0647\u0627\u062f\u06cc \u062f\u0648\u0644\u062a \u0645\u062f\u0631\u0646",
        "exact_target_span": "\u0646\u0638\u0645 \u0646\u0647\u0627\u062f\u06cc \u062f\u0648\u0644\u062a \u0645\u062f\u0631\u0646",
    }

    assert has_minimal_automatic_term_evidence(
        valid,
        translation="\u0627\u06cc\u0646 \u0646\u0638\u0645 \u0646\u0647\u0627\u062f\u06cc \u067e\u0627\u06cc\u062f\u0627\u0631 \u0627\u0633\u062a.",
    )
    assert not has_minimal_automatic_term_evidence(expanded)
    assert not has_minimal_automatic_term_evidence({
        **valid,
        "context_independent": False,
    })


def test_compound_given_name_anchor_keeps_the_full_persian_name() -> None:
    translation = "\u0645\u0631\u06cc \u0640 \u0622\u0646 \u0627\u0633\u0645\u06cc\u062a (Mary-Anne Smith) \u0633\u062e\u0646 \u06af\u0641\u062a."

    assert observed_bilingual_target(
        translation,
        "Mary-Anne Smith",
        category="person",
    ) == "\u0645\u0631\u06cc \u0640 \u0622\u0646 \u0627\u0633\u0645\u06cc\u062a"


class _ParagraphStyleDB:
    def get_chunk_review_reasons(self, job_id: str, chunk_index: int):
        return []

    def get_chunk_events(self, job_id: str, chunk_index: int):
        return [
            {"event_type": "chunk_started", "payload": {}},
            {
                "event_type": "critique_completed",
                "payload": {
                    "valid": True,
                    "blocking_issue_count": 0,
                    "scores": {
                        "accuracy": 9.5,
                        "fluency": 9.2,
                        "terminology": 9.4,
                        "register": 9.3,
                        "average": 9.35,
                    },
                    "issue_details": [{
                        "category": "fluency",
                        "severity": "minor",
                        "source_segment_id": "p1:s1",
                    }],
                },
            },
        ]


def test_style_memory_excludes_only_the_grounded_problem_paragraph() -> None:
    policy = _chunk_style_policy(_ParagraphStyleDB(), "job", 0)
    assert policy["approved"]
    assert policy["excluded_paragraphs"] == [0]

    manager = MemoryManager(TarjomehConfig())
    chunk = Chunk(
        0,
        ("First body paragraph with sufficient structural context. " * 4).strip()
        + "\n\n"
        + ("Second body paragraph with sufficient structural context. " * 4).strip(),
        "Chapter",
        "",
        metadata={
            "style_body_paragraphs": [0, 1],
            "style_eligible": True,
            "structural_roles": ["body", "body"],
        },
    )
    first = "\u0627\u06cc\u0646 \u0628\u0646\u062f \u0627\u0648\u0644 \u0647\u0646\u0648\u0632 \u0628\u0647 \u0628\u0627\u0632\u0628\u06cc\u0646\u06cc \u0646\u06cc\u0627\u0632 \u062f\u0627\u0631\u062f."
    second = (
        "\u0627\u06cc\u0646 \u0628\u0646\u062f \u062f\u0648\u0645 \u0627\u0633\u062a\u062f\u0644\u0627\u0644 \u0631\u0627 \u0628\u0627 \u0646\u062b\u0631\u06cc \u0631\u0648\u0634\u0646 \u0648 \u062f\u0642\u06cc\u0642 \u062f\u0646\u0628\u0627\u0644 \u0645\u06cc\u200c\u06a9\u0646\u062f. "
        "\u0647\u0645\u0686\u0646\u06cc\u0646 \u067e\u06cc\u0648\u0646\u062f \u0645\u06cc\u0627\u0646 \u0645\u0641\u0627\u0647\u06cc\u0645 \u0631\u0627 \u0628\u0647\u200c\u0631\u0648\u0634\u0646\u06cc \u0646\u0634\u0627\u0646 \u0645\u06cc\u200c\u062f\u0647\u062f. "
        "\u0646\u062a\u06cc\u062c\u0647 \u0628\u0627 \u0645\u0642\u062f\u0645\u0627\u062a \u0627\u0633\u062a\u062f\u0644\u0627\u0644 \u0646\u06cc\u0632 \u0633\u0627\u0632\u06af\u0627\u0631 \u0627\u0633\u062a."
    )
    result = manager.update_after_translation(
        chunk,
        first + "\n\n" + second,
        quality_approved=True,
        style_approved=True,
        long_term_reliable=True,
        style_excluded_paragraphs=[0],
    )

    assert result["continuity_retained"]
    assert result["style_sample_added"]
    assert manager.style_samples == [second]
    assert len(manager.short_term.get_entries()) == 1


def test_summary_storage_removes_directional_protocol_controls() -> None:
    summary = BilingualSummary()
    summary.update(
        "## English Summary\nArgument.\n"
        "## \u062e\u0644\u0627\u0635\u0647 \u0641\u0627\u0631\u0633\u06cc\n"
        "\u202b\u0627\u06cc\u0646 \u062e\u0644\u0627\u0635\u0647 \u062f\u0642\u06cc\u0642 \u0627\u0633\u062a.\u202c"
    )

    assert summary.persian_summary == "\u0627\u06cc\u0646 \u062e\u0644\u0627\u0635\u0647 \u062f\u0642\u06cc\u0642 \u0627\u0633\u062a."


def test_prompts_make_fluency_subordinate_to_full_source_fidelity() -> None:
    assert "Accuracy and completeness outrank fluency" in TRANSLATE_CHUNK_PROMPT
    assert "must never simplify" in TRANSLATE_CHUNK_PROMPT
    assert "generalize, reinterpret, or weaken" in TRANSLATE_CHUNK_PROMPT
    assert "predicate completeness" in CRITIQUE_PROMPT
    assert "context_independent" in GLOSSARY_EXTRACT_PROMPT
    assert "exact_target_span" in INCREMENTAL_NER_PROMPT
