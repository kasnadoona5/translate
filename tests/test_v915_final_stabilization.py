from __future__ import annotations

from tarjomeh.core.pipeline import (
    _chunk_style_approved,
    _critique_requires_refinement,
    restore_document_source_identifiers,
)
from tarjomeh.core.term_notes import normalize_adjacent_original_citations
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.quality.back_translator import BackTranslator
from tarjomeh.quality.critique import CritiqueResult
from tarjomeh.quality.integrity import extract_identifiers


def test_prose_followed_by_footnote_is_not_a_machine_identifier() -> None:
    assert not extract_identifiers("The account concerns states.2")
    assert extract_identifiers("See classification A.2 for details.")


def test_document_identifier_reconciliation_runs_on_final_text() -> None:
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text="Visit politybooks.com; grant RES-051-27-0303.",
        translated_text=(
            "Visit politybooks. com; grant RES-\u06f0\u06f5\u06f1 - "
            "\u06f2\u06f7 - \u06f0\u06f3\u06f0\u06f3."
        ),
    )])

    report = restore_document_source_identifiers(document)

    text = document.paragraphs[0].translated_text
    assert "politybooks.com" in text
    assert "RES-051-27-0303" in text
    assert report["repair_count"] == 2
    assert report["unresolved_count"] == 0


def test_first_seen_entities_defer_to_final_term_anchoring() -> None:
    result = BackTranslator(object()).compare(
        "Shmuel Eisenstadt developed this account.",
        "A scholar developed this account.",
        "\u0634\u0645\u0648\u0626\u0644 \u0622\u06cc\u0632\u0646\u0634\u062a\u0627\u062a \u0627\u06cc\u0646 \u062a\u0628\u06cc\u06cc\u0646 \u0631\u0627 \u0628\u0633\u0637 \u062f\u0627\u062f.",
        entity_aliases={},
    )

    assert "named_entities_missing" not in result.diagnostics["risk_flags"]
    assert result.diagnostics["entities_deferred_until_term_anchoring"]


def test_known_entity_without_its_alias_still_flags_back_translation() -> None:
    result = BackTranslator(object()).compare(
        "Shmuel Eisenstadt developed this account.",
        "A scholar developed this account.",
        "\u0627\u06cc\u0646 \u062a\u0628\u06cc\u06cc\u0646 \u0628\u0633\u0637 \u06cc\u0627\u0641\u062a.",
        entity_aliases={
            "Shmuel Eisenstadt": [
                "\u0634\u0645\u0648\u0626\u0644 \u0622\u06cc\u0632\u0646\u0634\u062a\u0627\u062a"
            ]
        },
    )

    assert "named_entities_missing" in result.diagnostics["risk_flags"]


class _StyleDB:
    def get_chunk_events(self, _job: str, _chunk: int):
        return [
            {"event_type": "chunk_started", "payload": {}},
            {
                "event_type": "critique_completed",
                "payload": {
                    "valid": True,
                    "blocking_issue_count": 0,
                    "scores": {
                        "accuracy": 9,
                        "fluency": 8,
                        "terminology": 9,
                        "register": 10,
                        "average": 9,
                    },
                },
            },
        ]


def test_one_sub_nine_dimension_cannot_seed_style_memory() -> None:
    assert not _chunk_style_approved(_StyleDB(), "job", 0)


def test_long_grounded_fluency_issue_gets_one_bounded_review_pass() -> None:
    critique = CritiqueResult(
        accuracy=9,
        fluency=8,
        terminology=9,
        register=9,
        average=8.75,
        issues=["[MINOR/fluency] incomplete coordination"],
        issue_details=[{
            "issue_id": "mqm-sentence-completeness",
            "category": "fluency",
            "severity": "minor",
            "confidence": 0.70,
            "source_quote": (
                "The research on state formation and the three-volume study "
                "of government are also exemplary instances of this approach."
            ),
            "current_persian_quote": (
                "\u067e\u0698\u0648\u0647\u0634 \u062f\u0631\u0628\u0627\u0631\u0647 \u0634\u06a9\u0644\u200c\u06af\u06cc\u0631\u06cc \u062f\u0648\u0644\u062a \u0648 \u0645\u0637\u0627\u0644\u0639\u0647 "
                "\u0633\u0647\u200c\u062c\u0644\u062f\u06cc \u062f\u0631\u0628\u0627\u0631\u0647 \u062a\u0627\u0631\u06cc\u062e \u062d\u06a9\u0648\u0645\u062a."
            ),
            "suggested_correction": (
                "\u067e\u0698\u0648\u0647\u0634 \u062f\u0631\u0628\u0627\u0631\u0647 \u0634\u06a9\u0644\u200c\u06af\u06cc\u0631\u06cc \u062f\u0648\u0644\u062a \u0648 \u0645\u0637\u0627\u0644\u0639\u0647 "
                "\u0633\u0647\u200c\u062c\u0644\u062f\u06cc \u062a\u0627\u0631\u06cc\u062e \u062d\u06a9\u0648\u0645\u062a \u0646\u06cc\u0632 \u0627\u0632 \u0646\u0645\u0648\u0646\u0647\u200c\u0647\u0627\u06cc \u0628\u0627\u0631\u0632\u0646\u062f."
            ),
        }],
    )

    assert _critique_requires_refinement(critique, 9.0)


def test_short_fluency_preference_remains_advisory() -> None:
    critique = CritiqueResult(
        accuracy=9,
        fluency=8,
        terminology=9,
        register=9,
        average=8.75,
        issues=["[MINOR/fluency] preference"],
        issue_details=[{
            "issue_id": "mqm-short-preference",
            "category": "fluency",
            "severity": "minor",
            "confidence": 0.99,
            "source_quote": "pay attention",
            "current_persian_quote": "\u062a\u0648\u062c\u0647 \u06a9\u0646\u06cc\u062f",
            "suggested_correction": "\u062f\u0642\u062a \u06a9\u0646\u06cc\u062f",
        }],
    )

    assert not _critique_requires_refinement(critique, 9.0)


def test_citation_year_list_removes_english_conjunction_only() -> None:
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text="The studies appeared in 1986, 1996, 2012a, and 2012b.",
        translated_text="\u0627\u06cc\u0646 \u0622\u062b\u0627\u0631 (1986, 1996, 2012a, and 2012b) \u0645\u0646\u062a\u0634\u0631 \u0634\u062f\u0646\u062f.",
    )])

    report = normalize_adjacent_original_citations(document, {})

    assert "and" not in document.paragraphs[0].translated_text
    assert "(1986, 1996, 2012a, 2012b)" in document.paragraphs[0].translated_text
    assert report["normalized_count"] == 1
