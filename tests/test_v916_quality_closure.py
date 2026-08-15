from __future__ import annotations

from tarjomeh.core.pipeline import (
    _chunk_needs_review,
    audit_document_final_text,
    audit_translation_language,
)
from tarjomeh.core.term_notes import ensure_inline_proper_noun_originals
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    extract_identifiers,
    restore_source_identifiers,
    unexpected_latin_prose,
)


def _document(source: str, target: str) -> TranslatedDocument:
    return TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text=source,
        translated_text=target,
    )])


def test_complete_machine_identifiers_win_over_numeric_subspans() -> None:
    source = "Grant ARC-042-19-7001 uses catalogue code 410.7-xr52."
    candidate = (
        "\u06af\u0631\u0646\u062a ARC- \u06f0\u06f4\u06f2-\u06f1\u06f9-\u06f7\u06f0\u06f0\u06f1 "
        "\u0628\u0627 \u06a9\u062f \u06f4\u06f1\u06f0. \u06f7-xr\u06f5\u06f2."
    )

    repaired, report = restore_source_identifiers(source, candidate)

    assert "ARC-042-19-7001" in repaired
    assert "410.7-xr52" in repaired
    assert extract_identifiers(source) == extract_identifiers(repaired)
    assert report["repair_count"] == 2


def test_ordered_list_words_are_not_false_missing_numbers() -> None:
    source = "Two consequences follow: (1) institutional stability; (2) adaptation."
    target = (
        "\u062f\u0648 \u067e\u06cc\u0627\u0645\u062f \u062f\u0627\u0631\u062f: "
        "\u0627\u0648\u0644\u06cc \u062b\u0628\u0627\u062a \u0646\u0647\u0627\u062f\u06cc\u061b "
        "\u062f\u0648\u0645\u06cc \u0633\u0627\u0632\u06af\u0627\u0631\u06cc."
    )

    result = PostEditIntegrityGate().evaluate(source, target)

    assert "numbers_missing" not in {item.check_id for item in result.blocking}


def test_single_parenthesized_number_remains_protected() -> None:
    source = "The result appears in equation (1)."
    target = "\u0646\u062a\u06cc\u062c\u0647 \u062f\u0631 \u0645\u0639\u0627\u062f\u0644\u0647 \u0622\u0645\u062f\u0647 \u0627\u0633\u062a."

    result = PostEditIntegrityGate().evaluate(source, target)

    assert "numbers_missing" in {item.check_id for item in result.blocking}


def test_unrelated_digits_do_not_satisfy_ordered_list_markers() -> None:
    source = "The claims are (1) stability and (2) adaptation."
    target = (
        "\u062f\u0631 \u0633\u0627\u0644 \u06f1\u06f9\u06f9\u06f2 \u0627\u06cc\u0646 \u0627\u062f\u0639\u0627\u0647\u0627 \u0628\u0631\u0631\u0633\u06cc \u0634\u062f."
    )

    result = PostEditIntegrityGate().evaluate(source, target)

    assert "numbers_missing" in {item.check_id for item in result.blocking}


def test_unexplained_latin_prose_is_review_evidence_not_auto_deleted() -> None:
    source = "The institution changed rapidly (Taylor 1998)."
    target = (
        "\u0646\u0647\u0627\u062f \u0628\u0647\u200c\u0633\u0631\u0639\u062a \u062a\u063a\u06cc\u06cc\u0631 \u06a9\u0631\u062f "
        "(Taylor 1998) komment."
    )

    findings = unexpected_latin_prose(source, target)
    report = audit_translation_language(source, target)

    assert [item["token"] for item in findings] == ["komment"]
    assert report["review_required"]
    assert "komment" in target


def test_source_grounded_apparatus_and_authorized_original_are_allowed() -> None:
    source = "The Northern Policy Archive (NPA) cites Taylor (1998), p. xiv."
    target = (
        "\u0628\u0627\u06cc\u06af\u0627\u0646\u06cc \u0633\u06cc\u0627\u0633\u062a \u0634\u0645\u0627\u0644\u06cc "
        "(Northern Policy Archive) \u0628\u0647 Taylor (1998), p. xiv \u0627\u0631\u062c\u0627\u0639 \u0645\u06cc\u200c\u062f\u0647\u062f."
    )

    assert unexpected_latin_prose(
        source,
        target,
        allowed_originals=("Northern Policy Archive",),
    ) == []


def test_nested_source_entity_does_not_create_nested_parenthetical() -> None:
    document = _document(
        "The Northern Policy Archive published the record.",
        "\u0628\u0627\u06cc\u06af\u0627\u0646\u06cc \u0633\u06cc\u0627\u0633\u062a \u0634\u0645\u0627\u0644\u06cc \u0633\u0646\u062f \u0631\u0627 \u0645\u0646\u062a\u0634\u0631 \u06a9\u0631\u062f.",
    )

    report = ensure_inline_proper_noun_originals(
        document,
        {
            "Northern Policy Archive": "\u0628\u0627\u06cc\u06af\u0627\u0646\u06cc \u0633\u06cc\u0627\u0633\u062a \u0634\u0645\u0627\u0644\u06cc",
            "Policy Archive": "\u0628\u0627\u06cc\u06af\u0627\u0646\u06cc \u0633\u06cc\u0627\u0633\u062a",
        },
        PersianTypographer({"convert_numerals": False}),
        return_report=True,
    )

    text = document.paragraphs[0].translated_text
    assert "(Northern Policy Archive)" in text
    assert "(Policy Archive)" not in text
    assert report["overlap_suppressed_count"] == 1


def test_final_text_audit_distinguishes_language_and_identifier_findings() -> None:
    document = _document(
        "Record ARC-042-19-7001 was archived.",
        "\u0633\u0646\u062f komment \u0628\u0627\u06cc\u06af\u0627\u0646\u06cc \u0634\u062f.",
    )

    report = audit_document_final_text(document)

    assert report["review_required"]
    assert report["language_finding_count"] == 1
    assert report["unresolved_identifier_count"] == 1


def test_language_review_marks_chunk_for_review() -> None:
    class FakeDB:
        def get_chunk_events(self, _job_id: str, _chunk_index: int):
            return [
                {"event_type": "chunk_started", "payload": {}},
                {"event_type": "language_quality_review", "payload": {}},
            ]

    assert _chunk_needs_review(FakeDB(), "job", 0)
