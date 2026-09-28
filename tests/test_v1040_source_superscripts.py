from tarjomeh.exporters.docx_exporter import source_superscript_spans
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.core.pipeline import audit_document_final_text
from tarjomeh.parsers.base import Paragraph
from tarjomeh.parsers.pdf_parser import (
    _joined_superscript_markers,
    _markers_in_text,
    _merge_continuation_paragraphs,
)


def test_split_marker_only_follows_its_source_fragment():
    marker = {"text": "7", "relative_position": 0.5}
    assert _markers_in_text("An argument.", [marker]) == []
    kept = _markers_in_text("A note.7", [marker])
    assert [(item["text"], item["offset"], item["relative_position"])
            for item in kept] == [("7", 7, 0.9375)]


def test_continuation_preserves_unique_right_hand_note():
    left = Paragraph(
        "This idea concerns", metadata={"page": 1, "bbox": (0, 75, 10, 90),
                                        "page_height": 100}
    )
    right = Paragraph(
        "the state.7", metadata={"page": 2, "bbox": (0, 5, 10, 20),
                                  "page_height": 100,
                                  "superscript_markers": [{"text": "7"}]}
    )
    assert _joined_superscript_markers(left, right, " ")
    result = _merge_continuation_paragraphs([left, right])
    assert len(result) == 1
    assert result[0].metadata["superscript_markers"][0]["text"] == "7"


def test_export_never_chooses_chapter_digit_as_note():
    metadata = {"superscript_markers": [{"text": "3", "relative_position": 0.5}]}
    assert source_superscript_spans("(فصل ۳)", metadata, "The argument.3") == []
    assert source_superscript_spans("نکته.۳", metadata, "The argument.3") == [(5, 6)]
    assert source_superscript_spans("نکته ۳ و فصل ۳", metadata, "The argument.3") == []


def test_ambiguous_note_position_is_listed_for_review():
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0, source_text="The argument.3", translated_text="\u0646\u06a9\u062a\u0647 \u06f3 \u0648 \u0641\u0635\u0644 \u06f3",
        metadata={"superscript_markers": [{"text": "3"}]},
    )])
    report = audit_document_final_text(document)
    assert any(item["check_id"] == "source_superscript_alignment_uncertain"
               and item["disposition"] == "REVIEW"
               for item in report["report_only_findings"])
