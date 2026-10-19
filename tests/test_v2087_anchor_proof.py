"""Contiguous, source-bound anchors preserve canonical Persian."""

import copy

import pytest

from tarjomeh.core.anchor_spans import anchor_spans
from tarjomeh.core.render_identity import (
    RenderChangeLedger,
    _proof_anchor,
    compare_docx_to_rendered,
)
from tarjomeh.core.term_notes import ensure_inline_proper_noun_originals
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.docx_exporter import DocxExporter
from tarjomeh.persian.typography import PersianTypographer

TARGET = "\u062a\u062d\u0648\u0644 \u0641\u0646\u06cc"
INFLECTED = "\u062a\u062d\u0648\u0644\u06cc \u0641\u0646\u06cc"


def document(text, source="A technical change matters."):
    return TranslatedDocument(
        title="Synthetic",
        paragraphs=[TranslatedParagraph(index=0, source_text=source, translated_text=text)],
    )


@pytest.mark.parametrize("mode", ["target_only", "inline", "side_by_side"])
def test_inflected_anchor_has_proof_and_real_docx_readback(tmp_path, mode):
    before = INFLECTED + "."
    doc = document(before)
    ledger = RenderChangeLedger(doc)
    ledger.set_authorized({"technical change": TARGET})
    ledger.step(
        "inline_original_anchor",
        ensure_inline_proper_noun_originals,
        {"technical change": TARGET},
        PersianTypographer(),
        {"technical change": "theory"},
    )
    assert ledger.canonical == [before]
    assert doc.paragraphs[0].translated_text == INFLECTED + " (technical change)."
    audit = ledger.audit()
    assert audit["passed"] and audit["changes"][0]["replay_verified"]
    assert audit["changes"][0]["anchor_evidence"][0]["kind"] == "internal_indefinite_yeh"
    path = tmp_path / "review.docx"
    DocxExporter().export(doc, path, bilingual_mode=mode)
    assert compare_docx_to_rendered(doc, path, mode)["passed"]


@pytest.mark.parametrize(
    "text", [TARGET, INFLECTED, TARGET + "\u0650", "\u00ab" + INFLECTED + "\u00bb"]
)
def test_exact_inflection_marks_and_quotes_are_proven(text):
    assert not _proof_anchor(
        "A technical change matters.",
        text,
        text + " (technical change)",
        {"authorized": {"technical change": TARGET}},
    )


@pytest.mark.parametrize(
    "text",
    [
        "\u062a\u062d\u0648\u0644 \u0645\u0647\u0645 \u0641\u0646\u06cc",
        "\u062a\u062d\u0648\u0644. \u0641\u0646\u06cc",
        "\u062a\u062d\u0648\u0644\n\u0641\u0646\u06cc",
        "\u0641\u0646\u06cc \u062a\u062d\u0648\u0644",
        "\u0628\u06cc\u062a\u062d\u0648\u0644 \u0641\u0646\u06cc",
    ],
)
def test_unrelated_or_noncontiguous_targets_never_anchor(text):
    assert not anchor_spans(text, TARGET)
    assert _proof_anchor(
        "A technical change matters.",
        text,
        text + " (technical change)",
        {"authorized": {"technical change": TARGET}},
    )


@pytest.mark.parametrize(
    "source,text",
    [
        ("A technical change follows another technical change.", INFLECTED),
        ("A technical change matters.", INFLECTED + "; " + INFLECTED),
    ],
)
def test_ambiguous_inflection_is_deferred_not_inserted(source, text):
    doc = document(text, source)
    report = ensure_inline_proper_noun_originals(
        doc, {"technical change": TARGET}, PersianTypographer(), return_report=True
    )
    assert doc.paragraphs[0].translated_text == text
    assert report["ambiguous"][0]["reason"] == "ambiguous_inflected_anchor"


def test_wrong_source_authorization_counts_and_persian_edits_still_fail():
    ctx = {"authorized": {"technical change": TARGET}}
    after = TARGET + " (technical change)"
    assert _proof_anchor("Something else.", TARGET, after, ctx)
    assert _proof_anchor("A technical change.", TARGET, after, {})
    assert _proof_anchor("A technical change.", TARGET, after + " (technical change)", ctx)
    assert _proof_anchor("A technical change.", TARGET + "!", after, ctx)


def test_repeated_export_is_idempotent_and_tampering_still_blocks():
    doc = document(INFLECTED)
    args = ({"technical change": TARGET}, PersianTypographer(), {"technical change": "theory"})
    ensure_inline_proper_noun_originals(doc, *args)
    saved = copy.deepcopy(doc)
    ensure_inline_proper_noun_originals(doc, *args)
    assert doc == saved
    ledger = RenderChangeLedger(doc)
    doc.paragraphs[0].translated_text += " altered"
    assert not ledger.audit()["passed"]


def test_full_authorized_phrase_can_include_an_internal_modifier():
    target = ("\u0646\u0638\u0631\u06cc\u0647 \u0633\u0646\u062a\u06cc "
              "\u0633\u0647\u200c\u0639\u0646\u0635\u0631\u06cc")
    original = "traditional three-element theory"
    doc = document(target, "A traditional three-element theory is introduced.")
    ledger = RenderChangeLedger(doc)
    ledger.set_authorized({original: target})
    ledger.step("inline_original_anchor", ensure_inline_proper_noun_originals,
                {original: target}, PersianTypographer(), {original: "theory"})
    assert doc.paragraphs[0].translated_text == target + f" ({original})"
    assert ledger.audit()["passed"]
