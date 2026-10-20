"""Scoped v20.8.8 regression controls."""

import copy
from types import SimpleNamespace

import pytest

from tarjomeh.core.render_identity import RenderChangeLedger, _proof_cleanup
from tarjomeh.core.term_notes import (
    audit_inline_english_originals,
    plan_inline_english_original_cleanup,
)
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph


def document(*, second_source="The editor works at Example Press."):
    return TranslatedDocument(
        paragraphs=[
            TranslatedParagraph(5, "Published by Example Press.", "نشر: نمونه (Example Press)."),
            TranslatedParagraph(46, second_source, "ویراستار در نمونه (Example Press) کار می‌کند."),
        ]
    )


AUTHORIZED = {"Example Press": "نمونه"}


def ledger_for(doc):
    ledger = RenderChangeLedger(doc)
    ledger.set_authorized(AUTHORIZED)
    return ledger


def test_cross_paragraph_cleanup_has_hash_bound_earlier_keeper():
    doc = document()
    before = copy.deepcopy(doc)
    planned, evidence = plan_inline_english_original_cleanup(doc, AUTHORIZED)
    assert doc == before
    ledger = ledger_for(doc)
    report = ledger.step("english_original_cleanup", audit_inline_english_originals, AUTHORIZED)
    assert [p.translated_text for p in doc.paragraphs] == planned
    assert report == evidence
    assert ledger.audit()["passed"]
    removal = report["removed_duplicates"][0]
    assert removal["paragraph_index"] == 46
    assert removal["keeper"]["paragraph_index"] == 5
    assert removal["keeper"]["source_sha256"]
    assert removal["before_sha256"] != removal["after_sha256"]
    assert "(Example Press)" in doc.paragraphs[0].translated_text
    assert "(Example Press)" not in doc.paragraphs[1].translated_text


@pytest.mark.parametrize(
    "source",
    [
        "The publisher (Example Press) returned.",
        "Example Press (Jones, 2020) returned.",
    ],
)
def test_source_authored_annotation_or_citation_survives(source):
    doc = document(second_source=source)
    if "Jones" in source:
        doc.paragraphs[1].translated_text += " (Jones, 2020)"
    ledger = ledger_for(doc)
    ledger.step("english_original_cleanup", audit_inline_english_originals, AUTHORIZED)
    assert ledger.audit()["passed"]
    expected = "(Jones, 2020)" if "Jones" in source else "(Example Press)"
    assert expected in doc.paragraphs[1].translated_text


def test_ungrounded_first_annotation_cannot_be_a_keeper():
    doc = document()
    doc.paragraphs[0].source_text = "This is an unrelated publisher."
    ledger = ledger_for(doc)
    report = ledger.step("english_original_cleanup", audit_inline_english_originals, AUTHORIZED)
    assert report["removed_duplicate_count"] == 0
    assert report["unapproved_ungrounded_count"] == 1
    assert ledger.audit()["passed"]
    assert "(Example Press)" in doc.paragraphs[1].translated_text


def test_no_earlier_keeper_means_no_removal():
    doc = document()
    doc.paragraphs.reverse()
    ledger = ledger_for(doc)
    report = ledger.step("english_original_cleanup", audit_inline_english_originals, AUTHORIZED)
    assert report["removed_duplicates"][0]["keeper"]["paragraph_index"] == 46
    assert "(Example Press)" in doc.paragraphs[0].translated_text
    assert ledger.audit()["passed"]


@pytest.mark.parametrize("damage", ["text", "source", "evidence"])
def test_misbehaving_cleanup_cannot_pass_by_replaying_itself(damage):
    doc = document()
    ledger = ledger_for(doc)

    def producer(document, authorized):
        report = audit_inline_english_originals(document, authorized)
        if damage == "text":
            document.paragraphs[1].translated_text += " اضافه"
        elif damage == "source":
            document.paragraphs[1].source_text = "Unrelated source."
        else:
            report["removed_duplicates"][0]["keeper"]["source_sha256"] = "bad"
        return report

    ledger.step("english_original_cleanup", producer, AUTHORIZED)
    assert not ledger.audit()["passed"]


@pytest.mark.parametrize("damage", ["annotation", "source", "index"])
def test_keeper_must_survive_until_publication(damage):
    doc = document()
    ledger = ledger_for(doc)
    ledger.step("english_original_cleanup", audit_inline_english_originals, AUTHORIZED)
    if damage == "annotation":
        doc.paragraphs[0].translated_text = "نشر: نمونه."
    elif damage == "source":
        doc.paragraphs[0].source_text = "Another publisher."
    else:
        doc.paragraphs[0].index = 9
    result = ledger.audit()
    assert not result["passed"]
    assert any(x["proof_failure"] == "cleanup_keeper_did_not_survive" for x in result["unproven"])


def test_legacy_paragraph_proof_does_not_get_a_blanket_exemption():
    assert _proof_cleanup("Example Press", "نمونه (Example Press)", "نمونه", {})


def test_keeper_can_survive_an_independently_proven_citation_merge():
    from tarjomeh.core.term_notes import normalize_adjacent_original_citations

    doc = document()
    doc.paragraphs[0].source_text = "Example Press (2020) publishes books."
    doc.paragraphs[0].translated_text = (
        doc.paragraphs[0].translated_text.rstrip(".") + " (2020)."
    )
    ledger = ledger_for(doc)
    ledger.step("english_original_cleanup", audit_inline_english_originals, AUTHORIZED)
    ledger.step("citation_merge", normalize_adjacent_original_citations, AUTHORIZED)
    assert "(Example Press, 2020)" in doc.paragraphs[0].translated_text
    assert ledger.audit()["passed"]


@pytest.mark.parametrize("year,removed", [("2020", 1), ("1999", 0)])
def test_citation_keeper_requires_source_proven_citation_not_just_the_name(year, removed):
    doc = document()
    doc.paragraphs[0].source_text = "Example Press (2020) publishes books."
    doc.paragraphs[0].translated_text = f"نمونه (Example Press, {year})."
    ledger = ledger_for(doc)
    report = ledger.step("english_original_cleanup", audit_inline_english_originals, AUTHORIZED)
    assert report["removed_duplicate_count"] == removed
    assert ledger.audit()["passed"]


def test_same_paragraph_duplicates_and_multiple_keys_keep_valid_spans():
    doc = document()
    doc.paragraphs[0].source_text += " Example Institute meets."
    doc.paragraphs[0].translated_text += " نمونه (Example Press)، نهاد (Example Institute)."
    doc.paragraphs[1].source_text += " Example Institute meets."
    doc.paragraphs[1].translated_text += " نهاد (Example Institute)."
    authorized = {**AUTHORIZED, "Example Institute": "نهاد"}
    ledger = RenderChangeLedger(doc)
    ledger.set_authorized(authorized)
    report = ledger.step("english_original_cleanup", audit_inline_english_originals, authorized)
    assert report["removed_duplicate_count"] == 3
    assert ledger.audit()["passed"]
    for removal in report["removed_duplicates"]:
        keeper = removal["keeper"]
        start, end = keeper["retained_span"]
        assert doc.paragraphs[keeper["position"]].translated_text[start:end] == keeper["annotation"]


@pytest.mark.parametrize("mode", ["target_only", "inline", "side_by_side"])
def test_actual_publication_still_blocks_unexplained_text_and_cleans_temp(tmp_path, mode):
    from tarjomeh.core.pipeline import TranslationPipeline
    from tarjomeh.exporters.docx_exporter import DocxExporter

    doc = document()
    ledger = ledger_for(doc)
    ledger.step("english_original_cleanup", audit_inline_english_originals, AUTHORIZED)
    doc.paragraphs[1].translated_text += " اضافه"
    artifacts = {}
    fake = SimpleNamespace(
        config=SimpleNamespace(output=SimpleNamespace(bilingual_mode=mode)),
        db=SimpleNamespace(save_job_artifact=lambda job, key, data: artifacts.update({key: data})),
    )
    output = tmp_path / "blocked.docx"
    with pytest.raises(RuntimeError, match="canonical-to-render proof failed"):
        TranslationPipeline._export_with_render_identity(
            fake, "synthetic", doc, DocxExporter(), output, ledger, "docx"
        )
    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp.docx"))
    assert not artifacts["render_identity_audit"]["passed"]
    assert artifacts["render_identity_audit"]["delivered"]["passed"]
