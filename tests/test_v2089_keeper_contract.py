"""Source-authored containers cannot seed an unprovable original removal."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tarjomeh.core.render_identity import RenderChangeLedger
from tarjomeh.core.term_notes import (
    audit_inline_english_originals,
    merge_inline_english_original_audits,
    original_keeper_kind,
    plan_inline_english_original_cleanup,
)
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph

NAME = "Example Press"
TARGET = "\u0646\u0645\u0648\u0646\u0647"
AUTHORIZED = {NAME: TARGET}


def container_document(suffix, *, third=False):
    annotation = f"({NAME}, {suffix})"
    paragraphs = [
        TranslatedParagraph(0, f"Published by {annotation}.", f"{TARGET} {annotation}."),
        TranslatedParagraph(1, f"{NAME} publishes books.", f"{TARGET} ({NAME})."),
    ]
    if third:
        paragraphs.append(TranslatedParagraph(2, f"{NAME} returned.", f"{TARGET} ({NAME})."))
    return TranslatedDocument(paragraphs=paragraphs)


@pytest.mark.parametrize("suffix", ["London", "Ltd", "2020", "1999", "London, 2020"])
def test_protected_source_containers_and_later_original_are_preserved(suffix):
    doc = container_document(suffix)
    before = copy.deepcopy(doc)
    expected, planned = plan_inline_english_original_cleanup(doc, AUTHORIZED)
    assert doc == before
    ledger = RenderChangeLedger(doc)
    ledger.set_authorized(AUTHORIZED)
    report = ledger.step("english_original_cleanup", audit_inline_english_originals, AUTHORIZED)
    assert report == planned and [p.translated_text for p in doc.paragraphs] == expected
    assert doc == before
    assert report["removed_duplicate_count"] == 0
    assert report["deferred_deduplication_count"] == 1
    assert report["deferred_deduplication"][0]["reason"] == "protected_container_not_proven_keeper"
    assert ledger.audit()["passed"]


@pytest.mark.parametrize("suffix", ["London", "Ltd", "2020"])
def test_next_short_original_can_prove_only_subsequent_duplicates(suffix):
    doc = container_document(suffix, third=True)
    ledger = RenderChangeLedger(doc)
    ledger.set_authorized(AUTHORIZED)
    report = ledger.step("english_original_cleanup", audit_inline_english_originals, AUTHORIZED)
    assert report["removed_duplicate_count"] == 1 and ledger.audit()["passed"]
    removal = report["removed_duplicates"][0]
    assert removal["paragraph_index"] == 2 and removal["keeper"]["paragraph_index"] == 1
    assert f"({NAME}, {suffix})" in doc.paragraphs[0].translated_text
    assert f"({NAME})" in doc.paragraphs[1].translated_text
    assert f"({NAME})" not in doc.paragraphs[2].translated_text
    doc.paragraphs[1].translated_text = TARGET
    assert not ledger.audit()["passed"]


@pytest.mark.parametrize("source,content,expected", [
    ("Example Press publishes.", "Example Press", "exact_original"),
    ("The publisher (Example Press) returned.", "Example Press", "exact_original"),
    ("Example Press (2020) publishes.", "Example Press, 2020", "source_proven_citation"),
    ("Example Press (2020) publishes.", "Example Press, 1999", ""),
    ("Unrelated source.", "Example Press", ""),
    ("Published by (Example Press, London).", "Example Press, London", ""),
    ("Published by (Example Press, 2020).", "Example Press, 2020", ""),
])
def test_shared_keeper_classification_is_narrow(source, content, expected):
    assert original_keeper_kind(source, content, NAME) == expected


def test_complete_authorized_original_with_comma_remains_an_exact_original():
    name = "Example Press, Ltd"
    assert original_keeper_kind(f"Published by ({name}).", name, name) == "exact_original"


@pytest.mark.parametrize("separator", [" ", "   ", "\t", "\u00a0", "\u2009"])
def test_citation_whitespace_has_one_classification_before_and_after_cleanup(separator):
    first = f"{TARGET} (Example{separator}Press, 2020)."
    doc = TranslatedDocument(paragraphs=[
        TranslatedParagraph(0, "Example Press (2020) publishes.", first),
        TranslatedParagraph(1, "Example Press returned.", f"{TARGET} ({NAME})."),
    ])
    ledger = RenderChangeLedger(doc)
    ledger.set_authorized(AUTHORIZED)
    report = ledger.step("english_original_cleanup", audit_inline_english_originals, AUTHORIZED)
    assert report["removed_duplicate_count"] == 1 and ledger.audit()["passed"]
    assert original_keeper_kind(doc.paragraphs[0].source_text,
                                f"Example{separator}Press, 2020", NAME) == "source_proven_citation"


def test_deferral_is_nonblocking_and_retained_across_passes():
    doc = container_document("London")
    reports = [audit_inline_english_originals(doc, AUTHORIZED) for _ in range(3)]
    merged = merge_inline_english_original_audits(reports)
    assert merged["deferred_deduplication_count"] == len(merged["deferred_deduplication"]) == 3
    assert merged["removed_duplicate_count"] == 0
    assert all(row["authority"] == "report_only_no_memory_or_quality_penalty"
               for row in merged["deferred_deduplication"])


@pytest.mark.parametrize("mode", ["target_only", "inline", "side_by_side"])
@pytest.mark.parametrize("suffix", ["London", "Ltd", "2020"])
@pytest.mark.parametrize("first_anchor", [False, True])
def test_normal_producer_composition_and_atomic_docx_publication(
    tmp_path, mode, suffix, first_anchor,
):
    from tarjomeh.core.pipeline import TranslationPipeline
    from tarjomeh.core.term_notes import (
        ensure_inline_proper_noun_originals,
        normalize_adjacent_original_citations,
        reconcile_redundant_original_fragments,
    )
    from tarjomeh.exporters.docx_exporter import DocxExporter
    from tarjomeh.persian.typography import PersianTypographer

    doc = container_document(suffix, third=True)
    if not first_anchor:
        doc.paragraphs[0].translated_text = f"ناشر ({NAME}, {suffix})."
    ledger = RenderChangeLedger(doc)
    canonical = copy.deepcopy(ledger.canonical)
    ledger.set_authorized(AUTHORIZED)
    for _ in range(3):
        ledger.step("inline_original_anchor", ensure_inline_proper_noun_originals,
                    AUTHORIZED, PersianTypographer(), {NAME: "organization"},
                    aliases={}, return_report=True)
        ledger.step("citation_merge", normalize_adjacent_original_citations, AUTHORIZED)
        ledger.step("original_fragment_reconciliation", reconcile_redundant_original_fragments,
                    AUTHORIZED)
        ledger.step("english_original_cleanup", audit_inline_english_originals, AUTHORIZED)
    assert ledger.canonical == canonical and ledger.audit()["passed"]
    assert f"({NAME}, {suffix})" in doc.paragraphs[0].translated_text
    if first_anchor and suffix != "2020":
        # An independently proven insertion resolves uncertainty before cleanup.
        assert f"({NAME})" in doc.paragraphs[0].translated_text
        assert f"({NAME})" not in doc.paragraphs[1].translated_text
        assert any(c["producer"] == "inline_original_anchor" for c in ledger.changes)
    else:
        assert f"({NAME})" in doc.paragraphs[1].translated_text
    assert f"({NAME})" not in doc.paragraphs[2].translated_text
    artifacts = {}
    fake = SimpleNamespace(
        config=SimpleNamespace(output=SimpleNamespace(bilingual_mode=mode)),
        db=SimpleNamespace(save_job_artifact=lambda job, key, data: artifacts.update({key: data})),
    )
    output = tmp_path / "checkpoint.docx"
    TranslationPipeline._export_with_render_identity(
        fake, "synthetic-not-a-job", doc, DocxExporter(), output, ledger, "docx")
    assert output.is_file() and artifacts["render_identity_audit"]["passed"]
    assert not list(tmp_path.glob("*.tmp.docx"))


def test_new_probe_has_no_files_or_configuration_and_cannot_pass_a_noop(monkeypatch):
    import tarjomeh.core.term_notes as notes
    from tarjomeh.runtime import _v2089_behavior_probes

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected file I/O")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "is_file", forbidden)
    assert all(_v2089_behavior_probes().values())
    monkeypatch.setattr(
        notes, "audit_inline_english_originals",
        lambda *a, **k: {"removed_duplicate_count": 0, "deferred_deduplication_count": 0},
    )
    assert not _v2089_behavior_probes()["shared_original_keeper_eligibility"]


def test_new_probe_rejects_the_old_overbroad_keeper_rule(monkeypatch):
    import tarjomeh.core.term_notes as notes
    from tarjomeh.runtime import _v2089_behavior_probes

    monkeypatch.setattr(notes, "original_keeper_kind", lambda *a: "exact_original")
    assert not _v2089_behavior_probes()["shared_original_keeper_eligibility"]


def test_reporting_distinguishes_legacy_absence_and_never_changes_state():
    from tarjomeh.core.evidence_audit import checkpoint_classification_evidence

    state = {"proper_nouns": {"nouns": {NAME: TARGET}}}
    before = copy.deepcopy(state)
    old = checkpoint_classification_evidence(state, {}, [])
    assert old["deferred_deduplication"]["available"] is False
    assert old["deferred_deduplication"]["record_count"] is None
    cleanup = audit_inline_english_originals(container_document("London"), AUTHORIZED)
    report = checkpoint_classification_evidence(state, {}, [], original_audit=cleanup)
    assert report["deferred_deduplication"]["record_count"] == 1
    assert report["authority"] == "report_only_no_memory_or_quality_penalty"
    assert state == before
