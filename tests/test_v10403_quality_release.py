"""v10.40.3: complete style pairs, per-paragraph affix repair, render identity,
per-occurrence book terms, imprint names and malformed-option screening.

Real strings from the audited book appear only here, never in production code.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    TranslationPipeline,
    audit_translation_language,
    repair_document_source_grounded_language_artifacts,
    repair_source_grounded_paragraphs,
)
from tarjomeh.core.render_identity import (
    RenderChangeLedger,
    compare_docx_to_rendered,
)
from tarjomeh.core.term_notes import normalize_adjacent_original_citations
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.docx_exporter import DocxExporter
from tarjomeh.glossary.book_review import (
    REVIEW_VERSION,
    check_reviewed_book_terms,
    find_term_occurrences,
    persian_option_defect,
    resolve_reviewed_book_terms,
)
from tarjomeh.jobs.database import JobDatabase, JobStatus
from tarjomeh.memory.manager import (
    MemoryManager,
    _clean_style_sample,
    _style_record_is_authoritative,
    _style_sample_quality,
)
from tarjomeh.quality.integrity import (
    spaced_optional_prefix_artifacts,
    unexpected_latin_prose,
)

FOUR_NINES = dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9.5)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk(index: int, paragraphs: list[str], first_global: int,
           roles: list[str] | None = None, chapter: int = 1) -> Chunk:
    text = "\n\n".join(paragraphs)
    spans, cursor = [], 0
    for paragraph in paragraphs:
        spans.append([cursor, cursor + len(paragraph)])
        cursor += len(paragraph) + 2
    return Chunk(index=index, text=text, chapter_title="Chapter", section_title="",
                 metadata={
                     "paragraph_indices": list(range(first_global, first_global + len(paragraphs))),
                     "source_paragraph_spans": spans,
                     "source_paragraph_hashes": [_sha(p) for p in paragraphs],
                     "structural_roles": roles or ["body"] * len(paragraphs),
                     "chapter_position": chapter,
                 })


def _index(chunks: list[Chunk]) -> list[dict]:
    entries = []
    for chunk in chunks:
        for local, global_index in enumerate(chunk.metadata["paragraph_indices"]):
            paragraph = chunk.text.split("\n\n")[local]
            entries.append({"chunk_index": chunk.index, "local_paragraph": local,
                            "paragraph_index": global_index,
                            "paragraph_sha256": _sha(paragraph),
                            "chapter_position": chunk.metadata["chapter_position"]})
    return entries


def _records(chunks: list[Chunk]) -> dict:
    return {chunk.index: (chunk.text, chunk.metadata) for chunk in chunks}


# --- 1. Complete style pairs (R118/T20) -----------------------------------

CITED_TARGET = (
    "این جمله نمونه‌ای روشن از نثر دانشگاهی پیوسته و دقیق است. "
    "این ادعا در پژوهش‌های پیشین بررسی شده است (Tilly 1975; Spruyt 1993)."
)


def test_style_record_stores_the_complete_paragraph_with_both_hashes():
    manager = MemoryManager(TarjomehConfig())
    source = "A clear and continuous academic sentence. The claim was examined (Tilly 1975)."
    manager._update_style_profile(
        CITED_TARGET, source_paragraphs=[source], source_paragraph_indices=[0],
        source_alignment_proven=True, final_scores=FOUR_NINES,
    )
    record = manager.style_sample_records[-1]
    assert record["text"] == CITED_TARGET
    assert "Tilly" in record["text"]
    assert record["sample_scope"] == "complete_paragraph"
    assert record["alignment_status"] == "exact_paragraph"
    assert record["text_hash"] == _sha(CITED_TARGET)
    assert record["source_text_hash"] == _sha(source)
    assert _style_record_is_authoritative(record)


def test_style_length_limit_is_eligibility_not_authority():
    long_paragraph = ("این جمله‌ای روشن و کامل در نثر دانشگاهی است. " * 40).strip()
    assert len(long_paragraph) > 1600
    assert _clean_style_sample(long_paragraph) == ""
    assert _style_sample_quality(long_paragraph)["reasons"] == [
        "paragraph_exceeds_style_sample_limit"
    ]


def test_citations_are_masked_for_scoring_but_artifacts_checked_unmasked():
    cited = (
        "این دیدگاه در آثار متعدد بررسی شده است "
        "(Aaron 1990, 12; Baker 1991, 14; Cohen 1992, 16; Dunn 1993, 18). "
        "نتیجه روشن است."
    )
    assert _style_sample_quality(cited)["approved"]
    source = "It implies (meta)theoretical pluralism."
    spaced = "این امر متضمن کثرت‌گراییِ (فرا) نظری است."
    record = {
        "text": spaced, "source_text": source, "alignment_status": "exact_paragraph",
        "sample_scope": "complete_paragraph", "text_hash": _sha(spaced),
        "source_text_hash": _sha(source), "representative": True,
        "quality_score": 100.0, "final_scores": FOUR_NINES,
    }
    assert spaced_optional_prefix_artifacts(source, spaced)
    assert not _style_record_is_authoritative(record)


def test_flattened_note_marker_sentence_disqualifies_the_paragraph():
    assert _clean_style_sample(CITED_TARGET + " 2 و این یادداشت است.") == ""


def test_legacy_clipped_record_is_quarantined_but_a_complete_one_is_verified():
    manager = MemoryManager(TarjomehConfig())
    source = "The academic argument continues here."
    full = "استدلال دانشگاهی در اینجا ادامه می‌یابد."
    clipped = "استدلال دانشگاهی."
    state = {
        "style_samples": [clipped, full],
        "style_sample_records": [
            {"text": clipped, "text_hash": _sha(clipped), "source_text": source,
             "alignment_status": "exact_paragraph", "representative": True,
             "quality_score": 100.0, "final_scores": FOUR_NINES,
             "source_chunk_index": 3},
            {"text": full, "text_hash": _sha(full), "source_text": source,
             "alignment_status": "exact_paragraph", "representative": True,
             "quality_score": 100.0, "final_scores": FOUR_NINES,
             "source_chunk_index": 3},
        ],
        "past_translations": [{"entry_id": 0, "source": source, "translation": full,
                               "reliable": True, "chunk_index": 3}],
    }
    manager.from_dict(state)
    statuses = [record["legacy_style_status"] for record in manager.style_sample_records]
    assert statuses == ["unverifiable_quarantined", "verified_complete_pair"]
    assert not _style_record_is_authoritative(manager.style_sample_records[0])
    assert _style_record_is_authoritative(manager.style_sample_records[1])
    assert manager.style_samples == [clipped, full]


# --- 2. Optional-prefix and plural repair before canonical hashing --------

def test_prefix_is_repaired_per_paragraph_inside_a_mixed_chunk():
    source = "1 Introduction\n\nIt implies (meta)theoretical pluralism."
    target = "۱ مقدمه\n\nاین امر متضمن کثرت‌گراییِ (فرا) نظری است."
    repaired, report = repair_source_grounded_paragraphs(
        source, target, ["heading", "body"], require_monotonic=True,
    )
    assert repaired == "۱ مقدمه\n\nاین امر متضمن کثرت‌گراییِ (فرا)نظری است."
    assert report["scope"] == "aligned_paragraphs"
    assert [item["paragraph_index"] for item in report["repairs"]] == [1]


def test_prefix_count_makes_the_repair_a_monotonic_improvement():
    source = "It implies (meta)theoretical pluralism."
    before = audit_translation_language(source, "کثرت‌گراییِ (فرا) نظری.")
    after = audit_translation_language(source, "کثرت‌گراییِ (فرا)نظری.")
    assert before["spaced_optional_prefix_count"] == 1
    assert after["spaced_optional_prefix_count"] == 0


def test_repeated_identical_prefixes_join_but_ambiguous_ones_stay_review():
    twice, _ = repair_source_grounded_paragraphs(
        "Both (inter)state systems and (inter)state rivalry matter.",
        "هم نظام‌های (بینا) دولتی و هم رقابت (بینا) دولتی مهم‌اند.",
        ["body"], require_monotonic=True,
    )
    assert twice.count("(بینا)دولتی") == 2
    mixed_target = "هم دیدگاه‌های (فرا) نظری و هم (بینا) دولتی مهم‌اند."
    unchanged, _ = repair_source_grounded_paragraphs(
        "Both (meta)theoretical and (inter)state views matter.",
        mixed_target, ["body"], require_monotonic=True,
    )
    assert unchanged == mixed_target


def test_joined_optional_plural_before_next_word_is_not_a_finding():
    source = "discourse(s) shape (meta)theoretical claims."
    target = "گفتمان(ها) به مدعیات (فرا)نظری شکل می‌دهند."
    assert spaced_optional_prefix_artifacts(source, target) == []
    document = TranslatedDocument(title="t", author="a", paragraphs=[
        TranslatedParagraph(index=0, source_text=source, translated_text=target,
                            metadata={"structure_role": "body"}),
    ], metadata={})
    from tarjomeh.core.pipeline import audit_document_final_text

    findings = audit_document_final_text(document)["report_only_findings"]
    assert not [item for item in findings if item.get("check_id") == "spaced_optional_affix"]


def _legacy_document() -> TranslatedDocument:
    return TranslatedDocument(title="t", author="a", paragraphs=[
        TranslatedParagraph(
            index=0,
            source_text="Critical discourse analysis explores how discourse(s) shape the state.",
            translated_text=(
                "تحلیل گفتمان انتقادی بررسی می‌کند که چگونه "
                "گفتمان (ها) به دولت شکل می‌دهند."
            ),
            metadata={"structure_role": "body"},
        ),
    ], metadata={})


def test_legacy_reexport_repairs_at_render_records_it_and_keeps_memory():
    document = _legacy_document()
    canonical = document.paragraphs[0].translated_text
    ledger = RenderChangeLedger(document)
    ledger.step("render_language_repair", repair_document_source_grounded_language_artifacts)
    audit = ledger.audit()
    assert "گفتمان(ها)" in document.paragraphs[0].translated_text
    assert audit["passed"]
    assert audit["changes_by_producer"] == {"render_language_repair": 1}
    assert ledger.canonical[0] == canonical


def test_fresh_job_has_no_render_affix_change_after_precanonical_repair():
    document = _legacy_document()
    fixed, _ = repair_source_grounded_paragraphs(
        document.paragraphs[0].source_text, document.paragraphs[0].translated_text,
        ["body"], require_monotonic=True,
    )
    document.paragraphs[0].translated_text = fixed
    ledger = RenderChangeLedger(document)
    ledger.step("render_language_repair", repair_document_source_grounded_language_artifacts)
    assert ledger.audit()["change_count"] == 0


# --- 3. Render identity ----------------------------------------------------

def _document(source: str, target: str) -> TranslatedDocument:
    return TranslatedDocument(title="t", author="a", paragraphs=[
        TranslatedParagraph(index=0, source_text=source, translated_text=target,
                            metadata={"structure_role": "body"}),
    ], metadata={})


def test_adjacent_original_citation_merge_passes_by_replaying_its_producer():
    document = _document(
        "Exemplary here is Shmuel Eisenstadt's (1963) work.",
        "نمونهٔ برجسته اثر شموئل آیزنشتات (Shmuel Eisenstadt) (1963) است.",
    )
    ledger = RenderChangeLedger(document)
    ledger.step("citation_merge", normalize_adjacent_original_citations,
                {"Shmuel Eisenstadt": "شموئل آیزنشتات"})
    audit = ledger.audit()
    assert "(Shmuel Eisenstadt, 1963)" in document.paragraphs[0].translated_text
    assert audit["passed"], audit["unproven"]
    assert audit["changes"][0]["replay_verified"] is True


def test_citation_change_that_alters_a_year_blocks():
    document = _document("As Tilly (1975) argued.", "چنان‌که تیلی (Tilly) (1975) استدلال کرد.")

    def wrong_year(doc, _authorized):
        doc.paragraphs[0].translated_text = "چنان‌که تیلی (Tilly, 1976) استدلال کرد."

    ledger = RenderChangeLedger(document)
    ledger.step("citation_merge", wrong_year, {})
    audit = ledger.audit()
    assert not audit["passed"]
    assert audit["unproven"][0]["proof_failure"] == "citation_years_changed"


def _anchor_audit(target_after: str, source: str = "The State was examined.") -> dict:
    document = _document(source, "دولت بررسی شد و نهاد نیز.")

    def insert(doc, *_args, **_kwargs):
        doc.paragraphs[0].translated_text = target_after

    ledger = RenderChangeLedger(document)
    ledger.set_authorized({"State": "دولت"})
    ledger.step("inline_original_anchor", insert)
    return ledger.audit()


def test_anchor_next_to_its_persian_rendering_passes():
    assert _anchor_audit("دولت (State) بررسی شد و نهاد نیز.")["passed"]


def test_anchor_at_the_wrong_position_fails():
    audit = _anchor_audit("دولت بررسی شد و نهاد (State) نیز.")
    assert audit["unproven"][0]["proof_failure"] == "anchor_position_unproven"


def test_more_anchors_than_source_occurrences_fail():
    audit = _anchor_audit("دولت (State) بررسی شد و دولت (State) نیز.")
    assert not audit["passed"]


def test_unknown_difference_outside_the_ledger_blocks():
    document = _document("A source.", "یک متن.")
    ledger = RenderChangeLedger(document)
    document.paragraphs[0].translated_text = "یک متن تغییر یافته."
    audit = ledger.audit()
    assert not audit["passed"]
    assert audit["unexplained"][0]["reason"] == "difference_without_recorded_producer"


def test_docx_readback_maps_contents_rows_and_detects_tampering(tmp_path):
    document = TranslatedDocument(title="t", author="a", paragraphs=[
        TranslatedParagraph(
            index=0, source_text="Preface viii",
            translated_text="پیش‌گفتار viii",
            metadata={"structure_role": "contents_entry", "toc_page_label": "viii"},
        ),
        TranslatedParagraph(index=1, source_text="Body.", translated_text="متن اصلی.",
                            metadata={"structure_role": "body"}),
    ], metadata={})
    path = tmp_path / "out.docx"
    DocxExporter({}).export(document=document, output_path=path, bilingual_mode="target_only")
    assert compare_docx_to_rendered(document, path, "target_only")["passed"]
    document.paragraphs[1].translated_text = "متن دیگر."
    assert not compare_docx_to_rendered(document, path, "target_only")["passed"]


def test_export_is_published_only_when_both_identities_pass(tmp_path):
    pipeline = object.__new__(TranslationPipeline)
    pipeline.config = TarjomehConfig()
    pipeline.db = MagicMock()
    document = _document("A source.", "یک متن.")
    output = tmp_path / "book.docx"
    ledger = RenderChangeLedger(document)
    pipeline._export_with_render_identity(
        "job", document, DocxExporter({}), output, ledger, "docx")
    assert output.is_file()
    blocked = tmp_path / "blocked.docx"
    ledger = RenderChangeLedger(document)
    document.paragraphs[0].translated_text = "متن بی‌سند."
    with pytest.raises(RuntimeError, match="Export blocked"):
        pipeline._export_with_render_identity(
            "job", document, DocxExporter({}), blocked, ledger, "docx")
    assert not blocked.exists()
    assert not list(tmp_path.glob(".blocked*"))


# --- 4. Per-occurrence book terms -----------------------------------------

APPARATUS = "The state apparatus acts. Later the apparatus changes."
SINGLE = "The apparatus of rule is examined."


def test_occurrences_have_unique_identities_even_for_identical_paragraphs():
    chunks = [_chunk(0, [SINGLE], 0), _chunk(1, [SINGLE, "Polity Press"], 5)]
    found = find_term_occurrences("apparatus", _index(chunks), _records(chunks))
    assert [item["paragraph_index"] for item in found] == [0, 5]
    assert not find_term_occurrences("Polity", _index(chunks), _records(chunks))


def _term(chunk: Chunk, ticks: list[dict]) -> dict:
    return {"source": "apparatus", "target": "دستگاه", "status": "approved",
            "scope_mode": "reviewed_occurrences", "approved_occurrences": ticks}


def test_all_ticked_single_occurrence_is_mandatory_and_unticked_is_not():
    chunk = _chunk(0, [SINGLE, "The apparatus again."], 0)
    occurrences = find_term_occurrences("apparatus", _index([chunk]), _records([chunk]))
    matches, review = resolve_reviewed_book_terms(
        chunk.text, chunk.metadata, 1, [_term(chunk, occurrences[:1])])
    assert [match["paragraph_index"] for match in matches] == [0]
    assert review == []


def test_partial_ticks_in_one_paragraph_are_review_not_prompted():
    chunk = _chunk(0, [APPARATUS], 0)
    occurrences = find_term_occurrences("apparatus", _index([chunk]), _records([chunk]))
    assert len(occurrences) == 2
    matches, review = resolve_reviewed_book_terms(
        chunk.text, chunk.metadata, 1, [_term(chunk, occurrences[:1])])
    assert matches == []
    assert review[0]["reason"] == "partial_occurrence_approval"


def test_repeated_occurrences_are_prompted_but_never_reported_compliant():
    chunk = _chunk(0, [APPARATUS], 0)
    occurrences = find_term_occurrences("apparatus", _index([chunk]), _records([chunk]))
    matches, review = resolve_reviewed_book_terms(
        chunk.text, chunk.metadata, 1, [_term(chunk, occurrences)])
    assert matches and matches[0]["source_occurrence_count"] == 2
    assert review[0]["reason"] == "multiple_occurrences_unverifiable"
    report, uncertain = check_reviewed_book_terms(
        "دستگاه دولت عمل می‌کند. سپس آن تغییر می‌کند.", chunk.text, chunk.metadata, matches)
    assert report.total_checked == 0
    assert uncertain[0]["reasons"] == [
        "multiple_occurrences_unverifiable", "rendering_count_short"]


def test_stale_occurrence_identity_is_review():
    chunk = _chunk(0, [SINGLE], 0)
    occurrences = find_term_occurrences("apparatus", _index([chunk]), _records([chunk]))
    stale = [{**occurrences[0], "paragraph_sha256": "0" * 64}]
    matches, review = resolve_reviewed_book_terms(
        chunk.text, chunk.metadata, 1, [_term(chunk, stale)])
    assert matches == []
    assert review[0]["reason"] == "approved_occurrence_changed"


@pytest.fixture
def review_client(tmp_path, monkeypatch):
    source = tmp_path / "book.txt"
    source.write_text("book", encoding="utf-8")
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("job", source, {})
    db.update_job_status("job", JobStatus.AWAITING_BOOK_TERM_REVIEW)
    chunks = [_chunk(0, [SINGLE], 0), _chunk(1, [APPARATUS], 1)]
    db.save_chunks("job", chunks)
    source_hash = db.get_job_source_hash("job")
    db.save_job_artifact("job", "book_term_body_index_v1",
                         {"source_sha256": source_hash, "entries": _index(chunks)})
    db.save_job_artifact("job", "book_term_review_v1", {
        "phase": "awaiting", "review_version": REVIEW_VERSION,
        "source_sha256": source_hash,
        "proposals": [{"source": "apparatus", "target": "", "status": "proposed",
                       "source_evidence": SINGLE,
                       "source_evidence_sha256": _sha(SINGLE)}],
    })
    monkeypatch.setenv("UI_SECRET_TOKEN", "test-token")
    from tarjomeh.web.app import create_app

    app = create_app()
    app.config["TESTING"] = True
    with patch("tarjomeh.jobs.database.JobDatabase", return_value=db):
        yield SimpleNamespace(
            client=app.test_client(), db=db,
            headers={"Authorization": "Bearer test-token"},
            route="/api/jobs/job/book-term-review",
        )


def test_api_lists_occurrences_and_verifies_ticks(review_client):
    c = review_client
    listed = c.client.get(c.route + "?occurrences=0", headers=c.headers).get_json()
    assert listed["count"] == 3
    assert all(item["paragraph"] for item in listed["occurrences"])
    forged = {**listed["occurrences"][0], "start": 1}
    decision = {"action": "decide", "confirm_bulk": True, "decisions": [{
        "index": 0, "status": "approved", "target": "دستگاه",
        "scope_mode": "reviewed_occurrences", "occurrences": [forged]}]}
    assert c.client.post(c.route, headers=c.headers, json=decision).status_code == 409
    decision["decisions"][0]["occurrences"] = listed["occurrences"][:1]
    assert c.client.post(c.route, headers=c.headers, json=decision).status_code == 200
    saved = c.db.get_job_artifact("job", "book_term_review_v1")["proposals"][0]
    assert saved["approved_occurrences"][0]["paragraph_index"] == 0
    assert c.client.post(c.route, headers=c.headers, json={"action": "start"}).status_code == 200
    stored = c.db.get_job_artifact("job", "book_term_review_v1")
    assert stored["proposals"][0]["approved_occurrences"]


def test_api_rejects_new_all_body_approval(review_client):
    c = review_client
    decision = {"action": "decide", "confirm_broad_scope": True, "decisions": [{
        "index": 0, "status": "approved", "target": "دستگاه", "scope_mode": "all_body"}]}
    assert c.client.post(c.route, headers=c.headers, json=decision).status_code == 409


def test_api_adds_a_job_scoped_term_without_touching_the_shared_glossary(review_client):
    c = review_client
    with patch("tarjomeh.glossary.manager.GlossaryManager.save") as saved:
        response = c.client.post(c.route, headers=c.headers, json={
            "action": "add", "source": "state apparatus", "target": "دستگاه دولتی"})
    assert response.status_code == 200
    assert not saved.called
    added = c.db.get_job_artifact("job", "book_term_review_v1")["proposals"][-1]
    assert added["origin"] == "user_added"
    assert added["source_count"] == 1
    missing = c.client.post(c.route, headers=c.headers, json={
        "action": "add", "source": "hegemonic project", "target": "پروژهٔ هژمونیک"})
    assert missing.status_code == 400
    duplicate = c.client.post(c.route, headers=c.headers, json={
        "action": "add", "source": "Apparatus", "target": "دستگاه"})
    assert duplicate.status_code == 409


def test_legacy_review_keeps_accepting_confirmed_all_body(tmp_path, monkeypatch):
    source = tmp_path / "book.txt"
    source.write_text("book", encoding="utf-8")
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("old", source, {})
    db.update_job_status("old", JobStatus.AWAITING_BOOK_TERM_REVIEW)
    db.save_job_artifact("old", "book_term_review_v1", {
        "phase": "awaiting", "source_sha256": db.get_job_source_hash("old"),
        "proposals": [{"source": "apparatus", "target": "", "status": "proposed",
                       "source_evidence": SINGLE, "source_evidence_sha256": _sha(SINGLE)}],
    })
    monkeypatch.setenv("UI_SECRET_TOKEN", "test-token")
    from tarjomeh.web.app import create_app

    app = create_app()
    app.config["TESTING"] = True
    with patch("tarjomeh.jobs.database.JobDatabase", return_value=db):
        response = app.test_client().post(
            "/api/jobs/old/book-term-review",
            headers={"Authorization": "Bearer test-token"},
            json={"action": "decide", "confirm_broad_scope": True, "confirm_bulk": True,
                  "decisions": [{"index": 0, "status": "approved", "target": "دستگاه",
                                 "scope_mode": "all_body"}]},
        )
    assert response.status_code == 200


# --- 5. Printer imprint names (R131) --------------------------------------

IMPRINT = ("Typeset in 10.5 on 12 pt Sabon by Toppan Best-set Premedia Limited "
           "Printed and bound in the United Kingdom by Clays Ltd, St Ives PLC")
IMPRINT_TARGET = ("حروف‌چینی با قلم سابون توسط تاپان بست-ست پریمیدیا لیمیتد "
                  "(Toppan Best-set Premedia Limited)؛ چاپ و صحافی توسط کلیز لیمیتد (Clays Ltd).")


def test_exact_imprint_names_are_not_unexplained_latin():
    assert unexpected_latin_prose(IMPRINT, IMPRINT_TARGET, allowed_originals=["Clays Ltd"]) == []


def test_leaked_imprint_sentence_and_body_printed_stay_under_scrutiny():
    leaked = unexpected_latin_prose(
        IMPRINT, IMPRINT_TARGET + " Printed and bound in the United Kingdom.",
        allowed_originals=["Clays Ltd", "United Kingdom"])
    assert {"Printed", "bound"} <= {item["token"] for item in leaked}
    body = unexpected_latin_prose(
        "The pamphlet was printed by the Party Press in 1920.",
        "این جزوه توسط Party Press چاپ شد.")
    assert body


# --- 6. Malformed Persian proposals ---------------------------------------

def test_malformed_persian_options_are_withheld_with_a_reason():
    assert persian_option_defect("رویکرد راهبردی‌ـ‌رابطه‌ای") == "tatweel_in_persian_option"
    assert persian_option_defect("دولت‌‌ملت") == "double_zwnj_in_persian_option"
    assert persian_option_defect("سیاستpolicy") == "mixed_script_persian_option"
    assert persian_option_defect("رویکرد راهبردی–رابطه‌ای") == ""


# --- 7. Duplication measurement only --------------------------------------

def test_previous_chunk_duplication_is_measured_not_removed():
    manager = MemoryManager(TarjomehConfig())
    manager.short_term.add("Source one.", "ترجمهٔ یک.")
    context = manager.get_context_for_chunk(SimpleNamespace(text="Source two."))
    measurement = context.references["duplication"]
    assert measurement["short_term_translation_sha256"] == [_sha("ترجمهٔ یک.")]
    assert "ترجمهٔ یک." in context.short_term
