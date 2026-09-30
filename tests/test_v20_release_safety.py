"""Regression checks for selected chapters, delivered DOCX text and job terms."""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest

from tarjomeh.chunking.chunker import Chunk, SemanticChunker
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import apply_chapter_selection, build_chapter_manifest
from tarjomeh.core.render_identity import compare_docx_to_rendered
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.docx_exporter import DocxExporter
from tarjomeh.glossary.book_review import body_paragraph_index, find_term_occurrences
from tarjomeh.jobs.database import JobDatabase, JobStatus
from tarjomeh.parsers.base import Chapter, Document, Paragraph, Section
from tarjomeh.web.app import create_app


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_selected_chapter_term_index_uses_selected_paragraph_numbering():
    first = Chapter(
        "Copyright", sections=[Section("", 2, [Paragraph("Copyright 2026.")])]
    )
    body_text = "The apparatus mediates state power in this chapter."
    second = Chapter(
        "Chapter 1: Institutions", sections=[Section("", 2, [Paragraph(body_text)])]
    )
    full = Document("Book", chapters=[first, second])
    build_chapter_manifest(full)
    selected = apply_chapter_selection(full, [2])
    chunks = SemanticChunker(
        max_tokens=1000, overlap_sentences=0, token_counter=lambda text: len(text.split())
    ).chunk(selected)

    index = body_paragraph_index(selected, chunks)
    assert len(index) == 1
    assert index[0]["paragraph_index"] == 0
    occurrences = find_term_occurrences(
        "apparatus", index,
        {chunk.index: (chunk.text, chunk.metadata) for chunk in chunks},
    )
    assert len(occurrences) == 1
    assert occurrences[0]["chapter_position"] == 2


@pytest.mark.parametrize("mode", ["target_only", "inline", "side_by_side"])
def test_docx_readback_checks_every_supported_layout_and_both_languages(tmp_path, mode):
    document = TranslatedDocument(
        title="Book", author="Author", paragraphs=[
            TranslatedParagraph(
                index=0, source_text="A source paragraph.",
                translated_text="یک بند فارسی.", metadata={"structure_role": "body"},
            ),
            TranslatedParagraph(
                index=1, source_text="The next paragraph.",
                translated_text="بند بعدی.", metadata={"structure_role": "body"},
            ),
        ], metadata={},
    )
    path = tmp_path / "book.docx"
    DocxExporter({}).export(document, path, bilingual_mode=mode)
    assert compare_docx_to_rendered(document, path, mode)["passed"]

    document.paragraphs[1].translated_text = "متن متفاوت."
    assert not compare_docx_to_rendered(document, path, mode)["passed"]
    if mode != "target_only":
        document.paragraphs[1].translated_text = "بند بعدی."
        document.paragraphs[0].source_text = "Changed English source."
        assert not compare_docx_to_rendered(document, path, mode)["passed"]


def test_unrecognized_docx_layout_never_passes_without_readback(tmp_path):
    document = TranslatedDocument(title="Book", author="", paragraphs=[], metadata={})
    audit = compare_docx_to_rendered(document, tmp_path / "absent.docx", "unknown")
    assert audit["passed"] is False


def test_job_glossary_conflicts_override_different_server_glossary(tmp_path, monkeypatch):
    job_glossary = tmp_path / "job.csv"
    job_glossary.write_text(
        "source,target,tgt_lng\n"
        "apparatus,سازوکار,fa\n"
        "state apparatus,دستگاه حکومتی,fa\n", encoding="utf-8",
    )
    server_glossary = tmp_path / "server.csv"
    server_glossary.write_text(
        "source,target,tgt_lng\n"
        "apparatus,دستگاه,fa\n"
        "state apparatus,دستگاه دولتی,fa\n", encoding="utf-8",
    )
    upload = tmp_path / "book.txt"
    upload.write_text("book", encoding="utf-8")
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("job", upload, {"glossary": {"path": str(job_glossary), "paths": []}})
    db.update_job_status("job", JobStatus.AWAITING_BOOK_TERM_REVIEW)
    paragraph = "The state apparatus and its apparatus matter."
    chunk = Chunk(
        index=0, text=paragraph, chapter_title="Chapter", section_title="",
        metadata={
            "paragraph_indices": [0], "source_paragraph_spans": [[0, len(paragraph)]],
            "source_paragraph_hashes": [_sha(paragraph)], "structural_roles": ["body"],
            "chapter_position": 1,
        },
    )
    db.save_chunks("job", [chunk])
    source_hash = db.get_job_source_hash("job")
    db.save_job_artifact("job", "book_term_body_index_v1", {
        "source_sha256": source_hash,
        "entries": [{
            "chunk_index": 0, "local_paragraph": 0, "paragraph_index": 0,
            "paragraph_sha256": _sha(paragraph), "chapter_position": 1,
        }],
    })
    db.save_job_artifact("job", "book_term_review_v1", {
        "phase": "awaiting", "review_version": 2, "source_sha256": source_hash,
        "proposals": [{
            "source": "apparatus", "target": "", "status": "proposed",
            "source_evidence_sha256": _sha(paragraph),
        }],
    })
    config = TarjomehConfig()
    config.glossary.path = str(server_glossary)
    monkeypatch.setenv("UI_SECRET_TOKEN", "test-token")
    app = create_app(config)
    app.config["TESTING"] = True
    route = "/api/jobs/job/book-term-review"
    headers = {"Authorization": "Bearer test-token"}
    with patch("tarjomeh.jobs.database.JobDatabase", return_value=db):
        added = app.test_client().post(route, headers=headers, json={
            "action": "add", "source": "state apparatus", "target": "دستگاه دولتی",
        })
        decided = app.test_client().post(route, headers=headers, json={
            "action": "decide", "decisions": [{
                "index": 0, "status": "approved", "target": "دستگاه",
                "scope_mode": "evidence_paragraph",
            }],
        })
    assert added.status_code == 409
    assert decided.status_code == 409
