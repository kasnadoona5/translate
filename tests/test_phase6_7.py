from __future__ import annotations

import gc
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from tarjomeh.context.book_researcher import BookResearcher
from tarjomeh.context.search_providers import SearchResult
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import _research_context_for_memory
from tarjomeh.core.term_notes import apply_term_notes
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.markdown_exporter import MarkdownExporter
from tarjomeh.exporters.docx_exporter import DocxExporter, HAS_DOCX
from tarjomeh.exporters.epub_exporter import EpubExporter
from tarjomeh.glossary.manager import GlossaryManager
from tarjomeh.jobs.database import JobDatabase
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.persian.typography import PersianTypographer


class TestPhase6TermNotes(unittest.TestCase):
    def setUp(self) -> None:
        self.glossary = GlossaryManager()
        self.glossary.add_term("capital", "سرمایه")
        self.typographer = PersianTypographer({
            "convert_numerals": False,
            "normalize_zwnj": False,
            "fix_punctuation": False,
        })
        self.document = TranslatedDocument(
            title="Test",
            paragraphs=[
                TranslatedParagraph(
                    index=0,
                    source_text="Capital and Gramsci appear here.",
                    translated_text="سرمایه و گرامشی در اینجا ظاهر می‌شوند.",
                ),
                TranslatedParagraph(
                    index=1,
                    source_text="Capital appears again.",
                    translated_text="سرمایه دوباره ظاهر می‌شود.",
                ),
            ],
        )

    def test_defaults_preserve_existing_inline_behaviour(self) -> None:
        config = TarjomehConfig()
        self.assertEqual(config.output.term_notes, "inline")
        self.assertFalse(config.translation.enable_book_research)
        self.assertEqual(
            apply_term_notes(
                self.document,
                self.glossary,
                {"Gramsci": "گرامشی"},
                self.typographer,
                mode="inline",
            ),
            [],
        )
        self.assertNotIn("term_notes", self.document.metadata)

    def test_note_mode_removes_inline_parenthetical_memory_instruction(self) -> None:
        config = TarjomehConfig()
        config.output.term_notes = "footnote"
        memory = MemoryManager(config)
        memory.proper_nouns.add_noun("Gramsci", "گرامشی")
        context = memory.get_context_for_chunk(
            type("ChunkStub", (), {"text": "Gramsci"})()
        ).proper_nouns
        self.assertIn("exporter adds", context)
        self.assertNotIn("add (Gramsci) after", context)

    def test_notes_are_numbered_once_in_document_order(self) -> None:
        notes = apply_term_notes(
            self.document,
            self.glossary,
            {"Gramsci": "گرامشی"},
            self.typographer,
            mode="footnote",
        )
        self.assertEqual(
            [(note["number"], note["original"]) for note in notes],
            [(1, "capital"), (2, "Gramsci")],
        )
        self.assertEqual(
            [ref["number"] for ref in self.document.paragraphs[0].metadata["term_note_refs"]],
            [1, 2],
        )
        self.assertNotIn("term_note_refs", self.document.paragraphs[1].metadata)

    def test_markdown_export_has_native_notes(self) -> None:
        apply_term_notes(
            self.document,
            self.glossary,
            {"Gramsci": "گرامشی"},
            self.typographer,
            mode="footnote",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book.md"
            MarkdownExporter().export(self.document, path)
            content = path.read_text(encoding="utf-8")
        self.assertIn("سرمایه[^1]", content)
        self.assertIn("گرامشی[^2]", content)
        self.assertIn("[^1]: capital (سرمایه)", content)
        self.assertEqual(content.count("[^1]:"), 1)

    def test_epub_export_has_linked_footnotes(self) -> None:
        apply_term_notes(
            self.document,
            self.glossary,
            {"Gramsci": "گرامشی"},
            self.typographer,
            mode="footnote",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book.epub"
            EpubExporter().export(self.document, path)
            with zipfile.ZipFile(path) as archive:
                chapter = archive.read("OEBPS/chapter_1.xhtml").decode()
                notes = archive.read("OEBPS/notes.xhtml").decode()
        self.assertIn('epub:type="noteref"', chapter)
        self.assertIn('id="term-note-1"', notes)
        self.assertIn("capital", notes)

    @unittest.skipUnless(HAS_DOCX, "python-docx is not installed")
    def test_docx_export_uses_reliable_endnotes_section(self) -> None:
        import docx

        apply_term_notes(
            self.document,
            self.glossary,
            {"Gramsci": "گرامشی"},
            self.typographer,
            mode="footnote",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book.docx"
            DocxExporter().export(self.document, path)
            text = chr(10).join(p.text for p in docx.Document(path).paragraphs)
        self.assertIn("یادداشت‌ها", text)
        self.assertIn("capital", text)


class TestPhase7Persistence(unittest.TestCase):
    def test_unapproved_research_is_explicitly_non_mandatory_context(self) -> None:
        context = _research_context_for_memory({
            "status": "completed",
            "book_context": "Book context",
            "terms": [
                {
                    "source": "capital",
                    "target": "پایتخت",
                    "status": "suggested",
                },
                {
                    "source": "state",
                    "target": "کشور",
                    "status": "rejected",
                },
            ],
        })
        self.assertIn("not mandatory terminology", context)
        self.assertIn("capital -> پایتخت", context)
        self.assertNotIn("state -> کشور", context)

    def test_job_artifact_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = JobDatabase(Path(tmp) / "jobs.db")
            db.create_job("job-1", "book.pdf", {})
            payload = {
                "status": "completed",
                "terms": [{"source": "capital", "target": "سرمایه"}],
            }
            db.save_job_artifact("job-1", "book_research", payload)
            self.assertEqual(
                db.get_job_artifact("job-1", "book_research"),
                payload,
            )
            del db
            gc.collect()

    def test_book_context_memory_is_additive_and_serialized(self) -> None:
        memory = MemoryManager(TarjomehConfig())
        memory.book_context = "Political theory context"
        state = memory.to_dict()
        self.assertIn("proper_nouns", state)
        self.assertIn("bilingual_summary", state)
        self.assertIn("past_translations", state)
        self.assertIn("short_term_context", state)

        restored = MemoryManager(TarjomehConfig())
        restored.from_dict(state)
        self.assertEqual(restored.book_context, "Political theory context")
        self.assertIn(
            "Political theory context",
            restored.get_context_for_chunk(
                type("ChunkStub", (), {"text": ""})()
            ).format(),
        )


class TestBookResearcher(unittest.IsolatedAsyncioTestCase):
    async def test_research_is_bounded_reviewable_and_source_checked(self) -> None:
        config = TarjomehConfig()
        llm = MagicMock()
        llm.chat = AsyncMock(return_value=json.dumps({
            "book_context": "A study of political operations.",
            "terms": [{
                "source": "operation",
                "target": "عملیات",
                "reason": "Central book concept",
                "confidence": "high",
                "source_urls": [
                    "https://example.test/book",
                    "https://invented.test/not-allowed",
                ],
            }],
        }))
        researcher = BookResearcher(config, llm)
        researcher.provider = MagicMock()
        researcher.provider.search = AsyncMock(return_value=[
            SearchResult(
                title="Book page",
                url="https://example.test/book",
                snippet="A book about operations.",
            )
        ])
        document = type("DocumentStub", (), {
            "title": "The Politics of Operations",
            "author": "Test Author",
            "raw_toc": ["Introduction"],
            "all_paragraphs": [
                type("ParagraphStub", (), {"text": "Operations shape politics."})()
            ],
        })()

        result = await researcher.research(document)

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.terms), 1)
        self.assertTrue(result.terms[0]["is_auto"])
        self.assertEqual(result.terms[0]["status"], "suggested")
        self.assertEqual(
            result.terms[0]["source_urls"],
            ["https://example.test/book"],
        )
        self.assertEqual(researcher.provider.search.await_count, 3)
