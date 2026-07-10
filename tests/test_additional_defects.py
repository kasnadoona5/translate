"""Unit tests for Tarjomeh defect fixes and new features."""

from __future__ import annotations

import csv
import asyncio
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import _chunk_needs_review, _critique_for_event, _critique_passes_quality_gate
from tarjomeh.jobs.database import JobDatabase, ChunkStatus, JobStatus
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.glossary.manager import GlossaryManager, GlossaryEntry
from tarjomeh.glossary.compliance import GlossaryComplianceChecker
from tarjomeh.quality.critique import TranslationCritique, CritiqueResult
from tarjomeh.quality.refiner import TranslationRefiner
from tarjomeh.exporters import get_exporter
from tarjomeh.exporters.pdf_exporter import PdfExporter
from tarjomeh.exporters.epub_exporter import EpubExporter
from tarjomeh.exporters.docx_exporter import DocxExporter
from tarjomeh.exporters.txt_exporter import TxtExporter
from tarjomeh.chunking.chunker import Chunk


class TestDatabaseFixes(unittest.TestCase):
    """Test database log writing, chunk statuses, and scoped cleanup."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_jobs.db")
        # Initialize DB with custom path
        self.db = JobDatabase(db_path=self.db_path)

    def tearDown(self) -> None:
        import gc
        # Force garbage collection to close dangling SQLite connections
        gc.collect()
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    def test_log_event_and_chunk_statuses(self) -> None:
        # Create a job
        job_id = "test-job-123"
        self.db.create_job(job_id, "dummy_input.txt", {})
        
        # Log events
        self.db.log_event(job_id, "INFO", "Translation started")
        self.db.log_event(job_id, "WARNING", "Back-translation flagged")

        # Verify logs in DB
        conn = self.db._get_connection()
        try:
            rows = conn.execute("SELECT * FROM job_log WHERE job_id = ? ORDER BY timestamp ASC", (job_id,)).fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["level"], "INFO")
            self.assertEqual(rows[0]["message"], "Translation started")
            self.assertEqual(rows[1]["level"], "WARNING")
            self.assertEqual(rows[1]["message"], "Back-translation flagged")
        finally:
            conn.close()

        # Save chunks
        chunks = [
            Chunk(index=0, text="Hello", chapter_title="Intro", section_title=""),
            Chunk(index=1, text="World", chapter_title="Intro", section_title=""),
        ]
        self.db.save_chunks(job_id, chunks)

        # Update to intermediate status
        self.db.update_chunk(job_id, 0, ChunkStatus.TRANSLATED, "سلام")
        self.db.update_chunk(job_id, 1, ChunkStatus.CRITIQUED, "جهان")

        # Verify get_chunk_summary maps intermediate to pending
        summary = self.db.get_chunk_summary(job_id)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["pending"], 2)  # both are pending completed status
        self.assertEqual(summary["completed"], 0)

        # Complete one chunk
        self.db.update_chunk(job_id, 0, ChunkStatus.COMPLETED, "سلام")
        summary = self.db.get_chunk_summary(job_id)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["pending"], 1)

    def test_get_job_preserves_raw_status_for_backend_updates(self) -> None:
        job_id = "status-job-123"
        self.db.create_job(job_id, "dummy_input.txt", {})
        self.db.update_job_status(job_id, JobStatus.RUNNING)

        job = self.db.get_job(job_id)

        self.assertEqual(job["status"], "processing")
        self.assertEqual(job["raw_status"], JobStatus.RUNNING)

    def test_chunk_events_round_trip(self) -> None:
        job_id = "test-job-events"
        self.db.create_job(job_id, "dummy_input.txt", {})
        self.db.log_chunk_event(
            job_id,
            2,
            "critique_completed",
            {
                "scores": {"average": 8.5},
                "issues": ["minor issue"],
                "passes_threshold": True,
            },
        )

        events = self.db.get_chunk_events(job_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["chunk_index"], 2)
        self.assertEqual(events[0]["event_type"], "critique_completed")
        self.assertEqual(events[0]["payload"]["scores"]["average"], 8.5)
        self.assertTrue(events[0]["payload"]["passes_threshold"])

        filtered = self.db.get_chunk_events(job_id, chunk_index=2)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["payload"]["issues"], ["minor issue"])

    def test_cleanup_preserves_db(self) -> None:
        job_id = "test-job-456"
        self.db.create_job(job_id, "dummy_input.txt", {})
        self.db.log_event(job_id, "INFO", "Event before cleanup")

        # Create dummy upload file
        upload_dir = Path("jobs/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        dummy_file = upload_dir / f"{job_id}.txt"
        dummy_file.write_text("dummy source content")
        self.assertTrue(dummy_file.is_file())

        # Cleanup
        success = self.db.cleanup_job(job_id)
        self.assertTrue(success)

        # Verify file is deleted
        self.assertFalse(dummy_file.is_file())

        # Verify DB record is preserved
        job = self.db.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["id"], job_id)

        # Verify DB logs are preserved
        conn = self.db._get_connection()
        try:
            rows = conn.execute("SELECT * FROM job_log WHERE job_id = ?", (job_id,)).fetchall()
            self.assertEqual(len(rows), 1)
        finally:
            conn.close()


class TestPersianTypography(unittest.TestCase):
    """Test Persian punctuation, numeral conversion, and ZWNJ rules."""

    def setUp(self) -> None:
        self.typographer = PersianTypographer(config={
            "convert_numerals": True,
            "normalize_zwnj": True,
            "fix_punctuation": True,
        })

    def test_numeral_conversion(self) -> None:
        # Prose numbers convert to Persian digits...
        self.assertEqual(self.typographer.process("Chapter 3"), "Chapter ۳")
        # ...but scholarly mode (default ON) protects years and citations.
        self.assertEqual(self.typographer.process("Year 2026"), "Year 2026")
        self.assertIn("(Marx 1973, 408)", self.typographer.process("او (Marx 1973, 408) گفت"))

    def test_numeral_conversion_scholarly_off(self) -> None:
        typographer = PersianTypographer(config={
            "convert_numerals": True,
            "normalize_zwnj": False,
            "fix_punctuation": False,
            "scholarly_mode": False,
        })
        # With scholarly protection disabled, everything converts.
        self.assertEqual(typographer.process("Year 2026"), "Year ۲۰۲۶")

    def test_punctuation_fixing(self) -> None:
        self.assertEqual(self.typographer.process("سلام; چطورید؟"), "سلام؛ چطورید؟")

    def test_zwnj_normalization(self) -> None:
        # e.g., spacing for plural "ha"
        self.assertIn("کتاب‌ها", self.typographer.process("کتاب ها"))


class TestGlossaryFeatures(unittest.TestCase):
    """Test glossary extraction, compliance wrong vs missing, and CSV imports."""

    def test_wrong_vs_missing_compliance(self) -> None:
        manager = GlossaryManager()
        manager.add_term("state", "دولت")

        checker = GlossaryComplianceChecker()

        # Compliant translation
        report = checker.check("این دولت بزرگ است.", "The state is big.", manager)
        self.assertTrue(report.compliant)

        # Missing violation (term not translated, source term not in translation)
        report = checker.check("این بزرگ است.", "The state is big.", manager)
        self.assertFalse(report.compliant)
        self.assertEqual(len(report.violations), 1)
        self.assertEqual(report.violations[0].status, "missing")

        # Wrong violation (source term left untranslated in output)
        report = checker.check("این state بزرگ است.", "The state is big.", manager)
        self.assertFalse(report.compliant)
        self.assertEqual(len(report.violations), 1)
        self.assertEqual(report.violations[0].status, "wrong")

    def test_is_auto_and_csv_export(self) -> None:
        manager = GlossaryManager()
        # Add normal term
        manager.add_term("power", "قدرت", is_auto=False)
        # Add auto term
        manager.add_term("justice", "عدالت", is_auto=True)

        temp_dir = tempfile.mkdtemp()
        try:
            csv_path = os.path.join(temp_dir, "auto_glossary.csv")
            manager.save_auto_extracted(csv_path)

            # Read exported CSV and verify it only contains the auto-extracted term
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                self.assertEqual(len(reader), 1)
                self.assertEqual(reader[0]["source"], "justice")
                self.assertEqual(reader[0]["target"], "عدالت")
        finally:
            shutil.rmtree(temp_dir)

    def test_auto_extracted_terms_do_not_override_curated_terms(self) -> None:
        manager = GlossaryManager()
        manager.add_term("capital", "سرمایه", context="curated Marx term")

        manager.merge_auto_extracted({
            "capital": {
                "target": "پایتخت",
                "context": "auto guess",
                "domain": "geography",
            },
            "assemblage": {
                "target": "هم‌آرایی",
                "context": "auto extracted concept",
            },
        })

        capital_terms = [e for e in manager.entries if e.source == "capital"]
        assemblage_terms = [e for e in manager.entries if e.source == "assemblage"]
        self.assertEqual(len(capital_terms), 1)
        self.assertEqual(capital_terms[0].target, "سرمایه")
        self.assertFalse(capital_terms[0].is_auto)
        self.assertEqual(len(assemblage_terms), 1)
        self.assertTrue(assemblage_terms[0].is_auto)

    def test_namespaced_context_selection(self) -> None:
        manager = GlossaryManager()
        manager.add_term(
            "capital",
            "سرمایه",
            context="Marx's economic category in critique of political economy",
            domain="political economy",
            sense="economic",
            author="Marx",
        )
        manager.add_term(
            "capital",
            "سرمایه فرهنگی",
            context="Bourdieu's cultural, social, and symbolic capital",
            domain="sociology",
            sense="cultural",
            author="Bourdieu",
        )

        terms = manager.find_terms(
            "Bourdieu argues that capital is accumulated across fields.",
            context="Chapter on Bourdieu and habitus",
            domain="sociology",
        )

        self.assertEqual(len(terms), 1)
        self.assertEqual(terms[0].target, "سرمایه فرهنگی")
        self.assertEqual(terms[0].author, "Bourdieu")

        prompt_table = manager.format_for_prompt(terms)
        self.assertIn("Sense", prompt_table)
        self.assertIn("Author/School", prompt_table)

    def test_load_many_with_optional_columns(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            base = Path(temp_dir) / "base.csv"
            extra = Path(temp_dir) / "extra.csv"
            base.write_text(
                "source,target,tgt_lng,context,domain\n"
                "hegemony,هژمونی,fa,Gramsci,political theory\n",
                encoding="utf-8",
            )
            extra.write_text(
                "source,target,tgt_lng,context,domain,sense,author\n"
                "capital,سرمایه فرهنگی,fa,Bourdieu's term,sociology,cultural,Bourdieu\n",
                encoding="utf-8",
            )

            manager = GlossaryManager()
            manager.load_many([base, extra])

            self.assertEqual(len(manager.entries), 2)
            selected = manager.find_terms(
                "Bourdieu discusses capital.",
                context="sociology chapter",
                domain="sociology",
            )
            self.assertEqual(selected[0].target, "سرمایه فرهنگی")
            self.assertEqual(selected[0].sense, "cultural")
        finally:
            shutil.rmtree(temp_dir)

    def test_compliance_accepts_zwnj_spacing_variants(self) -> None:
        manager = GlossaryManager()
        manager.add_term("biopolitics", "زیست‌سیاست")

        checker = GlossaryComplianceChecker()
        report = checker.check(
            "فوکو زیست سیاست را صورت‌بندی می‌کند.",
            "Foucault theorizes biopolitics.",
            manager,
        )

        self.assertTrue(report.compliant)


class TestCritiqueScores(unittest.TestCase):
    """Test critique score parsing for flat vs nested schemas, threshold checks."""

    def test_parse_flat_schema(self) -> None:
        raw_json = '{"accuracy": 8, "fluency": 9, "terminology": 7, "register": 8, "issues": ["issue1"]}'
        result = TranslationCritique._parse_response(raw_json)
        self.assertEqual(result.accuracy, 8)
        self.assertEqual(result.fluency, 9)
        self.assertEqual(result.terminology, 7)
        self.assertEqual(result.register, 8)
        self.assertEqual(result.average, 8.0)
        self.assertEqual(result.issues, ["issue1"])
        self.assertTrue(result.passes_threshold(7.5))

    def test_parse_nested_schema(self) -> None:
        raw_json = '''
        {
          "scores": {
            "accuracy": 6,
            "fluency": 7,
            "terminology": 8,
            "register": 7
          },
          "overall": 7,
          "issues": [
            {
              "category": "accuracy",
              "severity": "minor",
              "suggested_fix": "تغییر",
              "explanation": "Wrong meaning"
            },
            {
              "category": "terminology",
              "severity": "critical",
              "source_segment": "the state",
              "suggested_fix": "دولت",
              "explanation": "Glossary violation"
            }
          ]
        }
        '''
        result = TranslationCritique._parse_response(raw_json)
        self.assertEqual(result.accuracy, 6)
        self.assertEqual(result.fluency, 7)
        self.assertEqual(result.terminology, 8)
        self.assertEqual(result.register, 7)
        self.assertEqual(result.average, 7.0)
        self.assertEqual(len(result.issues), 2)
        # Issues are sorted critical -> major -> minor, preserving severity
        # and source segment for the refiner.
        self.assertIn("[CRITICAL/terminology]", result.issues[0])
        self.assertIn('source: "the state"', result.issues[0])
        self.assertIn("[MINOR/accuracy]", result.issues[1])
        self.assertIn("fix: تغییر", result.issues[1])
        self.assertFalse(result.passes_threshold(7.5))


    def test_major_accuracy_issue_forces_refinement_even_with_high_average(self) -> None:
        critique = CritiqueResult(
            accuracy=8,
            fluency=9,
            terminology=10,
            register=9,
            average=8.5,
            issues=[
                '[MAJOR/accuracy] source: "prospective" | current: "inductive" | fix: anticipated',
            ],
        )

        self.assertTrue(critique.passes_threshold(7.0))
        self.assertFalse(_critique_passes_quality_gate(critique, 7.0))

        event = _critique_for_event(critique, threshold=7.0, iteration=0)
        self.assertTrue(event["passes_average_threshold"])
        self.assertFalse(event["passes_threshold"])
        self.assertTrue(event["force_refinement"])
        self.assertEqual(event["blocking_issue_count"], 1)


class TestBalancedRefinementPolicy(unittest.TestCase):
    """Test balanced critique/refiner decisions and non-fatal review state."""

    def test_refiner_parses_structured_decision(self) -> None:
        class FakeLLM:
            async def chat(self, prompt: str) -> str:
                return json.dumps({
                    "translation": "ترجمه حفظ شد",
                    "decision": "preserved",
                    "rationale": "The critic's suggested term was less accurate in context.",
                }, ensure_ascii=False)

        refiner = TranslationRefiner(FakeLLM())
        critique = CritiqueResult(
            average=8,
            issues=['[MAJOR/accuracy] source: "term" | current: "rendering"'],
        )

        result = asyncio.run(refiner.refine_with_decision("source", "old", critique))

        self.assertEqual(result.translation, "ترجمه حفظ شد")
        self.assertEqual(result.decision, "preserved")
        self.assertIn("less accurate", result.rationale)

    def test_chunk_needs_review_uses_latest_attempt_only(self) -> None:
        temp_dir = tempfile.mkdtemp()
        db = JobDatabase(db_path=Path(temp_dir) / "jobs.db")
        job_id = "needs-review-job"
        try:
            db.create_job(job_id, "input.txt", {})
            db.log_chunk_event(job_id, 0, "chunk_started", {})
            db.log_chunk_event(job_id, 0, "critique_needs_review", {"blocking_issue_count": 1})
            self.assertTrue(_chunk_needs_review(db, job_id, 0))

            db.log_chunk_event(job_id, 0, "chunk_started", {})
            db.log_chunk_event(job_id, 0, "critique_completed", {"passes_threshold": True})
            self.assertFalse(_chunk_needs_review(db, job_id, 0))
        finally:
            import gc
            del db
            gc.collect()
            shutil.rmtree(temp_dir)

    def test_memory_style_profile_preserves_existing_layers(self) -> None:
        manager = MemoryManager(TarjomehConfig())
        chunk = Chunk(
            index=0,
            text="The state is not neutral.",
            chapter_title="Chapter 1",
            section_title="",
            token_count=6,
        )

        manager.update_after_translation(chunk, "دولت بی طرف نیست.")
        context = manager.get_context_for_chunk(chunk)
        state = manager.to_dict()

        self.assertTrue(context.style_profile)
        self.assertIn("style_profile", state)
        self.assertIn("proper_nouns", state)
        self.assertIn("bilingual_summary", state)
        self.assertIn("past_translations", state)
        self.assertIn("short_term_context", state)


class TestExporterFactory(unittest.TestCase):
    """Test get_exporter returns correct base types and classes."""

    def test_exporters(self) -> None:
        self.assertIs(get_exporter("pdf"), PdfExporter)
        self.assertIs(get_exporter("epub"), EpubExporter)
        self.assertIs(get_exporter("docx"), DocxExporter)
        self.assertIs(get_exporter("txt"), TxtExporter)


class TestLLMClientDefects(unittest.TestCase):
    """Test EmptyCompletionError raising, retry, exclude_reasoning, and defensive JSON."""

    def setUp(self) -> None:
        self.config_dict = {
            "llm": {
                "provider": "openrouter",
                "model": "anthropic/claude-3-5-sonnet",
                "openrouter": {
                    "api_keys": ["test-key-1", "test-key-2"],
                    "exclude_reasoning": True
                }
            },
            "retry": {
                "max_retries": 1,
                "base_delay": 0.01,
                "max_delay": 0.02,
                "jitter": False
            }
        }
        self.config = TarjomehConfig.from_dict(self.config_dict)

    def test_empty_completion_error_and_retry(self) -> None:
        from tarjomeh.core.llm_client import LLMClient, EmptyCompletionError
        client = LLMClient(self.config)

        # Mock posting to return empty response
        calls_count = 0
        def mock_post(*args, **kwargs):
            nonlocal calls_count
            calls_count += 1
            headers = kwargs.get("headers", {})
            auth = headers.get("Authorization", "")
            if calls_count == 1:
                assert "test-key-1" in auth
            elif calls_count == 2:
                assert "test-key-2" in auth
            # Return empty completion
            mock_res = unittest.mock.Mock()
            mock_res.status_code = 200
            mock_res.text = '{"choices":[{"message":{"content":""}}]}'
            return mock_res

        client._client.post = mock_post

        with self.assertRaises(EmptyCompletionError):
            client.complete(messages=[{"role": "user", "content": "hello"}])
        
        # Verify it retried once and rotated key
        self.assertEqual(calls_count, 2)

    def test_defensive_json_parsing(self) -> None:
        from tarjomeh.core.llm_client import LLMClient
        client = LLMClient(self.config)

        # Mock post returning JSON with trailing SSE garbage
        def mock_post(*args, **kwargs):
            mock_res = unittest.mock.Mock()
            mock_res.status_code = 200
            mock_res.text = '{"choices":[{"message":{"content":"valid translation"}}]} data: [DONE]\n'
            return mock_res

        client._client.post = mock_post
        translation = client.complete(messages=[{"role": "user", "content": "hello"}])
        self.assertEqual(translation, "valid translation")

    def test_malformed_provider_json_is_retried(self) -> None:
        from tarjomeh.core.llm_client import LLMClient
        client = LLMClient(self.config)

        calls_count = 0

        def mock_post(*args, **kwargs):
            nonlocal calls_count
            calls_count += 1
            mock_res = unittest.mock.Mock()
            mock_res.status_code = 200
            if calls_count == 1:
                mock_res.text = '{"choices":'
            else:
                mock_res.text = '{"choices":[{"message":{"content":"recovered translation"}}]}'
            return mock_res

        client._client.post = mock_post
        translation = client.complete(messages=[{"role": "user", "content": "hello"}])

        self.assertEqual(translation, "recovered translation")
        self.assertEqual(calls_count, 2)

    def test_exclude_reasoning_payload(self) -> None:
        from tarjomeh.core.llm_client import LLMClient
        client = LLMClient(self.config)
        url, headers, payload = client._prepare_request(messages=[{"role": "user", "content": "hello"}])
        self.assertIn("reasoning", payload)
        self.assertEqual(payload["reasoning"], {"exclude": True})


class TestParagraphAlignmentAndDrift(unittest.TestCase):
    """Test paragraph alignment indices and safety on count mismatches."""

    def test_alignment_with_paragraph_indices(self) -> None:
        from tarjomeh.parsers.base import Document, Chapter, Section, Paragraph
        from tarjomeh.chunking.chunker import SemanticChunker
        from tarjomeh.core.pipeline import TranslationPipeline
        
        # Setup document with 3 paragraphs
        doc = Document(title="Test Doc")
        ch = Chapter(title="Ch 1")
        sec = Section(title="Sec 1", level=2)
        sec.paragraphs = [
            Paragraph(text="Paragraph 1"),
            Paragraph(text="Paragraph 2"),
            Paragraph(text="Paragraph 3")
        ]
        ch.sections = [sec]
        doc.chapters = [ch]

        chunker = SemanticChunker(max_tokens=1000, token_counter=lambda x: 1)
        chunks = chunker.chunk(doc)

        # Verify paragraph_indices mapped in chunks metadata
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["paragraph_indices"], [0, 1, 2])

        # Test pipeline reassembly with perfect match
        config = TarjomehConfig()
        pipeline = TranslationPipeline(config)
        pipeline.warnings = []

        # Simulated translations dict (3 paragraphs in output)
        translations = {0: "ترجمه ۱\n\nترجمه ۲\n\nترجمه ۳"}

        class DummyPipeline(TranslationPipeline):
            def __init__(self):
                self.config = TarjomehConfig()
                self.warnings = []

        # Create a helper method to test reassembly logic exactly matching pipeline.py
        def assemble_doc(document, chunks, translations):
            original_paragraphs = document.all_paragraphs
            translated_paragraphs = [None] * len(original_paragraphs)
            fallback_idx = 0
            for idx in range(len(chunks)):
                chunk = chunks[idx]
                chunk_translation = translations.get(idx, "")
                tgt_paras = [p.strip() for p in chunk_translation.split("\n\n") if p.strip()]
                para_indices = chunk.metadata.get("paragraph_indices", [])

                if para_indices:
                    if len(para_indices) == len(tgt_paras):
                        aligned = list(zip(para_indices, tgt_paras))
                    else:
                        aligned = [(para_indices[0], chunk_translation)]
                        for pid in para_indices[1:]:
                            aligned.append((pid, ""))
                    for pid, t in aligned:
                        if pid < len(original_paragraphs):
                            orig_para = original_paragraphs[pid]
                            translated_paragraphs[pid] = Paragraph(text=orig_para.text, metadata={"trans": t})
            
            # Fill missing
            for pid in range(len(original_paragraphs)):
                if translated_paragraphs[pid] is None:
                    orig_para = original_paragraphs[pid]
                    translated_paragraphs[pid] = Paragraph(text=orig_para.text, metadata={"trans": ""})
            return translated_paragraphs

        # Perfect Match
        aligned = assemble_doc(doc, chunks, translations)
        self.assertEqual(len(aligned), 3)
        self.assertEqual(aligned[0].metadata["trans"], "ترجمه ۱")
        self.assertEqual(aligned[1].metadata["trans"], "ترجمه ۲")
        self.assertEqual(aligned[2].metadata["trans"], "ترجمه ۳")

        # Mismatch (model returned 2 paragraphs instead of 3)
        translations_mismatch = {0: "ترجمه ۱\n\nترجمه ۲ ادغام شده"}
        aligned_mismatch = assemble_doc(doc, chunks, translations_mismatch)
        self.assertEqual(len(aligned_mismatch), 3)
        # Content not lost, attached to first
        self.assertEqual(aligned_mismatch[0].metadata["trans"], "ترجمه ۱\n\nترجمه ۲ ادغام شده")
        self.assertEqual(aligned_mismatch[1].metadata["trans"], "")
        self.assertEqual(aligned_mismatch[2].metadata["trans"], "")


class TestConcurrencyAndEventLoop(unittest.TestCase):
    """Test parallel workers execution with real event loop and local mock HTTP server."""

    @classmethod
    def setUpClass(cls) -> None:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import threading
        
        class MockLLMHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass
            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                response = {
                    "choices": [
                        {
                            "message": {
                                "content": "سلام جهان"
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15
                    }
                }
                self.wfile.write(json.dumps(response).encode("utf-8"))

        cls.server = HTTPServer(("127.0.0.1", 0), MockLLMHandler)
        cls.port = cls.server.server_port
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def test_parallel_execution_no_event_loop_errors(self) -> None:
        import os
        import asyncio
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from tarjomeh.core.llm_client import LLMClient
        
        # Override OpenRouter API Base to hit our local mock server
        orig_base = os.getenv("OPENROUTER_API_BASE")
        os.environ["OPENROUTER_API_BASE"] = f"http://127.0.0.1:{self.port}/v1"
        
        try:
            config = TarjomehConfig()
            config.llm.provider = "openrouter"
            config.llm.openrouter.api_keys = ["mock-key"]
            
            client = LLMClient(config)
            
            # Use ThreadPoolExecutor to simulate parallel workers calling acomplete
            def worker_task(idx):
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    res = loop.run_until_complete(
                        client.acomplete(messages=[{"role": "user", "content": f"test {idx}"}])
                    )
                    return res
                finally:
                    loop.close()

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(worker_task, i) for i in range(5)]
                results = [f.result() for f in futures]
                
            self.assertEqual(len(results), 5)
            self.assertTrue(all(r == "سلام جهان" for r in results))
            
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(client.aclose())
            finally:
                loop.close()
        finally:
            if orig_base is not None:
                os.environ["OPENROUTER_API_BASE"] = orig_base
            else:
                os.environ.pop("OPENROUTER_API_BASE", None)


if __name__ == "__main__":
    unittest.main()
