"""Unit tests for Tarjomeh defect fixes and new features."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.jobs.database import JobDatabase, ChunkStatus, JobStatus
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.glossary.manager import GlossaryManager, GlossaryEntry
from tarjomeh.glossary.compliance import GlossaryComplianceChecker
from tarjomeh.quality.critique import TranslationCritique, CritiqueResult
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
        self.assertEqual(self.typographer.process("Year 2026"), "Year ۲۰۲۶")

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
              "suggested_fix": "تغییر",
              "explanation": "Wrong meaning"
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
        self.assertEqual(len(result.issues), 1)
        self.assertIn("[accuracy] Suggestion: تغییر", result.issues[0])
        self.assertFalse(result.passes_threshold(7.5))


class TestExporterFactory(unittest.TestCase):
    """Test get_exporter returns correct base types and classes."""

    def test_exporters(self) -> None:
        self.assertIs(get_exporter("pdf"), PdfExporter)
        self.assertIs(get_exporter("epub"), EpubExporter)
        self.assertIs(get_exporter("docx"), DocxExporter)
        self.assertIs(get_exporter("txt"), TxtExporter)


if __name__ == "__main__":
    unittest.main()
