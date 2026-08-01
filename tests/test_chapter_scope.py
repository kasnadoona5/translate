from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from tarjomeh.chunking.chunker import SemanticChunker
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    TranslationPipeline,
    apply_chapter_selection,
    build_chapter_manifest,
)
from tarjomeh.jobs.database import JobDatabase, JobStatus
from tarjomeh.parsers.base import Chapter, Document, Paragraph, Section


def _document() -> Document:
    return Document(
        title="Test Book",
        chapters=[
            Chapter(
                title="Chapter 1: Opening",
                number=1,
                sections=[Section("", 2, [Paragraph("First chapter text.")])],
                metadata={"start_page": 1, "end_page": 4},
            ),
            Chapter(
                title="Chapter 2: Argument",
                number=2,
                sections=[Section("", 2, [Paragraph("Second chapter text.")])],
                metadata={"start_page": 5, "end_page": 9},
            ),
        ],
    )


class TestChapterStructure(unittest.TestCase):
    def test_manifest_and_chunks_preserve_original_chapter_identity(self) -> None:
        document = _document()
        manifest = build_chapter_manifest(document)
        selected = apply_chapter_selection(document, [2])
        chunks = SemanticChunker(max_tokens=100).chunk(selected)

        self.assertEqual([item["position"] for item in manifest], [1, 2])
        self.assertEqual(len(document.chapters), 2)
        self.assertEqual(manifest[1]["start_page"], 5)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["chapter_position"], 2)
        self.assertEqual(chunks[0].metadata["chapter_number"], 2)

    def test_chunk_metadata_is_persisted_and_loaded(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            db = JobDatabase(Path(temp_dir) / "jobs.db")
            db.create_job("chapter-meta", "book.txt", {})
            chunks = SemanticChunker(max_tokens=100).chunk(_document())
            db.save_chunks("chapter-meta", chunks)

            loaded = db.get_chunks("chapter-meta")

            self.assertEqual(loaded[0]["metadata"]["chapter_position"], 1)
            self.assertEqual(loaded[1]["metadata"]["chapter_position"], 2)


class TestChapterPipeline(unittest.TestCase):
    def _config(self) -> TarjomehConfig:
        config = TarjomehConfig()
        config.output.format = "txt"
        config.output.bilingual_mode = "target_only"
        config.translation.enable_critique = False
        config.translation.enable_back_translation = False
        config.translation.enable_web_context = False
        config.translation.enable_book_research = False
        config.glossary.enable_auto_extraction = False
        config.memory.enable_4layer = False
        return config

    @staticmethod
    def _fake_translation(**kwargs: object) -> str:
        chunk = kwargs["chunk"]
        return "FA " + chunk.text

    def test_selected_chapter_is_a_complete_scoped_job(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            root = Path(temp_dir)
            source = root / "book.txt"
            source.write_text(
                "Chapter 1: Opening\nFirst chapter text.\n\n"
                "Chapter 2: Argument\nSecond chapter text.",
                encoding="utf-8",
            )
            output = root / "selected.txt"
            config = self._config()
            config.translation.chapter_selection = [2]
            pipeline = TranslationPipeline(config)
            pipeline.db = JobDatabase(root / "jobs.db")
            pipeline._translate_single_chunk = MagicMock(
                side_effect=self._fake_translation
            )

            pipeline.run(source, output, job_id="selected-job")

            job = pipeline.db.get_job("selected-job")
            self.assertEqual(job["raw_status"], JobStatus.COMPLETED)
            self.assertEqual(len(pipeline.db.get_chunks("selected-job")), 1)
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("Second chapter text", rendered)
            self.assertNotIn("First chapter text", rendered)

    def test_checkpoint_exports_prefix_then_resume_finishes_book(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            root = Path(temp_dir)
            source = root / "book.txt"
            source.write_text(
                "Chapter 1: Opening\nFirst chapter text.\n\n"
                "Chapter 2: Argument\nSecond chapter text.",
                encoding="utf-8",
            )
            output = root / "translated.txt"
            config = self._config()
            config.translation.stop_after_chapter = 1
            pipeline = TranslationPipeline(config)
            pipeline.db = JobDatabase(root / "jobs.db")
            pipeline._translate_single_chunk = MagicMock(
                side_effect=self._fake_translation
            )

            pipeline.run(source, output, job_id="checkpoint-job")

            paused = pipeline.db.get_job("checkpoint-job")
            self.assertEqual(paused["raw_status"], JobStatus.PAUSED)
            preview = output.read_text(encoding="utf-8")
            self.assertIn("First chapter text", preview)
            self.assertNotIn("Second chapter text", preview)

            pipeline.run(source, output, job_id="checkpoint-job")

            completed = pipeline.db.get_job("checkpoint-job")
            self.assertEqual(completed["raw_status"], JobStatus.COMPLETED)
            final = output.read_text(encoding="utf-8")
            self.assertIn("First chapter text", final)
            self.assertIn("Second chapter text", final)
            checkpoints = pipeline.db.get_job_artifact(
                "checkpoint-job", "chapter_checkpoints"
            )
            self.assertEqual(checkpoints["reached_positions"], [1])


if __name__ == "__main__":
    unittest.main()
