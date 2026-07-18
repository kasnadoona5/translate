"""Unit tests for Phase 2: Core translation engine."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import LLMClient
from tarjomeh.core.pipeline import TranslationPipeline, PipelineResult
from tarjomeh.jobs.database import JobDatabase, JobStatus, ChunkStatus


class TestTarjomehConfig(unittest.TestCase):
    """Test config loading, presets, and CLI override merging."""

    def test_default_config(self) -> None:
        config = TarjomehConfig()
        self.assertEqual(config.translation.mode, "academic")
        self.assertEqual(config.translation.country, "Iran")
        self.assertEqual(config.chunking.max_chunk_tokens, 1000)

    def test_overrides(self) -> None:
        config = TarjomehConfig()
        config.llm.openrouter.api_keys = ["dummy"]
        config.update_from_overrides({
            "translation.mode": "fast",
            "llm.openrouter.site_url": "https://test.local"
        })
        self.assertEqual(config.translation.mode, "fast")
        self.assertEqual(config.llm.openrouter.site_url, "https://test.local")
        
        # In fast mode, chunk size should update accordingly
        self.assertEqual(config.chunking.max_chunk_tokens, 3000)

    def test_academic_mode_worker_constraint(self) -> None:
        config = TarjomehConfig()
        config.llm.openrouter.api_keys = ["dummy"]
        config.translation.mode = "academic"
        config.translation.parallel_workers = 4
        # Merging/validating should force workers to 1
        config.update_from_overrides({})
        self.assertEqual(config.translation.parallel_workers, 1)

    def test_glossary_paths_from_dict(self) -> None:
        config = TarjomehConfig.from_dict({
            "llm": {"openrouter": {"api_keys": ["dummy"]}},
            "glossary": {
                "path": "glossary/base.csv",
                "paths": ["glossary/extra.csv"],
            },
        })

        self.assertEqual(config.glossary.path, "glossary/base.csv")
        self.assertEqual(config.glossary.paths, ["glossary/extra.csv"])


class TestLLMClient(unittest.TestCase):
    """Test LLM Client token counting, key rotation, and request formatting."""

    def setUp(self) -> None:
        self.config = TarjomehConfig()
        self.config.llm.openrouter.api_keys = ["key1", "key2", "  "]
        self.client = LLMClient(self.config)

    def tearDown(self) -> None:
        self.client.close()

    def test_token_counting(self) -> None:
        text = "Hello, world! This is a test."
        tokens = self.client.count_tokens(text)
        self.assertGreater(tokens, 0)

    def test_key_rotation(self) -> None:
        # Rotation should cycle between non-empty keys
        key_a = self.client._get_next_api_key()
        key_b = self.client._get_next_api_key()
        key_c = self.client._get_next_api_key()

        self.assertEqual(key_a, "key1")
        self.assertEqual(key_b, "key2")
        self.assertEqual(key_c, "key1")  # back to start

    def test_prepare_request_openrouter(self) -> None:
        self.config.llm.provider = "openrouter"
        self.config.llm.model = "anthropic/claude-3"
        
        messages = [{"role": "user", "content": "Hello"}]
        url, headers, payload = self.client._prepare_request(messages, system_prompt="Sys")
        
        self.assertEqual(url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertIn("Authorization", headers)
        self.assertEqual(payload["model"], "anthropic/claude-3")
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "Sys")


class TestTranslationPipeline(unittest.TestCase):
    """Test translation pipeline flow and state machine integration."""

    @patch("tarjomeh.core.pipeline.JobDatabase")
    @patch("tarjomeh.core.pipeline.LLMClient")
    def test_pipeline_run_txt(self, mock_llm_cls: MagicMock, mock_db_cls: MagicMock) -> None:
        # Mock database
        mock_db = mock_db_cls.return_value
        mock_db.get_job.return_value = None
        mock_db.get_chunk_summary.return_value = {"total": 0, "completed": 0, "pending": 0, "errors": 0}

        # Mock LLM Client
        mock_llm = mock_llm_cls.return_value
        mock_llm.count_tokens.return_value = 10
        mock_llm.complete.return_value = "ترجمه تست"

        # Create temporary sample text file
        temp_file = Path("tests_sample_chapter.txt")
        with temp_file.open("w", encoding="utf-8") as f:
            f.write("Chapter 1: The State\nThis is a paragraph about political theory.")

        config = TarjomehConfig()
        config.translation.mode = "fast"
        config.translation.enable_critique = False
        config.translation.enable_web_context = False
        config.translation.enable_back_translation = False
        config.glossary.enable_auto_extraction = False
        config.output.format = "txt"
        
        pipeline = TranslationPipeline(config)
        pipeline.llm_client = mock_llm
        pipeline.db = mock_db

        output_file = Path("tests_sample_translated.txt")
        try:
            result = pipeline.run(input_path=temp_file, output_path=output_file)
            
            self.assertIsInstance(result, PipelineResult)
            self.assertEqual(result.total_chunks, 1)
            self.assertTrue(output_file.exists())
            event_types = [
                call.args[2]
                for call in mock_db.log_chunk_event.call_args_list
                if len(call.args) >= 3
            ]
            self.assertIn("chunk_started", event_types)
            self.assertIn("translation_completed", event_types)
            self.assertIn("critique_skipped", event_types)
            self.assertIn("back_translation_skipped", event_types)
            self.assertIn("chunk_completed", event_types)
            
            # Check content of output
            with output_file.open("r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("ترجمه تست", content)

        finally:
            # Clean up files
            if temp_file.exists():
                temp_file.unlink()
            if output_file.exists():
                output_file.unlink()

    @patch("tarjomeh.core.pipeline.JobDatabase")
    @patch("tarjomeh.core.pipeline.LLMClient")
    def test_pipeline_does_not_export_when_any_chunk_fails(
        self,
        mock_llm_cls: MagicMock,
        mock_db_cls: MagicMock,
    ) -> None:
        mock_db = mock_db_cls.return_value
        mock_db.get_job.return_value = None
        mock_db.get_chunk_summary.return_value = {"total": 0, "completed": 0, "pending": 0, "errors": 0}

        mock_llm = mock_llm_cls.return_value
        mock_llm.count_tokens.side_effect = lambda text: len(text.split())
        mock_llm.complete.side_effect = [
            "ترجمه اول",
            RuntimeError("malformed router response"),
            "must not translate after the memory gap",
        ]

        temp_file = Path("tests_two_chunk_source.txt")
        temp_file.write_text(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa "
            "lambda mu nu xi omicron",
            encoding="utf-8",
        )

        output_file = Path("tests_should_not_exist.txt")
        config = TarjomehConfig()
        config.chunking.strategy = "fixed"
        config.chunking.max_chunk_tokens = 5
        config.translation.enable_critique = False
        config.translation.enable_web_context = False
        config.translation.enable_back_translation = False
        config.glossary.enable_auto_extraction = False
        config.output.format = "txt"

        pipeline = TranslationPipeline(config)
        pipeline.llm_client = mock_llm
        pipeline.db = mock_db

        try:
            with self.assertRaises(RuntimeError) as ctx:
                pipeline.run(input_path=temp_file, output_path=output_file)

            self.assertIn("paused at chunk 1", str(ctx.exception))
            self.assertIn("later chunks do not advance", str(ctx.exception))
            self.assertFalse(output_file.exists())
            self.assertEqual(mock_llm.complete.call_count, 2)
            self.assertTrue(
                any(
                    call.args and call.args[1] == JobStatus.PAUSED_ERROR
                    for call in mock_db.update_job_status.call_args_list
                )
            )
            mock_db.log_event.assert_any_call(
                pipeline.current_job_id,
                "ERROR",
                "Chunk 1 failed: RuntimeError: malformed router response",
            )
        finally:
            if temp_file.exists():
                temp_file.unlink()
            if output_file.exists():
                output_file.unlink()


if __name__ == "__main__":
    unittest.main()
