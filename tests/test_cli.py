"""Unit tests for Phase 6: CLI Interface."""

from __future__ import annotations

import argparse
import unittest
from unittest.mock import MagicMock, patch

from tarjomeh.cli.main import setup_logging, cmd_translate


class TestCLI(unittest.TestCase):
    """Test CLI commands and options parsing."""

    def test_setup_logging(self) -> None:
        try:
            setup_logging(verbose=True)
            setup_logging(verbose=False)
        except Exception as e:
            self.fail(f"setup_logging raised exception: {e}")

    @patch("tarjomeh.core.pipeline.TranslationPipeline")
    @patch("tarjomeh.core.config.TarjomehConfig")
    @patch("tarjomeh.cli.main.console")
    def test_cmd_translate_success(
        self, mock_console: MagicMock, mock_config_cls: MagicMock, mock_pipeline_cls: MagicMock
    ) -> None:
        # Mock CLI args
        args = argparse.Namespace(
            file="pyproject.toml",  # File that exists
            config=None,
            mode="fast",
            output_format="txt",
            bilingual="target_only",
            provider="openrouter",
            model="dummy-model",
            output=None,
            resume=None,
        )

        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run.return_value = MagicMock()

        # Call CLI translate function
        ret = cmd_translate(args)
        self.assertEqual(ret, 0)
        mock_pipeline.run.assert_called_once()

    def test_cmd_translate_file_not_found(self) -> None:
        args = argparse.Namespace(file="non_existent_file.pdf")
        ret = cmd_translate(args)
        self.assertEqual(ret, 1)


if __name__ == "__main__":
    unittest.main()
