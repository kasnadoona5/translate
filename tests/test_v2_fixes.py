"""Regression tests for v0.2.0 fixes.

Covers:
- Phase 0.2: proportional intra-chunk translation distribution
- Phase 0.3: duplicate paragraph-index append (FixedChunker sub-chunks)
- Phase 1:  independent critic model client construction
- PDF parser: line→paragraph merging, dehyphenation, continuation stitching
"""

from __future__ import annotations

import os
import unittest

from tarjomeh.core.pipeline import (
    TranslationPipeline,
    _align_chunk_translation,
    _distribute_translation,
    _split_sentences_fa,
)
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.parsers.base import Paragraph


_BLOB = (
    "جملهٔ اول است. جملهٔ دوم است؟ جملهٔ سوم است! "
    "جملهٔ چهارم است. جملهٔ پنجم است. جملهٔ ششم و پایانی است."
)


class TestTranslationDistribution(unittest.TestCase):
    """Phase 0.2 — mismatched paragraph counts are redistributed, never dumped."""

    def test_no_content_lost_and_no_blank_parts(self) -> None:
        parts = _distribute_translation(_BLOB, 3, [100, 100, 100])
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(p.strip() for p in parts))
        joined = " ".join(parts)
        for sentence in _split_sentences_fa(_BLOB):
            self.assertIn(sentence, joined)

    def test_proportional_to_source_weights(self) -> None:
        even = _distribute_translation(_BLOB, 3, [100, 100, 100])
        skewed = _distribute_translation(_BLOB, 3, [600, 100, 100])
        # A much heavier first source paragraph gets a larger first part.
        self.assertGreater(len(skewed[0]), len(even[0]))

    def test_single_sentence_cannot_be_split(self) -> None:
        parts = _distribute_translation("تنها یک جمله.", 3, [1, 1, 1])
        self.assertEqual(parts, ["تنها یک جمله.", "", ""])

    def test_single_part_passthrough(self) -> None:
        self.assertEqual(_distribute_translation(_BLOB, 1, [1]), [_BLOB])

    def test_bad_weights_fall_back_to_uniform(self) -> None:
        parts = _distribute_translation(_BLOB, 3, [0, 0])  # wrong length + zero sum
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(p.strip() for p in parts))

    def test_heading_omission_does_not_shift_body_into_heading(self) -> None:
        paragraphs = [
            Paragraph("Introduction", metadata={"heading_level": 3}),
            Paragraph("First body paragraph."),
            Paragraph("Second body paragraph."),
        ]

        aligned = _align_chunk_translation(
            original_paragraphs=paragraphs,
            para_indices=[0, 1, 2],
            tgt_paras=["ترجمه بدنه اول.", "ترجمه بدنه دوم."],
            chunk_translation="ترجمه بدنه اول.\n\nترجمه بدنه دوم.",
        )

        self.assertEqual(aligned[0], (0, "مقدمه"))
        self.assertEqual(aligned[1], (1, "ترجمه بدنه اول."))
        self.assertEqual(aligned[2], (2, "ترجمه بدنه دوم."))

    def test_heading_candidate_is_used_when_model_translates_it(self) -> None:
        paragraphs = [
            Paragraph("Introduction", metadata={"heading_level": 3}),
            Paragraph("Body paragraph."),
        ]

        aligned = _align_chunk_translation(
            original_paragraphs=paragraphs,
            para_indices=[0, 1],
            tgt_paras=["درآمد", "ترجمه بدنه."],
            chunk_translation="درآمد\n\nترجمه بدنه.",
        )

        self.assertEqual(aligned, [(0, "درآمد"), (1, "ترجمه بدنه.")])


class TestCriticClient(unittest.TestCase):
    """Phase 1 — independent judge model for critique / back-translation QA."""

    def setUp(self) -> None:
        os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

    def _base_config(self) -> TarjomehConfig:
        cfg = TarjomehConfig()
        cfg.llm.openrouter.api_keys = ["test-key"]
        return cfg

    def test_falls_back_to_translator_when_not_configured(self) -> None:
        pipeline = TranslationPipeline(self._base_config())
        self.assertIs(pipeline.critic_client, pipeline.llm_client)

    def test_separate_client_when_model_set(self) -> None:
        cfg = self._base_config()
        cfg.llm.critic.model = "anthropic/claude-opus-4-8"
        pipeline = TranslationPipeline(cfg)
        self.assertIsNot(pipeline.critic_client, pipeline.llm_client)
        self.assertEqual(
            pipeline.critic_client.config.llm.model, "anthropic/claude-opus-4-8"
        )
        # Judge runs near-deterministic; translator model is untouched.
        self.assertEqual(pipeline.critic_client.config.llm.temperature, 0.0)
        self.assertNotEqual(pipeline.config.llm.model, "anthropic/claude-opus-4-8")
        # API keys are inherited from the main [llm.openrouter] block.
        self.assertEqual(
            pipeline.critic_client.config.llm.openrouter.api_keys, ["test-key"]
        )

    def test_invalid_critic_provider_rejected(self) -> None:
        cfg = self._base_config()
        cfg.llm.critic.provider = "banana"
        with self.assertRaises(ValueError):
            cfg.validate()


class TestSimpleEnvOverrides(unittest.TestCase):
    """Flat TRANSLATOR_* / CRITIC_* .env variables configure everything."""

    _ENV = {
        "TRANSLATOR_API_BASE": "http://172.17.0.1:20128/v1",
        "TRANSLATOR_API_KEY": "nine-router-key",
        "TRANSLATOR_MODEL": "combo1",
        "TRANSLATOR_MAX_TOKENS": "16000",
        "CRITIC_ENABLED": "true",
        "CRITIC_API_BASE": "https://openrouter.ai/api/v1",
        "CRITIC_API_KEY": "sk-or-judge-key",
        "CRITIC_MODEL": "anthropic/claude-sonnet-5",
    }

    def test_env_overrides_apply(self) -> None:
        from unittest.mock import patch
        with patch.dict(os.environ, self._ENV, clear=False):
            cfg = TarjomehConfig()
            cfg._apply_env_overrides()
        self.assertEqual(cfg.llm.model, "combo1")
        self.assertEqual(cfg.llm.openrouter.api_keys, ["nine-router-key"])
        self.assertEqual(cfg.llm.openrouter.api_base, "http://172.17.0.1:20128/v1")
        self.assertEqual(cfg.llm.max_tokens, 16000)
        self.assertTrue(cfg.llm.critic.enabled)
        self.assertEqual(cfg.llm.critic.model, "anthropic/claude-sonnet-5")
        self.assertEqual(cfg.llm.critic.api_keys, ["sk-or-judge-key"])
        self.assertEqual(cfg.llm.critic.api_base, "https://openrouter.ai/api/v1")

    def test_unset_env_changes_nothing(self) -> None:
        from unittest.mock import patch
        cleared = {k: "" for k in self._ENV}
        with patch.dict(os.environ, cleared, clear=False):
            cfg = TarjomehConfig()
            before_model = cfg.llm.model
            cfg._apply_env_overrides()
        self.assertEqual(cfg.llm.model, before_model)
        self.assertFalse(cfg.llm.critic.enabled)

    def test_critic_enabled_false_wins(self) -> None:
        from unittest.mock import patch
        with patch.dict(os.environ, {"CRITIC_ENABLED": "false"}, clear=False):
            cfg = TarjomehConfig()
            cfg.llm.critic.enabled = True
            cfg._apply_env_overrides()
        self.assertFalse(cfg.llm.critic.enabled)


class TestPdfLineMerging(unittest.TestCase):
    """PDF lines must merge into real paragraphs with hyphenation repaired."""

    def test_join_block_lines_dehyphenation(self) -> None:
        from tarjomeh.parsers.pdf_parser import _join_block_lines
        SHY = "­"
        lines = [
            f"Looking out of the win{SHY}",   # soft hyphen → join, drop marker
            "dow, you see a billboard.",
            "It reads absent-",               # ASCII hyphen → join, keep hyphen
            "minded prose.",
            "A final line.",
        ]
        out = _join_block_lines(lines)
        self.assertIn("window", out)
        self.assertIn("absent-minded", out)
        self.assertIn("billboard. It reads", out)   # normal lines joined by space
        self.assertNotIn(SHY, out)

    def test_continuation_paragraphs_merged_across_blocks(self) -> None:
        from tarjomeh.parsers.pdf_parser import _merge_continuation_paragraphs
        from tarjomeh.parsers.base import Paragraph
        paras = [
            Paragraph(text="And as your thoughts roll on, you contemplate the", metadata={}),
            Paragraph(text="operations needed before value can be extracted.", metadata={}),
            Paragraph(text="A new sentence starts a new paragraph.", metadata={}),
            Paragraph(text="Heading Text", metadata={"heading_level": 2}),
        ]
        merged = _merge_continuation_paragraphs(paras)
        self.assertEqual(len(merged), 3)
        self.assertIn("contemplate the operations needed", merged[0].text)
        # Complete-sentence paragraph and heading are NOT merged.
        self.assertTrue(merged[1].text.startswith("A new sentence"))
        self.assertEqual(merged[2].metadata.get("heading_level"), 2)


if __name__ == "__main__":
    unittest.main()
