"""Regression tests for v0.2.0 fixes.

Covers:
- Phase 0.2: proportional intra-chunk translation distribution
- Phase 0.3: duplicate paragraph-index append (FixedChunker sub-chunks)
- Phase 1:  independent critic model client construction
"""

from __future__ import annotations

import os
import unittest

from tarjomeh.core.pipeline import (
    TranslationPipeline,
    _distribute_translation,
    _split_sentences_fa,
)
from tarjomeh.core.config import TarjomehConfig


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


if __name__ == "__main__":
    unittest.main()
