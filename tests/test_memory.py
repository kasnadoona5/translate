"""Unit tests for Phase 4: 4-Layer Memory System."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.chunking.chunker import Chunk
from tarjomeh.memory.proper_nouns import ProperNouns
from tarjomeh.memory.bilingual_summary import BilingualSummary
from tarjomeh.memory.long_term import LongTermMemory
from tarjomeh.memory.short_term import ShortTermMemory
from tarjomeh.memory.manager import MemoryManager, MemoryContext


class TestProperNouns(unittest.TestCase):
    """Test Layer 1: Proper Noun Records."""

    def test_proper_nouns_mapping(self) -> None:
        pn = ProperNouns()
        self.assertEqual(len(pn), 0)
        
        pn.add_noun("State", "دولت")
        pn.add_noun("Capital", "سرمایه")
        self.assertEqual(len(pn), 2)
        
        context = pn.get_context()
        self.assertIn("- Capital -> سرمایه", context)
        self.assertIn("- State -> دولت", context)

        # Test Serialization (v2 shape: nouns + introduced-tracking)
        serialized = pn.serialize()
        self.assertEqual(serialized["nouns"]["State"], "دولت")
        self.assertEqual(serialized["introduced"], [])

        # Test Deserialization
        pn2 = ProperNouns()
        pn2.deserialize(serialized)
        self.assertEqual(len(pn2), 2)
        self.assertEqual(pn2.serialize()["nouns"]["Capital"], "سرمایه")

        # Legacy (flat-dict) checkpoints still deserialize
        pn3 = ProperNouns()
        pn3.deserialize({"State": "دولت"})
        self.assertEqual(len(pn3), 1)

        # First-occurrence tracking: seen nouns flip to [introduced]
        pn.mark_seen_in_text("The State exists.")
        self.assertTrue(pn.is_introduced("State"))
        self.assertFalse(pn.is_introduced("Capital"))
        ctx = pn.get_context()
        self.assertIn("- State -> دولت  [introduced]", ctx)
        self.assertIn("first occurrence pending", ctx)


    def test_categories_control_first_occurrence_originals(self) -> None:
        pn = ProperNouns()
        pn.add_noun("capital", "سرمایه", category="term")
        pn.add_noun("Monsanto", "مونسانتو", category="organization")

        self.assertFalse(pn.is_inline_eligible("capital"))
        self.assertTrue(pn.is_inline_eligible("Monsanto"))

        pn.add_noun("capital", "سرمایه", category="approved_term")
        self.assertTrue(pn.is_inline_eligible("capital"))
        self.assertEqual(
            pn.pending_inline_originals("capital and Monsanto"),
            {"Monsanto": "مونسانتو", "capital": "سرمایه"},
        )

        restored = ProperNouns()
        restored.deserialize(pn.serialize())
        self.assertEqual(restored.category_for("capital"), "approved_term")
        self.assertEqual(restored.category_for("Monsanto"), "organization")

        pn.mark_seen_in_text("capital and Monsanto")
        self.assertTrue(pn.is_introduced("capital"))
        self.assertTrue(pn.is_introduced("Monsanto"))


class TestBilingualSummary(unittest.TestCase):
    """Test Layer 2: Running Bilingual Summary."""

    def test_bilingual_summary_parsing(self) -> None:
        bs = BilingualSummary()
        self.assertEqual(bs.english_summary, "")
        
        raw_output = (
            "## English Summary\n"
            "This chapter explores political power.\n"
            "And its structures.\n"
            "## خلاصه فارسی\n"
            "این فصل به بررسی قدرت سیاسی می‌پردازد.\n"
            "و ساختارهای آن.\n"
        )
        bs.update(raw_output)
        
        self.assertEqual(bs.english_summary, "This chapter explores political power.\nAnd its structures.")
        self.assertEqual(bs.persian_summary, "این فصل به بررسی قدرت سیاسی می‌پردازد.\nو ساختارهای آن.")

        context = bs.get_context()
        self.assertIn("## English Summary", context)
        self.assertIn("## خلاصه فارسی (RTL)", context)


class TestLongTermMemory(unittest.TestCase):
    """Test Layer 3: Long-term TF-IDF Memory."""

    def test_tfidf_retrieval(self) -> None:
        ltm = LongTermMemory(retrieval_k=2)
        
        # Add past translations
        ltm.add("The State represents political authority.", "دولت نماینده اقتدار سیاسی است.")
        ltm.add("Capitalism organizes the economy.", "سرمایه‌داری اقتصاد را سازماندهی می‌کند.")
        ltm.add("The class struggle is central to history.", "مبارزه طبقاتی برای تاریخ محوریت دارد.")

        # Query for something similar to "State and political authority"
        results = ltm.get_relevant("State and political authority")
        
        self.assertTrue(len(results) > 0)
        # The first result should be the State one since it has word overlaps
        self.assertEqual(results[0]["source"], "The State represents political authority.")

        context = ltm.get_context("State and political authority")
        self.assertIn("دولت نماینده اقتدار سیاسی است.", context)


class TestShortTermMemory(unittest.TestCase):
    """Test Layer 4: Sliding Window Memory."""

    def test_sliding_window_eviction(self) -> None:
        stm = ShortTermMemory(window_size=3)
        stm.add("S1", "T1")
        stm.add("S2", "T2")
        stm.add("S3", "T3")
        self.assertEqual(len(stm), 3)

        # Evict S1, T1
        stm.add("S4", "T4")
        self.assertEqual(len(stm), 3)
        
        context = stm.get_context()
        self.assertEqual(context[0], ("S2", "T2"))
        self.assertEqual(context[2], ("S4", "T4"))


class TestMemoryManager(unittest.IsolatedAsyncioTestCase):
    """Test MemoryManager coordinator and async updates."""

    def setUp(self) -> None:
        self.config = TarjomehConfig()
        self.config.llm.openrouter.api_keys = ["dummy"]
        self.manager = MemoryManager(self.config)

    def test_manager_get_context(self) -> None:
        self.manager.proper_nouns.add_noun("Plato", "افلاطون")
        self.manager.bilingual_summary.english_summary = "Intro summary."
        self.manager.short_term.add("Hello", "سلام")
        
        chunk = Chunk(index=0, text="Plato's Republic", chapter_title="Book I", section_title="")
        ctx = self.manager.get_context_for_chunk(chunk)
        
        self.assertIn("Plato -> افلاطون", ctx.proper_nouns)
        self.assertIn("Intro summary.", ctx.bilingual_summary)
        self.assertIn("EN: Hello\nFA: سلام", ctx.short_term)

        formatted = ctx.format()
        self.assertIn("Layer 1", formatted)
        self.assertIn("Layer 2", formatted)
        self.assertIn("Layer 4", formatted)

    async def test_async_updates(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock()
        
        # 1. Test update_proper_nouns
        mock_llm.chat.return_value = (
            '[{"term": "Hegemony", "category": "term", '
            '"suggested_persian": "هژمونی", '
            '"exact_source_span": "Hegemony", '
            '"exact_target_span": "", "context_independent": true}]'
        )
        await self.manager.update_proper_nouns(mock_llm, "English text discussing Hegemony")
        mock_llm.set_operation.assert_called_with("proper_noun_initial")
        self.assertEqual(
            self.manager.proper_nouns.serialize()["nouns"].get("Hegemony"), "هژمونی"
        )

        # 2. Test update_bilingual_summary
        mock_llm.chat.return_value = (
            "## English Summary\n"
            "Political philosophy introduction.\n"
            "## خلاصه فارسی\n"
            "مقدمه‌ای بر فلسفه سیاسی.\n"
        )
        await self.manager.update_bilingual_summary(mock_llm, "new source text", "new translation text")
        mock_llm.set_operation.assert_called_with("bilingual_summary_update")
        self.assertEqual(self.manager.bilingual_summary.english_summary, "Political philosophy introduction.")
        self.assertEqual(self.manager.bilingual_summary.persian_summary, "مقدمه‌ای بر فلسفه سیاسی.")


if __name__ == "__main__":
    unittest.main()
