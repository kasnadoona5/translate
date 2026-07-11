"""Memory manager coordinating the translation memory system.

Integrates a book-level style profile, proper noun records, bilingual summary,
long-term memory, and short-term memory for cohesive book-length translation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.chunking.chunker import Chunk
from tarjomeh.memory.proper_nouns import ProperNouns
from tarjomeh.memory.bilingual_summary import BilingualSummary
from tarjomeh.memory.long_term import LongTermMemory
from tarjomeh.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)


@dataclass
class MemoryContext:
    """Consolidated context from all four memory layers."""

    book_context: str = ""
    style_profile: str = ""
    proper_nouns: str = ""
    bilingual_summary: str = ""
    long_term: str = ""
    short_term: str = ""

    def format(self) -> str:
        """Format the memory layers into a single prompt-ready string."""
        parts = []
        if self.book_context:
            parts.append(
                "--- Reviewable Book Research Context ---"
                + chr(10)
                + self.book_context
            )
        if self.style_profile:
            parts.append(f"--- Style Profile: Book-Level Voice Guide ---\n{self.style_profile}")
        if self.proper_nouns:
            parts.append(f"--- Layer 1: Proper Nouns & Transliterations ---\n{self.proper_nouns}")
        if self.bilingual_summary:
            parts.append(f"--- Layer 2: Bilingual Summary ---\n{self.bilingual_summary}")
        if self.long_term:
            parts.append(f"--- Layer 3: Relevant Past Translations ---\n{self.long_term}")
        if self.short_term:
            parts.append(f"--- Layer 4: Short-term Context ---\n{self.short_term}")
        return "\n\n".join(parts) if parts else "No translation memory context available."


class MemoryManager:
    """Coordinates style profile, proper nouns, summaries, and translation memory."""

    def __init__(self, config: TarjomehConfig) -> None:
        self.config = config
        self.proper_nouns = ProperNouns()
        self.bilingual_summary = BilingualSummary()
        self.book_context = ""
        self.style_profile = ""
        self.style_samples: list[str] = []
        
        # Pull retrieval_k and window_size from config
        retrieval_k = config.to_dict().get("memory", {}).get("long_term_retrieval_k", 5)
        window_size = config.to_dict().get("memory", {}).get("short_term_window", 4)
        
        self.long_term = LongTermMemory(retrieval_k=retrieval_k)
        self.short_term = ShortTermMemory(window_size=window_size)

    def get_context_for_chunk(self, chunk: Chunk) -> MemoryContext:
        """Retrieve relevant context for translating the given chunk."""
        # Layer 1: Proper Nouns
        include_inline_originals = self.config.output.term_notes in (
            "inline",
            "both",
        )
        proper_nouns_str = self.proper_nouns.get_context(
            include_inline_originals=include_inline_originals
        )

        # Layer 2: Bilingual Summary
        bilingual_summary_str = self.bilingual_summary.get_context()

        # Layer 3: Long-term Memory (TF-IDF)
        long_term_str = self.long_term.get_context(chunk.text)

        # Layer 4: Short-term Memory (Window)
        pairs = self.short_term.get_context()
        short_term_str = ""
        if pairs:
            short_term_str = "\n\n".join(
                f"EN: {src}\nFA: {tgt}"
                for src, tgt in pairs
            )

        return MemoryContext(
            book_context=self.book_context,
            style_profile=self.style_profile,
            proper_nouns=proper_nouns_str,
            bilingual_summary=bilingual_summary_str,
            long_term=long_term_str,
            short_term=short_term_str,
        )

    def update_after_translation(self, chunk: Chunk, translation: str) -> None:
        """Update synchronous memory layers with a new source-translation pair."""
        self.short_term.add(chunk.text, translation)
        self.long_term.add(chunk.text, translation)
        self._update_style_profile(translation)
        # Any known proper noun occurring in this chunk has now had its first
        # appearance — later chunks must not repeat the English parenthetical.
        self.proper_nouns.mark_seen_in_text(chunk.text)

    def _update_style_profile(self, translation: str) -> None:
        """Maintain a compact book-level style guide from early translations."""
        text = " ".join((translation or "").split())
        if not text:
            return

        if len(self.style_samples) < 5:
            self.style_samples.append(text[:500])

        samples = "\n".join(
            f"{i + 1}. {sample}"
            for i, sample in enumerate(self.style_samples)
        )
        self.style_profile = (
            "Maintain one coherent scholarly Iranian-Persian voice across the book. "
            "Prefer formal academic diction, precise conceptual renderings, stable "
            "citation handling, and the sentence rhythm established in the samples below. "
            "Do not simplify later chapters into a different register.\n\n"
            f"Representative early translation samples:\n{samples}"
        )

    async def update_proper_nouns(self, llm_client: Any, text: str) -> None:
        """Incrementally identify new proper nouns in the text and add them."""
        from tarjomeh.core.prompts import INCREMENTAL_NER_PROMPT, GLOSSARY_EXTRACT_PROMPT

        include_inline_originals = self.config.output.term_notes in (
            "inline",
            "both",
        )
        known = self.proper_nouns.get_context(
            include_inline_originals=include_inline_originals
        )
        if not known:
            # First chapter/TOC extract
            prompt = GLOSSARY_EXTRACT_PROMPT.format(text=text)
        else:
            prompt = INCREMENTAL_NER_PROMPT.format(known_entities=known, text=text)

        try:
            response = await llm_client.chat(prompt)
            # Clean possible markdown fences
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()

            items = json.loads(cleaned)
            if isinstance(items, list):
                for item in items:
                    term = item.get("term")
                    persian = item.get("suggested_persian")
                    if term and persian:
                        self.proper_nouns.add_noun(term, persian)
            elif isinstance(items, dict):
                # Handle unexpected single dict object wrapping the list
                terms_list = items.get("terms", []) or items.get("extracted_terms", []) or []
                for item in terms_list:
                    term = item.get("term")
                    persian = item.get("suggested_persian")
                    if term and persian:
                        self.proper_nouns.add_noun(term, persian)
        except Exception as exc:
            logger.warning("Failed to extract proper nouns incrementally: %s", exc)

    async def update_bilingual_summary(self, llm_client: Any, new_content: str, translation: str) -> None:
        """Call the LLM to update the running bilingual summary of translated chapters."""
        from tarjomeh.core.prompts import SUMMARY_UPDATE_PROMPT

        current = self.bilingual_summary.get_context() or "None"
        prompt = SUMMARY_UPDATE_PROMPT.format(
            current_summary=current,
            new_content=new_content,
            translation=translation,
        )

        try:
            response = await llm_client.chat(prompt)
            self.bilingual_summary.update(response)
        except Exception as exc:
            logger.warning("Failed to update running bilingual summary: %s", exc)

    def to_dict(self) -> dict[str, Any]:
        """Serialize memory state for database checkpointing."""
        return {
            "book_context": self.book_context,
            "style_profile": self.style_profile,
            "style_samples": self.style_samples,
            "proper_nouns": self.proper_nouns.serialize(),
            "bilingual_summary": self.bilingual_summary.serialize(),
            "past_translations": self.long_term.serialize(),
            "short_term_context": self.short_term.serialize(),
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore memory state from a checkpoint dictionary."""
        self.book_context = str(data.get("book_context", ""))
        self.style_profile = str(data.get("style_profile", ""))
        self.style_samples = list(data.get("style_samples", []))
        self.proper_nouns.deserialize(data.get("proper_nouns", {}))
        self.bilingual_summary.deserialize(data.get("bilingual_summary", {}))
        self.long_term.deserialize(data.get("past_translations", []))
        self.short_term = ShortTermMemory.deserialize(
            data.get("short_term_context", []),
            window_size=self.short_term.window_size
        )
