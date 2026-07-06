"""Back-translation verification for Tarjomeh.

Translates a subset of Persian translations back to English and
compares them with the original source to flag semantic drift.
"""

from __future__ import annotations

import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

from tarjomeh.core.prompts import BACK_TRANSLATE_PROMPT


@dataclass
class BackTranslationResult:
    """Result of a back-translation comparison.

    Attributes
    ----------
    similarity_score : float
        Word-overlap ratio between the original English and the
        back-translated English (0.0 – 1.0).
    flagged : bool
        ``True`` if the chunk should be flagged for human review
        (similarity < threshold).
    differences : list[str]
        Words present in the original but missing from the
        back-translation (indicative of potential meaning loss).
    back_translated : str
        The full back-translated English text.
    """

    similarity_score: float = 0.0
    flagged: bool = False
    differences: list[str] = field(default_factory=list)
    back_translated: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "similarity_score": self.similarity_score,
            "flagged": self.flagged,
            "differences": self.differences,
            "back_translated": self.back_translated,
        }


class BackTranslator:
    """Performs probabilistic back-translation QA.

    A configurable percentage of translated chunks are randomly sampled,
    translated back to English, and compared with the original source
    text to detect semantic drift.

    Parameters
    ----------
    llm_client:
        Any object exposing an ``async chat(prompt) -> str`` method.
    sample_pct : float
        Percentage of chunks to back-translate (0–100).  Default ``20``
        (the ``academic`` mode default).
    similarity_threshold : float
        Minimum word-overlap similarity (0.0–1.0) required for a chunk
        to pass.  Chunks below this threshold are flagged for human
        review.  Default ``0.5``.
    """

    def __init__(
        self,
        llm_client: Any,
        sample_pct: float = 20.0,
        similarity_threshold: float = 0.5,
    ) -> None:
        self._llm = llm_client
        self._sample_pct = max(0.0, min(100.0, sample_pct))
        self._similarity_threshold = similarity_threshold

    # ── sampling ─────────────────────────────────────────────────────

    def should_sample(self) -> bool:
        """Decide whether the current chunk should be back-translated.

        Uses a simple random draw based on :attr:`sample_pct`.

        Returns
        -------
        bool
            ``True`` if this chunk should be sampled.
        """
        if self._sample_pct <= 0.0:
            return False
        if self._sample_pct >= 100.0:
            return True
        return random.random() * 100.0 < self._sample_pct

    # ── back-translation ─────────────────────────────────────────────

    async def back_translate(self, persian_text: str) -> str:
        """Translate *persian_text* back to English via the LLM.

        Parameters
        ----------
        persian_text:
            Persian text to translate back to English.

        Returns
        -------
        str
            English back-translation.
        """
        prompt = BACK_TRANSLATE_PROMPT.format(persian_text=persian_text)
        result: str = await self._llm.chat(prompt)

        # Strip accidental markdown fences.
        result = result.strip()
        if result.startswith("```"):
            lines = result.splitlines()
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            result = "\n".join(lines).strip()

        return result

    # ── comparison ───────────────────────────────────────────────────

    def compare(
        self,
        original_english: str,
        back_translated: str,
    ) -> BackTranslationResult:
        """Compare the original English with a back-translation.

        Uses a simple word-overlap ratio as the similarity metric:

        .. math::

            \\text{similarity} = \\frac{|W_{\\text{orig}} \\cap W_{\\text{back}}|}
                                      {|W_{\\text{orig}} \\cup W_{\\text{back}}|}

        where *W* is the multiset of lowercased tokens.

        Parameters
        ----------
        original_english:
            The original English source text.
        back_translated:
            The English text produced by back-translating the Persian.

        Returns
        -------
        BackTranslationResult
            Comparison result with similarity score and flagged status.
        """
        orig_tokens = _tokenize_simple(original_english)
        back_tokens = _tokenize_simple(back_translated)

        orig_counter = Counter(orig_tokens)
        back_counter = Counter(back_tokens)

        # Intersection (element-wise minimum) and union (element-wise maximum).
        intersection_size = sum((orig_counter & back_counter).values())
        union_size = sum((orig_counter | back_counter).values())

        similarity = intersection_size / union_size if union_size > 0 else 0.0

        # Words in the original but missing from the back-translation.
        missing = orig_counter - back_counter
        differences = sorted(missing.elements())

        flagged = similarity < self._similarity_threshold

        if flagged:
            logger.warning(
                "Back-translation flagged: similarity=%.2f (threshold=%.2f)",
                similarity,
                self._similarity_threshold,
            )

        return BackTranslationResult(
            similarity_score=round(similarity, 4),
            flagged=flagged,
            differences=differences,
            back_translated=back_translated,
        )


# ── Utility ──────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


def _tokenize_simple(text: str) -> list[str]:
    """Extract lowercased word tokens from English text.

    A deliberately simple tokenizer — strips punctuation and
    lowercases everything.  Good enough for word-overlap comparison.
    """
    return [m.group().lower() for m in _WORD_RE.finditer(text)]
