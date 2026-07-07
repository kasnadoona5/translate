"""LLM-based translation refinement for Tarjomeh.

Takes a translation that failed quality critique and produces an
improved version by feeding the critique feedback back to the LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from tarjomeh.quality.critique import CritiqueResult

logger = logging.getLogger(__name__)

import json
from tarjomeh.core.prompts import REFINE_PROMPT

# ── Mode → max iterations mapping ───────────────────────────────────
_MODE_MAX_ITERATIONS: dict[str, int] = {
    "academic": 2,
    "quality": 1,
    "fast": 0,
}


class TranslationRefiner:
    """Iteratively refines translations based on critique feedback.

    Parameters
    ----------
    llm_client:
        Any object exposing an ``async chat(prompt) -> str`` method.
    max_iterations : int | None
        Maximum number of refinement iterations.  If ``None``, the
        value is derived from the translation mode (academic=2,
        quality=1, fast=0).
    mode : str
        Translation mode — ``"academic"``, ``"quality"``, or ``"fast"``.
        Only used when *max_iterations* is ``None``.
    """

    def __init__(
        self,
        llm_client: Any,
        max_iterations: int | None = None,
        mode: str = "academic",
    ) -> None:
        self._llm = llm_client

        if max_iterations is not None:
            self.max_iterations = max_iterations
        else:
            self.max_iterations = _MODE_MAX_ITERATIONS.get(mode, 1)

    async def refine(
        self,
        source_text: str,
        translation: str,
        critique: CritiqueResult,
        terminology: str = "",
    ) -> str:
        """Produce a refined translation based on critique feedback.

        Parameters
        ----------
        source_text:
            Original English paragraph / chunk.
        translation:
            Current Persian translation to refine.
        critique:
            Structured critique result describing the quality issues.
            (Issues arrive pre-sorted critical → major → minor.)
        terminology:
            Mandatory glossary terms + established proper-noun renderings so
            refinement never drifts off-glossary while fixing other issues.

        Returns
        -------
        str
            The refined Persian translation text.  If the critique has
            no issues, the original *translation* is returned unchanged.
        """
        if not critique.issues:
            logger.info(
                "No issues in critique (avg=%.1f); skipping refinement.",
                critique.average,
            )
            return translation

        prompt = REFINE_PROMPT.format(
            source_text=source_text,
            translation=translation,
            critique=json.dumps(critique.to_dict(), indent=2, ensure_ascii=False),
            terminology=terminology or "(no glossary terms apply to this chunk)",
        )

        refined: str = await self._llm.chat(prompt)

        # Strip any accidental markdown fences the LLM might include.
        refined = refined.strip()
        if refined.startswith("```"):
            lines = refined.splitlines()
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            refined = "\n".join(lines).strip()

        logger.info(
            "Refined translation (critique avg was %.1f).",
            critique.average,
        )
        return refined
