"""LLM-based translation refinement for Tarjomeh.

Takes a translation that failed quality critique and produces an
improved version by feeding the critique feedback back to the LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from tarjomeh.quality.critique import CritiqueResult

logger = logging.getLogger(__name__)

# ── Refinement prompt ────────────────────────────────────────────────
REFINE_PROMPT = """\
You are a senior Persian translation editor specialising in academic \
texts (political theory, sociology, philosophy).

Below is an English source passage, its initial Persian translation, \
and a quality critique identifying specific issues.  Produce an \
**improved Persian translation** that addresses every issue raised in \
the critique while preserving the meaning and academic register.

### Source (English)
{source_text}

### Initial Translation (Persian)
{translation}

### Quality Critique
Scores — Accuracy: {accuracy}/10, Fluency: {fluency}/10, \
Terminology: {terminology}/10, Register: {register}/10 \
(Average: {average:.1f}/10)

Issues:
{issues_text}

### Instructions
- Fix every listed issue.
- Do NOT add commentary — return ONLY the refined Persian translation.
- Maintain publication-quality Iranian academic prose (نثر آکادمیک).
"""

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

        issues_text = "\n".join(f"- {issue}" for issue in critique.issues)

        prompt = REFINE_PROMPT.format(
            source_text=source_text,
            translation=translation,
            accuracy=critique.accuracy,
            fluency=critique.fluency,
            terminology=critique.terminology,
            register=critique.register,
            average=critique.average,
            issues_text=issues_text,
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
