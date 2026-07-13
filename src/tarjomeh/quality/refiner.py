"""LLM-based translation refinement for Tarjomeh.

Takes a translation that failed quality critique and produces an
improved version by feeding the critique feedback back to the LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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


@dataclass
class RefinementResult:
    """Structured result from a balanced refinement pass."""

    translation: str
    decision: str = "unknown"
    rationale: str = ""
    raw_response: str = ""
    valid: bool = True
    validation_errors: list[str] = field(default_factory=list)
    attempts: int = 1


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
        max_parse_retries: int = 1,
    ) -> None:
        self._llm = llm_client

        if max_iterations is not None:
            self.max_iterations = max_iterations
        else:
            self.max_iterations = _MODE_MAX_ITERATIONS.get(mode, 1)
        self.max_parse_retries = max(0, max_parse_retries)

    async def refine(
        self,
        source_text: str,
        translation: str,
        critique: CritiqueResult,
        terminology: str = "",
    ) -> str:
        """Return only the refined translation text for backward compatibility."""
        return (
            await self.refine_with_decision(
                source_text=source_text,
                translation=translation,
                critique=critique,
                terminology=terminology,
            )
        ).translation

    async def refine_with_decision(
        self,
        source_text: str,
        translation: str,
        critique: CritiqueResult,
        terminology: str = "",
    ) -> RefinementResult:
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
            return RefinementResult(
                translation=translation,
                decision="preserved",
                rationale="No critique issues were provided.",
            )

        prompt = REFINE_PROMPT.format(
            source_text=source_text,
            translation=translation,
            critique=json.dumps(critique.to_dict(), indent=2, ensure_ascii=False),
            terminology=terminology or "(no glossary terms apply to this chunk)",
        )

        raw: str = await self._llm.chat(prompt)
        result = self._parse_response(raw)
        result.attempts = 1
        all_errors = list(result.validation_errors)
        for retry in range(self.max_parse_retries):
            if result.valid:
                break
            logger.warning("Invalid refiner response; requesting JSON repair.")
            raw = await self._llm.chat(self._repair_prompt(raw, result.validation_errors))
            result = self._parse_response(raw)
            result.attempts = retry + 2
            all_errors.extend(result.validation_errors)

        if not result.valid:
            return RefinementResult(
                translation=translation,
                decision="preserved",
                rationale="Refiner output was invalid after bounded JSON repair; prior translation retained.",
                raw_response=raw,
                valid=False,
                validation_errors=list(dict.fromkeys(all_errors)),
                attempts=result.attempts,
            )

        if result.attempts > 1:
            result.validation_errors = list(dict.fromkeys(all_errors))

        logger.info(
            "Refined translation (critique avg was %.1f, decision=%s).",
            critique.average,
            result.decision,
        )
        return result

    @staticmethod
    def _parse_response(raw: str) -> RefinementResult:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return RefinementResult(
                translation="", raw_response=raw, valid=False,
                validation_errors=[f"invalid_json: {exc.msg}"],
            )
        if not isinstance(data, dict):
            return RefinementResult(
                translation="", raw_response=raw, valid=False,
                validation_errors=["top_level_must_be_object"],
            )
        errors = []
        refined = str(data.get("translation", "")).strip()
        decision = str(data.get("decision", "")).strip().lower()
        rationale = str(data.get("rationale", "")).strip()
        if not refined:
            errors.append("translation_is_required")
        if decision not in {"revised", "preserved", "mixed"}:
            errors.append("decision_must_be_revised_preserved_or_mixed")
        if not rationale:
            errors.append("rationale_is_required")
        return RefinementResult(
            translation=refined,
            decision=decision or "unknown",
            rationale=rationale,
            raw_response=raw,
            valid=not errors,
            validation_errors=errors,
        )

    @staticmethod
    def _repair_prompt(raw: str, errors: list[str]) -> str:
        return f"""The previous refinement response violated the required JSON schema.
Validation errors: {json.dumps(errors, ensure_ascii=False)}

Previous response:
{raw}

Return ONLY valid JSON with non-empty fields: translation, decision
(revised, preserved, or mixed), and rationale. Do not add markdown fences."""
