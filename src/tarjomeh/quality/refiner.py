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
from tarjomeh.core.structured_output import parse_structured_output

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
    issue_decisions: list[dict[str, Any]] = field(default_factory=list)


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
        review_context: str = "",
    ) -> str:
        """Return only the refined translation text for backward compatibility."""
        return (
            await self.refine_with_decision(
                source_text=source_text,
                translation=translation,
                critique=critique,
                terminology=terminology,
                review_context=review_context,
            )
        ).translation

    async def refine_with_decision(
        self,
        source_text: str,
        translation: str,
        critique: CritiqueResult,
        terminology: str = "",
        review_context: str = "",
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

        compact_issues = [
            {
                key: detail.get(key)
                for key in (
                    "issue_id", "category", "severity", "confidence",
                    "source_segment_id", "source_quote", "current_persian_quote",
                    "suggested_correction", "rationale",
                )
            }
            for detail in critique.issue_details
        ]
        expected_issue_ids = [
            str(detail.get("issue_id", "")).strip()
            for detail in compact_issues
            if str(detail.get("issue_id", "")).strip()
        ]
        critique_payload = {
            "scores": {
                "accuracy": critique.accuracy,
                "fluency": critique.fluency,
                "terminology": critique.terminology,
                "register": critique.register,
                "average": critique.average,
            },
            "issues": compact_issues,
        }
        prompt = REFINE_PROMPT.format(
            source_text=source_text,
            translation=translation,
            critique=json.dumps(
                critique_payload, ensure_ascii=False, separators=(",", ":")
            ),
            terminology=terminology or "(no glossary terms apply to this chunk)",
            review_context=review_context or "(no additional review context)",
        )

        if hasattr(self._llm, "set_operation"):
            self._llm.set_operation("refinement")
        raw: str = await self._llm.chat(prompt)
        result = self._parse_response(raw, expected_issue_ids)
        result.attempts = 1
        all_errors = list(result.validation_errors)
        for retry in range(self.max_parse_retries):
            if result.valid:
                break
            logger.warning("Invalid refiner response; requesting JSON repair.")
            if hasattr(self._llm, "limit_next_call_attempts"):
                self._llm.limit_next_call_attempts(1)
            if hasattr(self._llm, "set_operation"):
                self._llm.set_operation("refinement_json_repair")
            repair_attempt = retry + 2
            try:
                repaired_raw = await self._llm.chat(
                    self._repair_prompt(
                        raw, result.validation_errors, compact_issues
                    )
                )
            except Exception as exc:
                error = f"repair_call_error:{type(exc).__name__}"
                logger.warning(
                    "Refiner JSON repair call failed (%s); bounded repair may continue.",
                    type(exc).__name__,
                )
                all_errors.append(error)
                result.validation_errors = list(
                    dict.fromkeys([*result.validation_errors, error])
                )
                result.attempts = repair_attempt
                continue
            raw = repaired_raw
            result = self._parse_response(raw, expected_issue_ids)
            result.attempts = repair_attempt
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
    def _parse_response(
        raw: str,
        expected_issue_ids: list[str] | None = None,
    ) -> RefinementResult:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        try:
            data = parse_structured_output(cleaned, expected=dict)
        except json.JSONDecodeError as exc:
            return RefinementResult(
                translation="", raw_response=raw, valid=False,
                validation_errors=[f"invalid_json: {exc.msg}"],
            )
        except ValueError as exc:
            return RefinementResult(
                translation="", raw_response=raw, valid=False,
                validation_errors=[f"invalid_json: {exc}"],
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
        raw_decisions = data.get("issue_decisions", data.get("decisions", []))
        if not refined:
            errors.append("translation_is_required")
        if decision not in {"revised", "preserved", "mixed"}:
            errors.append("decision_must_be_revised_preserved_or_mixed")
        if not rationale:
            errors.append("rationale_is_required")
        decisions: list[dict[str, Any]] = []
        expected = list(expected_issue_ids or [])
        if expected_issue_ids is not None:
            if not isinstance(raw_decisions, list):
                errors.append("issue_decisions_must_be_array")
                raw_decisions = []
            seen: set[str] = set()
            for item in raw_decisions:
                if not isinstance(item, dict):
                    errors.append("issue_decision_must_be_object")
                    continue
                issue_id = str(item.get("issue_id", "")).strip()
                item_decision = str(item.get("decision", "")).strip().lower()
                if item_decision == "partial":
                    item_decision = "partially_applied"
                resulting_span = str(
                    item.get("resulting_span", item.get("resulting_persian", ""))
                ).strip()
                reason = str(
                    item.get("rationale", item.get("reason", ""))
                ).strip()
                if issue_id not in expected:
                    errors.append(f"unknown_issue_decision:{issue_id or 'missing'}")
                elif issue_id in seen:
                    errors.append(f"duplicate_issue_decision:{issue_id}")
                else:
                    seen.add(issue_id)
                if item_decision not in {
                    "accepted", "rejected", "partially_applied",
                }:
                    errors.append(f"invalid_issue_decision:{issue_id or 'missing'}")
                if not resulting_span:
                    errors.append(f"resulting_span_required:{issue_id or 'missing'}")
                elif refined and resulting_span not in refined:
                    errors.append(f"resulting_span_not_in_translation:{issue_id or 'missing'}")
                if not reason:
                    errors.append(f"issue_rationale_required:{issue_id or 'missing'}")
                decisions.append({
                    "issue_id": issue_id,
                    "decision": item_decision,
                    "resulting_span": resulting_span[:320],
                    "rationale": reason[:500],
                })
            for issue_id in expected:
                if issue_id not in seen:
                    errors.append(f"missing_issue_decision:{issue_id}")
        return RefinementResult(
            translation=refined,
            decision=decision or "unknown",
            rationale=rationale,
            raw_response=raw,
            valid=not errors,
            validation_errors=errors,
            issue_decisions=decisions,
        )

    @staticmethod
    def _repair_prompt(
        raw: str,
        errors: list[str],
        expected_issues: list[dict[str, Any]] | None = None,
    ) -> str:
        raw_preview = (raw or "")[-12000:]
        return f"""The previous refinement response violated the required JSON schema.
Validation errors: {json.dumps(errors, ensure_ascii=False)}
Required compact issues: {json.dumps(expected_issues or [], ensure_ascii=False)}

Previous response:
{raw_preview}

Return ONLY valid JSON with non-empty fields: translation, decision
(revised, preserved, or mixed), rationale, and exactly one compact
issue_decision for every required issue ID. Each issue decision needs decision
(accepted, rejected, or partially_applied), resulting_span, and rationale.
Include the complete Persian translation once only. Do not add markdown fences."""
