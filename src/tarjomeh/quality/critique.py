"""LLM-based translation critique engine for Tarjomeh.

Evaluates Persian translations on four dimensions — accuracy, fluency,
terminology, and register — using a structured LLM prompt, and returns
a :class:`CritiqueResult` with numeric scores and issue annotations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

from tarjomeh.core.prompts import CRITIQUE_PROMPT


@dataclass
class CritiqueResult:
    """Structured result of a translation quality critique.

    Attributes
    ----------
    accuracy : float
        Faithfulness to source meaning (1-10).
    fluency : float
        Natural flow as Persian academic prose (1-10).
    terminology : float
        Consistency and correctness of domain terms (1-10).
    register : float
        Appropriateness of formality level (1-10).
    average : float
        Arithmetic mean of the four scores.
    issues : list[str]
        Human-readable descriptions of identified problems.
    raw_response : str
        The raw LLM response text (for debugging / audit).
    """

    accuracy: float = 0.0
    fluency: float = 0.0
    terminology: float = 0.0
    register: float = 0.0
    average: float = 0.0
    issues: list[str] = field(default_factory=list)
    raw_response: str = ""
    valid: bool = True
    validation_errors: list[str] = field(default_factory=list)
    attempts: int = 1

    def passes_threshold(self, threshold: float = 7.0) -> bool:
        """Return ``True`` if the average score meets *threshold*."""
        return self.valid and self.average >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (suitable for JSON / SQLite)."""
        return {
            "accuracy": self.accuracy,
            "fluency": self.fluency,
            "terminology": self.terminology,
            "register": self.register,
            "average": self.average,
            "issues": self.issues,
            "raw_response": self.raw_response,
            "valid": self.valid,
            "validation_errors": self.validation_errors,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CritiqueResult:
        """Reconstruct from a dict (e.g. loaded from the database)."""
        return cls(
            accuracy=float(data.get("accuracy", 0)),
            fluency=float(data.get("fluency", 0)),
            terminology=float(data.get("terminology", 0)),
            register=float(data.get("register", 0)),
            average=float(data.get("average", 0.0)),
            issues=list(data.get("issues", [])),
            raw_response=str(data.get("raw_response", "")),
            valid=bool(data.get("valid", True)),
            validation_errors=list(data.get("validation_errors", [])),
            attempts=int(data.get("attempts", 1)),
        )


class TranslationCritique:
    """Evaluates translation quality via an LLM.

    Parameters
    ----------
    llm_client:
        Any object that exposes an async-compatible ``chat(prompt)``
        method returning a string response.  (Matches the Tarjomeh
        ``LLMClient`` protocol.)
    quality_threshold : float
        Minimum average score (out of 10) required to pass.  Defaults
        to ``7.0``.
    """

    def __init__(
        self,
        llm_client: Any,
        quality_threshold: float = 7.0,
        max_parse_retries: int = 1,
    ) -> None:
        self._llm = llm_client
        self.quality_threshold = quality_threshold
        self.max_parse_retries = max(0, max_parse_retries)

    async def critique(
        self,
        source_text: str,
        translation: str,
        terminology: str = "",
    ) -> CritiqueResult:
        """Run a critique of *translation* against *source_text*.

        Parameters
        ----------
        source_text:
            Original English paragraph / chunk.
        translation:
            Corresponding Persian translation.
        terminology:
            Mandatory glossary terms + established proper-noun renderings.
            The critic judges the "terminology" dimension against this list
            instead of guessing blind.

        Returns
        -------
        CritiqueResult
            Structured scores and issue list.
        """
        prompt = CRITIQUE_PROMPT.format(
            source_text=source_text,
            translation=translation,
            terminology=terminology or "(no glossary terms apply to this chunk)",
        )

        raw = await self._llm.chat(prompt)
        result = self._parse_response(raw)
        result.attempts = 1
        all_errors = list(result.validation_errors)
        for retry in range(self.max_parse_retries):
            if result.valid:
                break
            logger.warning("Invalid critique response; requesting JSON repair.")
            repair_prompt = self._repair_prompt(raw, result.validation_errors)
            raw = await self._llm.chat(repair_prompt)
            result = self._parse_response(raw)
            result.attempts = retry + 2
            all_errors.extend(result.validation_errors)
        if result.attempts > 1:
            result.validation_errors = list(dict.fromkeys(all_errors))
        return result

    @staticmethod
    def _repair_prompt(raw: str, errors: list[str]) -> str:
        return f"""The previous translation critique was not valid JSON for the required schema.
Validation errors: {json.dumps(errors, ensure_ascii=False)}

Previous response:
{raw}

Return ONLY a corrected JSON object with numeric 1-10 scores for accuracy,
fluency, terminology, and register; an optional numeric overall score; and an
issues array. Do not add markdown fences or commentary."""

    # ── response parsing ─────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw: str) -> CritiqueResult:
        """Parse a JSON response from the LLM into a CritiqueResult.

        Handles minor formatting issues (markdown fences, trailing
        commas) gracefully.
        """
        cleaned = raw.strip()

        # Strip optional markdown code fences.
        if cleaned.startswith("```"):
            # Remove ```json … ``` wrapping
            lines = cleaned.splitlines()
            lines = [
                ln
                for ln in lines
                if not ln.strip().startswith("```")
            ]
            cleaned = "\n".join(lines)

        try:
            data: dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse critique JSON.")
            return CritiqueResult(
                raw_response=raw,
                valid=False,
                validation_errors=[f"invalid_json: {exc.msg}"],
            )

        if not isinstance(data, dict):
            return CritiqueResult(
                raw_response=raw,
                valid=False,
                validation_errors=["top_level_must_be_object"],
            )

        scores = data.get("scores", {})
        if not isinstance(scores, dict):
            scores = {}

        errors: list[str] = []
        accuracy = _score(data, scores, "accuracy", errors)
        fluency = _score(data, scores, "fluency", errors)
        terminology = _score(data, scores, "terminology", errors)
        register = _score(data, scores, "register", errors)

        overall = data.get("overall")
        if overall is not None:
            average = _score_value(overall, "overall", errors)
        else:
            average = (accuracy + fluency + terminology + register) / 4.0

        raw_issues = data.get("issues", [])
        issues = []
        if isinstance(raw_issues, list):
            # Preserve severity + source segment (the refiner is instructed to
            # fix critical issues first) and sort critical → major → minor.
            severity_rank = {"critical": 0, "major": 1, "minor": 2}
            parsed: list[tuple[int, str]] = []
            for issue in raw_issues:
                if isinstance(issue, dict):
                    severity = str(issue.get("severity", "minor")).lower()
                    category = issue.get("category", "")
                    segment = issue.get("source_segment", "")
                    current = issue.get("current_translation", "")
                    fix = issue.get("suggested_fix", "")
                    explanation = issue.get("explanation", "")
                    text = f"[{severity.upper()}/{category}]"
                    if segment:
                        text += f' source: "{segment}"'
                    if current:
                        text += f' | current: "{current}"'
                    text += f" | fix: {fix} (Reason: {explanation})"
                    parsed.append((severity_rank.get(severity, 2), text))
                else:
                    parsed.append((2, str(issue)))
            parsed.sort(key=lambda t: t[0])
            issues = [text for _, text in parsed]
        else:
            issues = []
            errors.append("issues_must_be_array")

        return CritiqueResult(
            accuracy=accuracy,
            fluency=fluency,
            terminology=terminology,
            register=register,
            average=average,
            issues=issues,
            raw_response=raw,
            valid=not errors,
            validation_errors=errors,
        )


def _score(data: dict[str, Any], scores: dict[str, Any], key: str, errors: list[str]) -> float:
    value = scores.get(key) if scores.get(key) is not None else data.get(key)
    return _score_value(value, key, errors)


def _score_value(value: Any, key: str, errors: list[str]) -> float:
    """Validate one score without silently coercing missing values."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        errors.append(f"{key}_must_be_numeric")
        return 0.0
    if not 1.0 <= score <= 10.0:
        errors.append(f"{key}_must_be_between_1_and_10")
        return 0.0
    return score
