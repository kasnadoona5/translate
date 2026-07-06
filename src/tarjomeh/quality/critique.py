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
    accuracy : int
        Faithfulness to source meaning (1-10).
    fluency : int
        Natural flow as Persian academic prose (1-10).
    terminology : int
        Consistency and correctness of domain terms (1-10).
    register : int
        Appropriateness of formality level (1-10).
    average : float
        Arithmetic mean of the four scores.
    issues : list[str]
        Human-readable descriptions of identified problems.
    raw_response : str
        The raw LLM response text (for debugging / audit).
    """

    accuracy: int = 0
    fluency: int = 0
    terminology: int = 0
    register: int = 0
    average: float = 0.0
    issues: list[str] = field(default_factory=list)
    raw_response: str = ""

    def passes_threshold(self, threshold: float = 7.0) -> bool:
        """Return ``True`` if the average score meets *threshold*."""
        return self.average >= threshold

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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CritiqueResult:
        """Reconstruct from a dict (e.g. loaded from the database)."""
        return cls(
            accuracy=int(data.get("accuracy", 0)),
            fluency=int(data.get("fluency", 0)),
            terminology=int(data.get("terminology", 0)),
            register=int(data.get("register", 0)),
            average=float(data.get("average", 0.0)),
            issues=list(data.get("issues", [])),
            raw_response=str(data.get("raw_response", "")),
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
    ) -> None:
        self._llm = llm_client
        self.quality_threshold = quality_threshold

    async def critique(
        self,
        source_text: str,
        translation: str,
    ) -> CritiqueResult:
        """Run a critique of *translation* against *source_text*.

        Parameters
        ----------
        source_text:
            Original English paragraph / chunk.
        translation:
            Corresponding Persian translation.

        Returns
        -------
        CritiqueResult
            Structured scores and issue list.
        """
        prompt = CRITIQUE_PROMPT.format(
            source_text=source_text,
            translation=translation,
        )

        raw = await self._llm.chat(prompt)
        return self._parse_response(raw)

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
        except json.JSONDecodeError:
            logger.warning("Failed to parse critique JSON, returning zero scores.")
            return CritiqueResult(raw_response=raw)

        scores = data.get("scores", {})
        if not isinstance(scores, dict):
            scores = {}

        accuracy = _clamp(scores.get("accuracy") if scores.get("accuracy") is not None else data.get("accuracy", 0), 1, 10)
        fluency = _clamp(scores.get("fluency") if scores.get("fluency") is not None else data.get("fluency", 0), 1, 10)
        terminology = _clamp(scores.get("terminology") if scores.get("terminology") is not None else data.get("terminology", 0), 1, 10)
        register = _clamp(scores.get("register") if scores.get("register") is not None else data.get("register", 0), 1, 10)

        overall = data.get("overall")
        if overall is not None:
            average = _clamp(overall, 1, 10)
        else:
            average = (accuracy + fluency + terminology + register) / 4.0

        raw_issues = data.get("issues", [])
        issues = []
        if isinstance(raw_issues, list):
            for issue in raw_issues:
                if isinstance(issue, dict):
                    category = issue.get("category", "")
                    fix = issue.get("suggested_fix", "")
                    explanation = issue.get("explanation", "")
                    issues.append(f"[{category}] Suggestion: {fix} (Reason: {explanation})")
                else:
                    issues.append(str(issue))
        else:
            issues = [str(raw_issues)]

        return CritiqueResult(
            accuracy=accuracy,
            fluency=fluency,
            terminology=terminology,
            register=register,
            average=average,
            issues=issues,
            raw_response=raw,
        )


def _clamp(value: Any, lo: int, hi: int) -> int:
    """Clamp *value* to [lo, hi], converting to int first."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, v))
