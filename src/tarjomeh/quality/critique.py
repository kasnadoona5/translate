"""LLM-based translation critique engine for Tarjomeh.

Evaluates Persian translations on four dimensions — accuracy, fluency,
terminology, and register — using a structured LLM prompt, and returns
a :class:`CritiqueResult` with numeric scores and issue annotations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

from tarjomeh.core.prompts import (
    CRITIQUE_PROMPT,
    FOCUSED_ATTACHMENT_REVIEW_PROMPT,
    PERSIAN_READABILITY_REVIEW_PROMPT,
)
from tarjomeh.core.structured_output import parse_structured_output
from tarjomeh.persian.orthography import apply_safe_persian_orthography
from tarjomeh.quality.grounding import (
    concept_risks,
    indexed_source,
    resolve_source_segment,
    source_segments,
)


_MQM_CATEGORIES = {
    "accuracy", "omission", "addition", "terminology", "name", "number",
    "citation", "fluency", "register", "typography",
}
_MQM_SEVERITIES = {"critical", "major", "minor"}
_MAX_MQM_ISSUES = 8
_MAX_QUOTE_CHARS = 240
_MAX_FIX_CHARS = 320
_MAX_RATIONALE_CHARS = 500


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
    issue_details: list[dict[str, Any]] = field(default_factory=list)
    ignored_issue_details: list[dict[str, Any]] = field(default_factory=list)
    coverage_checked_segment_ids: list[str] = field(default_factory=list)
    uncovered_source_segment_ids: list[str] = field(default_factory=list)
    coverage_complete: bool = True
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
            "issue_details": self.issue_details,
            "ignored_issue_details": self.ignored_issue_details,
            "coverage_checked_segment_ids": self.coverage_checked_segment_ids,
            "uncovered_source_segment_ids": self.uncovered_source_segment_ids,
            "coverage_complete": self.coverage_complete,
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
            issue_details=list(data.get("issue_details", [])),
            ignored_issue_details=list(data.get("ignored_issue_details", [])),
            coverage_checked_segment_ids=list(
                data.get("coverage_checked_segment_ids", [])
            ),
            uncovered_source_segment_ids=list(
                data.get("uncovered_source_segment_ids", [])
            ),
            coverage_complete=bool(data.get("coverage_complete", True)),
            raw_response=str(data.get("raw_response", "")),
            valid=bool(data.get("valid", True)),
            validation_errors=list(data.get("validation_errors", [])),
            attempts=int(data.get("attempts", 1)),
        )


@dataclass
class ReadabilityReviewResult:
    """Bounded target-only evidence; never an independent edit authority."""

    issues: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""
    valid: bool = True
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class AttachmentReviewResult:
    """Source-grounded advisory evidence; never an independent edit authority."""

    issues: list[dict[str, Any]] = field(default_factory=list)
    reported_issue_count: int = 0
    raw_response: str = ""
    valid: bool = True
    validation_errors: list[str] = field(default_factory=list)


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
        review_context: str = "",
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
            source_text=indexed_source(source_text),
            translation=translation,
            terminology=terminology or "(no glossary terms apply to this chunk)",
            review_context=review_context or "(no additional review context)",
        )

        if hasattr(self._llm, "set_operation"):
            self._llm.set_operation("critique")
        raw = await self._llm.chat(prompt)
        result = self._parse_response(
            raw, source_text, translation, require_coverage=True
        )
        result.attempts = 1
        all_errors = list(result.validation_errors)
        for retry in range(self.max_parse_retries):
            if result.valid:
                break
            logger.warning("Invalid critique response; requesting JSON repair.")
            if hasattr(self._llm, "limit_next_call_attempts"):
                self._llm.limit_next_call_attempts(1)
            repair_prompt = self._repair_prompt(
                raw, result.validation_errors, source_text, translation
            )
            if hasattr(self._llm, "set_operation"):
                self._llm.set_operation("critique_json_repair")
            raw = await self._llm.chat(repair_prompt)
            result = self._parse_response(
                raw, source_text, translation, require_coverage=True
            )
            result.attempts = retry + 2
            all_errors.extend(result.validation_errors)
        if result.attempts > 1:
            result.validation_errors = list(dict.fromkeys(all_errors))
        return result

    async def review_persian_readability(
        self,
        translation: str,
    ) -> ReadabilityReviewResult:
        """Collect exact-span Persian readability evidence without editing."""
        prompt = PERSIAN_READABILITY_REVIEW_PROMPT.format(
            translation=translation,
        )
        if hasattr(self._llm, "set_operation"):
            self._llm.set_operation("persian_readability_review")
        raw = await self._llm.chat(prompt)
        return self._parse_readability_response(raw, translation)

    async def review_attachment(
        self,
        source_text: str,
        translation: str,
        *,
        terminology: str = "",
        review_context: str = "",
    ) -> AttachmentReviewResult:
        """Review only attachment and scope against the exact source text."""
        prompt = FOCUSED_ATTACHMENT_REVIEW_PROMPT.format(
            source_text=indexed_source(source_text),
            translation=translation,
            terminology=terminology or "(none)",
            review_context=review_context or "(none)",
        )
        if hasattr(self._llm, "set_operation"):
            self._llm.set_operation("focused_attachment_review")
        raw = await self._llm.chat(prompt)
        return self._parse_attachment_response(raw, source_text, translation)

    @staticmethod
    def _parse_attachment_response(
        raw: str, source_text: str, translation: str
    ) -> AttachmentReviewResult:
        try:
            data = parse_structured_output(raw.strip(), expected=dict)
        except (json.JSONDecodeError, ValueError) as exc:
            return AttachmentReviewResult(
                raw_response=raw, valid=False,
                validation_errors=[f"invalid_json: {exc}"],
            )
        raw_issues = data.get("issues", [])
        if not isinstance(raw_issues, list):
            return AttachmentReviewResult(
                raw_response=raw, valid=False,
                validation_errors=["issues_must_be_array"],
            )
        issues: list[dict[str, Any]] = []
        errors: list[str] = []
        if len(raw_issues) > 4:
            errors.append("too_many_issues")
        for index, raw_issue in enumerate(raw_issues[:4]):
            if not isinstance(raw_issue, dict):
                errors.append(f"issue_{index}_must_be_object")
                continue
            item = {
                key: str(raw_issue.get(key, "")).strip()
                for key in (
                    "category", "severity", "source_segment_id",
                    "source_quote", "current_persian_quote", "source_head",
                    "source_dependent", "persian_head", "persian_dependent",
                    "suggested_correction", "rationale",
                )
            }
            item["category"] = item["category"].lower()
            item["severity"] = item["severity"].lower()
            item_errors: list[str] = []
            if item["category"] not in {"accuracy", "fluency"}:
                item_errors.append("category_invalid")
            if item["severity"] not in {"major", "minor"}:
                item_errors.append("severity_invalid")
            try:
                confidence = float(raw_issue.get("confidence"))
                if not 0 <= confidence <= 1:
                    raise ValueError("outside 0-1")
            except (TypeError, ValueError):
                confidence = 0.0
                item_errors.append("confidence_invalid")
            for field, parent in (
                ("source_quote", source_text),
                ("current_persian_quote", translation),
            ):
                value = item[field]
                if not value or len(value) > _MAX_QUOTE_CHARS:
                    item_errors.append(f"{field}_invalid")
                elif value not in parent:
                    item_errors.append(f"{field}_not_found")
            for field, parent in (
                ("source_head", item["source_quote"]),
                ("source_dependent", item["source_quote"]),
                ("persian_head", item["current_persian_quote"]),
                ("persian_dependent", item["current_persian_quote"]),
            ):
                if not item[field] or item[field] not in parent:
                    item_errors.append(f"{field}_not_grounded")
            if (
                item["source_head"] == item["source_dependent"]
                or item["persian_head"] == item["persian_dependent"]
            ):
                item_errors.append("head_and_dependent_identical")
            if not item["suggested_correction"] or len(
                item["suggested_correction"]
            ) > _MAX_FIX_CHARS:
                item_errors.append("suggestion_invalid")
            if not item["rationale"] or len(item["rationale"]) > _MAX_RATIONALE_CHARS:
                item_errors.append("rationale_invalid")
            segment_id, segment_errors = resolve_source_segment(
                source_text, item["source_quote"], item["source_segment_id"]
            ) if item["source_quote"] else ("", ["segment_unresolved"])
            item_errors.extend(
                error for error in segment_errors
                if error != "issue_source_segment_id_corrected"
            )
            if item_errors:
                errors.extend(f"issue_{index}_{error}" for error in item_errors)
                continue
            item["source_segment_id"] = segment_id
            item["confidence"] = confidence
            item["issue_id"] = "attachment-" + hashlib.sha256(
                (item["source_quote"] + "\0" + item["current_persian_quote"])
                .encode("utf-8")
            ).hexdigest()[:12]
            item["formatted"] = (
                f"[{item['severity'].upper()}/{item['category']}] "
                f"source: {item['source_quote']} | current: "
                f"{item['current_persian_quote']} | fix: "
                f"{item['suggested_correction']}"
            )
            issues.append(item)
        return AttachmentReviewResult(
            issues=issues,
            reported_issue_count=len(raw_issues),
            raw_response=raw,
            valid=not errors,
            validation_errors=errors,
        )

    @staticmethod
    def _parse_readability_response(
        raw: str,
        translation: str,
    ) -> ReadabilityReviewResult:
        try:
            data = parse_structured_output(raw.strip(), expected=dict)
        except (json.JSONDecodeError, ValueError) as exc:
            return ReadabilityReviewResult(
                raw_response=raw,
                valid=False,
                validation_errors=[f"invalid_json: {exc}"],
            )
        raw_issues = data.get("issues", []) if isinstance(data, dict) else []
        if not isinstance(raw_issues, list):
            return ReadabilityReviewResult(
                raw_response=raw,
                valid=False,
                validation_errors=["issues_must_be_array"],
            )
        issues: list[dict[str, Any]] = []
        errors: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_issues[:4]):
            if not isinstance(item, dict):
                errors.append(f"issue_{index}_must_be_object")
                continue
            severity = str(item.get("severity", "")).strip().lower()
            current = str(item.get("current_persian_quote", "")).strip()
            suggested = str(item.get("suggested_correction", "")).strip()
            rationale = str(item.get("rationale", "")).strip()
            item_errors: list[str] = []
            if severity not in {"major", "minor"}:
                item_errors.append("severity_invalid")
            if not current or len(current) > _MAX_QUOTE_CHARS:
                item_errors.append("current_quote_invalid")
            elif not _span_is_grounded(current, translation):
                item_errors.append("current_quote_not_found")
            if not suggested or len(suggested) > _MAX_FIX_CHARS:
                item_errors.append("suggestion_invalid")
            if not rationale or len(rationale) > _MAX_RATIONALE_CHARS:
                item_errors.append("rationale_invalid")
            key = _normalize_span(current)
            if key in seen:
                item_errors.append("duplicate_issue")
            if item_errors:
                errors.extend(f"issue_{index}_{error}" for error in item_errors)
                continue
            seen.add(key)
            issues.append({
                "severity": severity,
                "current_persian_quote": current,
                "suggested_correction": suggested,
                "rationale": rationale,
            })
        return ReadabilityReviewResult(
            issues=issues,
            raw_response=raw,
            valid=not errors,
            validation_errors=errors,
        )

    @staticmethod
    def _repair_prompt(
        raw: str,
        errors: list[str],
        source_text: str = "",
        translation: str = "",
    ) -> str:
        raw_preview = (raw or "")[-12000:]
        return f"""The previous translation critique was not valid JSON for the required schema.
Validation errors: {json.dumps(errors, ensure_ascii=False)}

Previous response:
{raw_preview}

Current source for exact quote repair (stable sentence IDs included):
{indexed_source(source_text)[:7000]}

Current Persian translation for exact quote repair:
{translation[:6000]}

Return ONLY a corrected JSON object with numeric 1-10 scores for accuracy,
fluency, terminology, and register; an optional numeric overall score; and at
most {_MAX_MQM_ISSUES} compact MQM issues. Every issue must contain category,
severity, confidence (0-1), an exact source_quote, its source_segment_id, an
exact current_persian_quote, suggested_correction, and rationale. Do not add praise,
markdown fences, or commentary. Also return source_coverage with every stable
source label exactly once in checked_source_segment_ids, any meaning not represented
in uncovered_source_segment_ids, and complete=true only when that uncovered list is
empty. Every uncovered label needs a grounded omission or accuracy issue."""

    # ── response parsing ─────────────────────────────────────────────

    @staticmethod
    def _parse_response(
        raw: str,
        source_text: str = "",
        translation: str = "",
        *,
        require_coverage: bool = False,
    ) -> CritiqueResult:
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
            data: dict[str, Any] = parse_structured_output(
                cleaned, expected=dict
            )
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse critique JSON.")
            return CritiqueResult(
                raw_response=raw,
                valid=False,
                validation_errors=[f"invalid_json: {exc.msg}"],
            )
        except ValueError as exc:
            return CritiqueResult(
                raw_response=raw,
                valid=False,
                validation_errors=[f"invalid_json: {exc}"],
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
        issues: list[str] = []
        if isinstance(raw_issues, list):
            severity_rank = {"critical": 0, "major": 1, "minor": 2}
            parsed: list[tuple[int, str, dict[str, Any]]] = []
            ignored_issue_details: list[dict[str, Any]] = []
            seen_issue_keys: set[tuple[str, str]] = set()
            invalid_issue_count = 0
            for issue in raw_issues:
                if not isinstance(issue, dict):
                    if not source_text and not translation:
                        text = str(issue)
                        parsed.append((2, text, {"formatted": text}))
                    else:
                        invalid_issue_count += 1
                        ignored_issue_details.append({
                            "formatted": str(issue),
                            "ignored_reason": "invalid_issue",
                            "validation_errors": ["issue_must_be_object"],
                        })
                    continue
                category = str(issue.get("category", "")).strip().lower()
                severity = str(issue.get("severity", "")).strip().lower()
                segment = str(
                    issue.get("source_quote", issue.get("source_segment", ""))
                ).strip()
                requested_segment_id = str(
                    issue.get("source_segment_id", "")
                ).strip()
                current = str(
                    issue.get(
                        "current_persian_quote",
                        issue.get("current_translation", ""),
                    )
                ).strip()
                fix = str(
                    issue.get(
                        "suggested_correction", issue.get("suggested_fix", "")
                    )
                ).strip()
                fix, fix_orthography_edits = apply_safe_persian_orthography(fix)
                explanation = str(
                    issue.get("rationale", issue.get("explanation", ""))
                ).strip()
                preliminary_detail: dict[str, Any] = {
                    "issue_id": _stable_issue_id(category, segment),
                    "category": category,
                    "severity": severity,
                    "source_quote": segment,
                    "source_segment_id": requested_segment_id,
                    "current_persian_quote": current,
                    "suggested_correction": fix,
                    "rationale": explanation,
                    "source_segment": segment,
                    "current_translation": current,
                    "suggested_fix": fix,
                    "explanation": explanation,
                }
                if fix_orthography_edits:
                    preliminary_detail["suggestion_orthography_normalized"] = (
                        fix_orthography_edits
                    )
                if _is_terminology_orthography_equivalent(preliminary_detail):
                    ignored_issue_details.append({
                        **preliminary_detail,
                        "ignored_reason": "terminology_orthography_equivalent",
                    })
                    continue
                if _is_noop_issue(preliminary_detail):
                    ignored_issue_details.append({
                        **preliminary_detail,
                        "ignored_reason": "no_textual_change",
                    })
                    continue
                issue_errors: list[str] = []
                confidence = _confidence_value(
                    issue.get("confidence"), issue_errors
                )
                if category not in _MQM_CATEGORIES:
                    issue_errors.append(
                        f"issue_category_invalid:{category or 'missing'}"
                    )
                if severity not in _MQM_SEVERITIES:
                    issue_errors.append(
                        f"issue_severity_invalid:{severity or 'missing'}"
                    )
                if not segment and (source_text or translation):
                    issue_errors.append("issue_source_quote_required")
                elif len(segment) > _MAX_QUOTE_CHARS:
                    issue_errors.append("issue_source_quote_too_long")
                elif source_text and not _span_is_grounded(segment, source_text):
                    issue_errors.append("issue_source_quote_not_found")
                resolved_segment_id = requested_segment_id
                segment_errors: list[str] = []
                if segment and source_text:
                    resolved_segment_id, segment_errors = resolve_source_segment(
                        source_text, segment, requested_segment_id
                    )
                    issue_errors.extend(
                        error for error in segment_errors
                        if error not in {
                            "issue_source_segment_ambiguous",
                            "issue_source_segment_id_corrected",
                        }
                    )
                if not current and (source_text or translation):
                    issue_errors.append("issue_current_persian_quote_required")
                elif len(current) > _MAX_QUOTE_CHARS:
                    issue_errors.append("issue_current_persian_quote_too_long")
                elif translation and not _span_is_grounded(current, translation):
                    issue_errors.append("issue_current_persian_quote_not_found")
                if not fix:
                    issue_errors.append("issue_suggested_correction_required")
                elif len(fix) > _MAX_FIX_CHARS:
                    issue_errors.append("issue_suggested_correction_too_long")
                if not explanation:
                    issue_errors.append("issue_rationale_required")
                if issue_errors:
                    invalid_issue_count += 1
                    ignored_issue_details.append({
                        **preliminary_detail,
                        "confidence": confidence,
                        "ignored_reason": "invalid_issue",
                        "validation_errors": issue_errors,
                    })
                    continue
                issue_key = (category, _normalize_span(segment))
                if issue_key in seen_issue_keys:
                    ignored_issue_details.append({
                        **preliminary_detail,
                        "confidence": confidence,
                        "ignored_reason": "duplicate_issue",
                    })
                    continue
                seen_issue_keys.add(issue_key)
                rationale_truncated = len(explanation) > _MAX_RATIONALE_CHARS
                if rationale_truncated:
                    explanation = explanation[:_MAX_RATIONALE_CHARS].rstrip()

                detail = {
                    "issue_id": preliminary_detail["issue_id"],
                    "category": category,
                    "severity": severity,
                    "confidence": confidence,
                    "source_quote": segment,
                    "source_segment_id": resolved_segment_id,
                    "current_persian_quote": current,
                    "suggested_correction": fix,
                    "rationale": explanation,
                    # Legacy aliases retained for policy filters and old clients.
                    "source_segment": segment,
                    "current_translation": current,
                    "suggested_fix": fix,
                    "explanation": explanation,
                }
                if fix_orthography_edits:
                    detail["suggestion_orthography_normalized"] = (
                        fix_orthography_edits
                    )
                if "issue_source_segment_id_corrected" in segment_errors:
                    detail["source_segment_id_corrected"] = True
                risks = concept_risks(detail)
                if risks:
                    detail["risk_flags"] = risks
                if rationale_truncated:
                    detail["rationale_truncated"] = True
                text = f"[{severity.upper()}/{category}]"
                text += f' source: "{segment}" | current: "{current}"'
                text += f" | fix: {fix} (Reason: {explanation})"
                detail["formatted"] = text
                parsed.append((severity_rank.get(severity, 2), text, detail))
            parsed.sort(key=lambda item: item[0])
            if len(parsed) > _MAX_MQM_ISSUES:
                for _, _, detail in parsed[_MAX_MQM_ISSUES:]:
                    ignored_issue_details.append({
                        **detail, "ignored_reason": "compact_issue_limit",
                    })
                parsed = parsed[:_MAX_MQM_ISSUES]
            issues = [text for _, text, _ in parsed]
            issue_details = [detail for _, _, detail in parsed]
            if invalid_issue_count and not parsed:
                for detail in ignored_issue_details:
                    errors.extend(detail.get("validation_errors", []))
                errors.append("all_critic_issues_invalid")
        else:
            issue_details = []
            ignored_issue_details = []
            errors.append("issues_must_be_array")

        coverage_checked_segment_ids: list[str] = []
        uncovered_source_segment_ids: list[str] = []
        coverage_complete = True
        coverage = data.get("source_coverage")
        if isinstance(coverage, dict):
            raw_checked = coverage.get("checked_source_segment_ids", [])
            raw_uncovered = coverage.get("uncovered_source_segment_ids", [])
            coverage_complete = bool(coverage.get("complete", False))
            if isinstance(raw_checked, list):
                coverage_checked_segment_ids = [
                    str(value).strip() for value in raw_checked
                    if str(value).strip()
                ]
            else:
                errors.append("coverage_checked_ids_must_be_array")
            if isinstance(raw_uncovered, list):
                uncovered_source_segment_ids = [
                    str(value).strip() for value in raw_uncovered
                    if str(value).strip()
                ]
            else:
                errors.append("coverage_uncovered_ids_must_be_array")
        elif require_coverage:
            coverage_complete = False
            errors.append("source_coverage_required")

        if require_coverage:
            expected_ids = [
                str(segment["segment_id"])
                for segment in source_segments(source_text)
            ]
            if (
                len(coverage_checked_segment_ids)
                != len(set(coverage_checked_segment_ids))
            ):
                errors.append("coverage_checked_ids_duplicated")
            if set(coverage_checked_segment_ids) != set(expected_ids):
                errors.append("coverage_checked_ids_incomplete")
            if not set(uncovered_source_segment_ids) <= set(expected_ids):
                errors.append("coverage_uncovered_ids_unknown")
            issue_segment_ids = {
                str(detail.get("source_segment_id", ""))
                for detail in issue_details
                if str(detail.get("category", "")) in {"omission", "accuracy"}
            }
            if not set(uncovered_source_segment_ids) <= issue_segment_ids:
                errors.append("coverage_gap_without_grounded_issue")
            if coverage_complete == bool(uncovered_source_segment_ids):
                errors.append("coverage_complete_flag_inconsistent")

        return CritiqueResult(
            accuracy=accuracy,
            fluency=fluency,
            terminology=terminology,
            register=register,
            average=average,
            issues=issues,
            issue_details=issue_details,
            ignored_issue_details=ignored_issue_details,
            coverage_checked_segment_ids=coverage_checked_segment_ids,
            uncovered_source_segment_ids=uncovered_source_segment_ids,
            coverage_complete=coverage_complete,
            raw_response=raw,
            valid=not errors,
            validation_errors=errors,
        )


def _normalized_issue_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split()).strip(" .;:!?\"'")


def _is_noop_issue(detail: dict[str, Any]) -> bool:
    """Ignore advice that cannot produce a real textual edit."""
    current = _normalized_issue_text(detail.get("current_translation"))
    suggested = _normalized_issue_text(detail.get("suggested_fix"))
    explanation = _normalized_issue_text(detail.get("explanation"))
    if current and suggested and current == suggested:
        return True
    no_change_values = {
        "no change needed",
        "no correction needed",
        "keep as is",
        "preserve as is",
        "none",
        "n/a",
    }
    if suggested in no_change_values:
        return True
    return (
        (not suggested or suggested == current)
        and any(
            phrase in explanation
            for phrase in (
                "no change needed",
                "no correction needed",
                "already correct",
                "current translation is acceptable",
                "keep the current",
            )
        )
    )


def _persian_terminology_key(value: Any) -> str:
    """Compare lexical Persian forms without treating joiners as semantics."""
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    normalized = normalized.replace("ي", "ی").replace("ك", "ک")
    normalized = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", normalized)
    return re.sub(r"[\s\u200c\u200d]+", "", normalized).strip(" .;:!?\"'")


def _is_terminology_orthography_equivalent(detail: dict[str, Any]) -> bool:
    """Ignore terminology advice that changes only Persian joining/spacing."""
    if str(detail.get("category", "")).strip().lower() != "terminology":
        return False
    current = str(detail.get("current_translation", "")).strip()
    suggested = str(detail.get("suggested_fix", "")).strip()
    return bool(
        current
        and suggested
        and current != suggested
        and _persian_terminology_key(current)
        == _persian_terminology_key(suggested)
    )


def _normalize_span(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = normalized.replace("\u200c", " ")
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _span_is_grounded(quote: str, text: str) -> bool:
    return bool(quote and _normalize_span(quote) in _normalize_span(text))


def _stable_issue_id(category: str, source_quote: str) -> str:
    material = f"{category}\0{_normalize_span(source_quote)}".encode("utf-8")
    return "mqm-" + hashlib.sha256(material).hexdigest()[:12]


def _confidence_value(value: Any, errors: list[str]) -> float:
    if value is None or str(value).strip() == "":
        # Confidence is advisory, so an otherwise grounded issue remains usable
        # when an older model omits it.
        return 0.5
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        errors.append("issue_confidence_must_be_numeric")
        return 0.0
    if not 0.0 <= confidence <= 1.0:
        errors.append("issue_confidence_must_be_between_0_and_1")
        return 0.0
    return confidence


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
