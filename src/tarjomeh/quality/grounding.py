"""Stable source-segment grounding and non-blocking concept-risk evidence."""

from __future__ import annotations

import re
from typing import Any


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(\[]*[A-Z0-9])")

_HIGH_RISK_CONCEPTS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\bprospectiv(?:e|ely)\b", re.I), "prospectively", "temporal_orientation"),
    (re.compile(r"\bground\b", re.I), "ground", "conceptual_polysemy"),
    (re.compile(r"\boperations?\b", re.I), "operation", "domain_term"),
    (re.compile(r"\bimplications?\b", re.I), "implication", "conceptual_polysemy"),
    (re.compile(r"\bengagements?\b", re.I), "engagement", "context_sensitive_relation"),
    (re.compile(r"\bconstellations?\b", re.I), "constellation", "metaphorical_term"),
    (re.compile(r"\bfield\b", re.I), "field", "domain_polysemy"),
    (re.compile(r"\bcapital\b", re.I), "capital", "core_theoretical_term"),
)


def source_segments(source_text: str) -> list[dict[str, Any]]:
    """Split source paragraphs into stable, compact sentence identifiers."""
    segments: list[dict[str, Any]] = []
    offset = 0
    paragraphs = source_text.split("\n\n")
    for paragraph_number, paragraph in enumerate(paragraphs, 1):
        stripped = paragraph.strip()
        paragraph_start = source_text.find(stripped, offset) if stripped else -1
        if paragraph_start < 0:
            paragraph_start = offset
        offset = paragraph_start + len(stripped)
        sentences = [part.strip() for part in _SENTENCE_BREAK.split(stripped) if part.strip()]
        if not sentences and stripped:
            sentences = [stripped]
        search_from = 0
        for sentence_number, sentence in enumerate(sentences, 1):
            local_start = stripped.find(sentence, search_from)
            if local_start < 0:
                local_start = search_from
            local_end = local_start + len(sentence)
            search_from = local_end
            segments.append({
                "segment_id": f"p{paragraph_number}:s{sentence_number}",
                "text": sentence,
                "start": paragraph_start + local_start,
                "end": paragraph_start + local_end,
            })
    return segments


def indexed_source(source_text: str) -> str:
    """Render source once with sentence IDs, avoiding duplicate prompt context."""
    return "\n".join(
        f"[{segment['segment_id']}] {segment['text']}"
        for segment in source_segments(source_text)
    ) or source_text


def resolve_source_segment(
    source_text: str,
    source_quote: str,
    requested_id: str = "",
) -> tuple[str, list[str]]:
    """Resolve and validate one quote against a stable source segment."""
    segments = source_segments(source_text)
    by_id = {segment["segment_id"]: segment for segment in segments}
    quote_folded = " ".join((source_quote or "").casefold().split())
    errors: list[str] = []

    requested_error = ""
    if requested_id:
        segment = by_id.get(requested_id)
        if segment is not None:
            segment_folded = " ".join(str(segment["text"]).casefold().split())
            if not quote_folded or quote_folded in segment_folded:
                return requested_id, []
            requested_error = "issue_source_segment_id_corrected"
        else:
            requested_error = "issue_source_segment_id_corrected"

    matches = []
    for segment in segments:
        segment_folded = " ".join(str(segment["text"]).casefold().split())
        if quote_folded and quote_folded in segment_folded:
            matches.append(str(segment["segment_id"]))
    if len(matches) == 1:
        return matches[0], [requested_error] if requested_error else []
    if len(matches) > 1:
        return matches[0], ["issue_source_segment_ambiguous"]
    errors = ["issue_source_segment_unresolved"]
    if requested_error:
        errors.insert(0, requested_error)
    return "", errors


def concept_risks(issue: dict[str, Any]) -> list[dict[str, str]]:
    """Return review-only risks for a grounded critic issue."""
    source_quote = str(issue.get("source_quote", ""))
    risks: list[dict[str, str]] = []
    for pattern, term, reason in _HIGH_RISK_CONCEPTS:
        if pattern.search(source_quote):
            risks.append({"term": term, "reason": reason})

    severity = str(issue.get("severity", "")).lower()
    category = str(issue.get("category", "")).lower()
    if severity in {"critical", "major"} and category in {"accuracy", "terminology"}:
        risks.append({
            "term": source_quote[:120],
            "reason": "major_semantic_or_terminology_disagreement",
        })

    unique: dict[tuple[str, str], dict[str, str]] = {}
    for risk in risks:
        unique[(risk["term"].casefold(), risk["reason"])] = risk
    return list(unique.values())
