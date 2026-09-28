"""Source-paragraph matching for explicitly reviewed book terminology."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from tarjomeh.core.paragraph_protocol import split_paragraphs
from tarjomeh.glossary.compliance import ComplianceReport, Violation, target_present

_BODY_ROLES = frozenset({
    "body", "academic_argument", "expository_nonfiction",
    "narrative_prose", "dialogue",
})
_PUBLISHER_LINE = re.compile(
    r"(?:\b(?:press|publish(?:er|ing)?|books|ltd|inc|copyright|isbn)\b|\u00a9)",
    re.IGNORECASE,
)


def _verified_source_paragraphs(text: str, metadata: dict[str, Any]) -> list[str] | None:
    spans = metadata.get("source_paragraph_spans")
    hashes = metadata.get("source_paragraph_hashes")
    indices = metadata.get("paragraph_indices")
    if isinstance(spans, list) and isinstance(hashes, list) and isinstance(indices, list):
        if not spans or not (len(spans) == len(hashes) == len(indices)):
            return None
        paragraphs = []
        for span, digest in zip(spans, hashes, strict=True):
            if (not isinstance(span, list) or len(span) != 2
                    or not all(isinstance(value, int) for value in span)):
                return None
            start, end = span
            if start < 0 or end <= start or end > len(text):
                return None
            value = text[start:end]
            if hashlib.sha256(value.encode("utf-8")).hexdigest() != digest:
                return None
            paragraphs.append(value)
        return paragraphs
    paragraphs = split_paragraphs(text)
    return paragraphs if len(paragraphs) == 1 else None


def resolve_reviewed_book_terms(
    source_text: str,
    metadata: dict[str, Any],
    chapter_position: int,
    decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return exact in-scope matches and review-only uncertain occurrences."""
    relevant = [
        term for term in decisions
        if term.get("status") == "approved"
        and str(term.get("source", "")).strip()
        and (not term.get("chapter_positions")
             or chapter_position in term["chapter_positions"])
        and re.search(
            rf"(?<!\w){re.escape(str(term.get('source', '')))}(?!\w)",
            source_text, re.IGNORECASE,
        )
    ]
    if not relevant:
        return [], []
    paragraphs = _verified_source_paragraphs(source_text, metadata)
    if paragraphs is None:
        return [], [{"reason": "uncertain_paragraph_alignment"}]
    roles = metadata.get("structural_roles")
    if not isinstance(roles, list) or len(roles) != len(paragraphs):
        return [], [{"reason": "uncertain_paragraph_role"}]
    matches: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    for paragraph_index, (paragraph, role) in enumerate(zip(paragraphs, roles, strict=True)):
        candidates: dict[str, list[dict[str, Any]]] = {}
        for term in relevant:
            source = str(term.get("source", "")).strip()
            if not source:
                continue
            scope_mode = str(term.get("scope_mode", "evidence_paragraph"))
            if scope_mode == "evidence_paragraph" and (
                hashlib.sha256(paragraph.encode("utf-8")).hexdigest()
                != str(term.get("source_evidence_sha256", ""))
            ):
                continue
            if scope_mode not in {"evidence_paragraph", "all_body"}:
                review.append({"source": source, "reason": "invalid_approved_scope"})
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", re.IGNORECASE)
            if pattern.search(paragraph):
                candidates.setdefault(source.casefold(), []).append(term)
        for source_key, options in candidates.items():
            scope_reason = ""
            if role not in _BODY_ROLES:
                scope_reason = "non_body_paragraph"
            elif len(paragraph) < 140 and _PUBLISHER_LINE.search(paragraph):
                scope_reason = "possible_publisher_line"
            elif len(options) != 1:
                scope_reason = "ambiguous_approved_sense"
            term = options[0]
            if term.get("keep_original") and str(term.get("source", "")) not in paragraph:
                scope_reason = "original_not_exact_in_source"
            item = {
                "paragraph_index": paragraph_index,
                "source": term["source"],
                "target": term.get("target", ""),
                "keep_original": bool(term.get("keep_original")),
                "source_paragraph_hash": hashlib.sha256(paragraph.encode("utf-8")).hexdigest(),
                "sense_id": str(term.get("sense_id", "")),
            }
            if scope_reason:
                review.append({**item, "reason": scope_reason})
            else:
                matches.append(item)
    return matches, review


def check_reviewed_book_terms(
    translation: str,
    source_text: str,
    metadata: dict[str, Any],
    matches: list[dict[str, Any]],
) -> tuple[ComplianceReport, list[dict[str, Any]]]:
    """Check only proven source/target paragraph pairs, never the whole chunk."""
    source_parts = _verified_source_paragraphs(source_text, metadata)
    target_parts = split_paragraphs(translation)
    if source_parts is None or len(source_parts) != len(target_parts):
        return ComplianceReport(), [{"reason": "uncertain_target_paragraph_alignment"}]
    violations: list[Violation] = []
    for match in matches:
        index = match["paragraph_index"]
        if index >= len(source_parts) or (
            hashlib.sha256(source_parts[index].encode("utf-8")).hexdigest()
            != match["source_paragraph_hash"]
        ):
            return ComplianceReport(), [{"reason": "source_paragraph_changed"}]
        if not target_present(str(match["target"]), target_parts[index]):
            violations.append(Violation(
                term=str(match["source"]),
                expected=str(match["target"]),
                chunk_location=f"Paragraph {index + 1}",
            ))
    return ComplianceReport(violations=violations, total_checked=len(matches)), []


def format_reviewed_book_terms(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return ""
    lines = [
        "Approved book terms. Apply each rendering only in its named source "
        "paragraph and sense; preserve all other paragraphs and the source meaning:"
    ]
    for match in matches:
        lines.append(
            f"- Source paragraph {match['paragraph_index'] + 1}: "
            f"{match['source']} -> {match['target']}"
        )
    return "\n".join(lines)
