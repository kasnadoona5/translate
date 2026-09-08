"""Stable paragraph identity for multi-paragraph LLM edits."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_MARKER_RE = re.compile(r"\[\[P(\d{4})\]\]")


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]


def marker_for(index: int) -> str:
    return f"[[P{index + 1:04d}]]"


def encode_paragraphs(text: str) -> tuple[str, list[str]]:
    return encode_paragraph_units(split_paragraphs(text))


def encode_paragraph_units(paragraphs: list[str]) -> tuple[str, list[str]]:
    """Encode an already authoritative ordered paragraph sequence."""
    markers = [marker_for(index) for index in range(len(paragraphs))]
    encoded = "\n\n".join(
        f"{marker}\n{paragraph}"
        for marker, paragraph in zip(markers, paragraphs, strict=True)
    )
    return encoded, markers


@dataclass
class ParagraphProtocolResult:
    valid: bool
    text: str = ""
    paragraphs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def decode_paragraphs(candidate: str, expected_markers: list[str]) -> ParagraphProtocolResult:
    """Validate exact marker identity/order and return marker-free paragraphs."""
    raw = (candidate or "").strip()
    if not expected_markers:
        return ParagraphProtocolResult(valid=bool(raw), text=raw, paragraphs=[raw] if raw else [])

    found = [f"[[P{value}]]" for value in _MARKER_RE.findall(raw)]
    errors: list[str] = []
    if found != expected_markers:
        if len(found) != len(set(found)):
            errors.append("duplicate_paragraph_marker")
        missing = [marker for marker in expected_markers if marker not in found]
        extra = [marker for marker in found if marker not in expected_markers]
        if missing:
            errors.append("missing_paragraph_marker")
        if extra:
            errors.append("unexpected_paragraph_marker")
        if not missing and not extra:
            errors.append("reordered_paragraph_marker")

    parts = _MARKER_RE.split(raw)
    prefix = parts[0].strip() if parts else ""
    if prefix:
        errors.append("text_before_first_marker")
    paragraphs: list[str] = []
    for offset in range(1, len(parts), 2):
        body = parts[offset + 1].strip() if offset + 1 < len(parts) else ""
        if not body:
            errors.append("empty_marked_paragraph")
        paragraphs.append(body)
    if len(paragraphs) != len(expected_markers):
        errors.append("paragraph_count_mismatch")
    if _MARKER_RE.search("\n\n".join(paragraphs)):
        errors.append("nested_paragraph_marker")

    unique_errors = list(dict.fromkeys(errors))
    return ParagraphProtocolResult(
        valid=not unique_errors,
        text="\n\n".join(paragraphs) if not unique_errors else "",
        paragraphs=paragraphs,
        errors=unique_errors,
    )


def contains_protocol_marker(text: str) -> bool:
    return bool(_MARKER_RE.search(text or ""))


def protocol_instruction(markers: list[str]) -> str:
    if not markers:
        return ""
    return (
        "\n\n### Paragraph identity protocol\n"
        "The labels " + ", ".join(markers) + " identify source paragraphs. "
        "Return every label exactly once, unchanged, and in the same order. "
        "Place each label on its own line immediately before that paragraph's "
        "complete Persian translation. Do not translate, discuss, duplicate, or "
        "omit labels. Output no text before the first label."
    )


def repair_prompt(
    candidate: str,
    markers: list[str],
    errors: list[str],
) -> str:
    return f"""Repair paragraph labels in the Persian translation below.
This is formatting repair only: do not retranslate, rewrite, shorten, expand, or
change any Persian wording. Required labels, in exact order:
{', '.join(markers)}
Validation errors: {', '.join(errors)}

Return every required label exactly once on its own line immediately before its
existing Persian paragraph. Return no commentary and no text before the first label.

Candidate translation:
{candidate}
"""
