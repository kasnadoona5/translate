"""Shared helpers for deterministic first-occurrence term notes."""

from __future__ import annotations

from typing import Any


def paragraph_note_parts(
    text: str,
    metadata: dict[str, Any],
) -> list[tuple[str, dict[str, Any] | None]]:
    """Split text at note targets; each reference belongs after its segment."""
    refs = metadata.get("term_note_refs", [])
    if not isinstance(refs, list) or not refs:
        return [(text, None)]

    parts: list[tuple[str, dict[str, Any] | None]] = []
    cursor = 0
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        target = str(ref.get("target", ""))
        if not target:
            continue
        preferred = ref.get("offset")
        pos = (
            int(preferred)
            if isinstance(preferred, int)
            and preferred >= cursor
            and text[preferred:preferred + len(target)] == target
            else text.find(target, cursor)
        )
        if pos < 0:
            continue
        end = pos + len(target)
        parts.append((text[cursor:end], ref))
        cursor = end
    parts.append((text[cursor:], None))
    return parts


def document_term_notes(document: Any) -> list[dict[str, Any]]:
    notes = document.metadata.get("term_notes", [])
    return notes if isinstance(notes, list) else []
