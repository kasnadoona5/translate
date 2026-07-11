"""Build first-occurrence term-note metadata after translation assembly."""

from __future__ import annotations

from typing import Any

from tarjomeh.exporters.base import TranslatedDocument
from tarjomeh.glossary.manager import GlossaryManager
from tarjomeh.persian.typography import PersianTypographer


def apply_term_notes(
    document: TranslatedDocument,
    glossary_manager: GlossaryManager,
    proper_nouns: dict[str, str],
    typographer: PersianTypographer,
    *,
    domain: str = "",
    mode: str = "inline",
    extra_terms: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Attach stable, exporter-neutral note metadata to a translated book."""
    if mode == "inline":
        return []

    notes: list[dict[str, Any]] = []
    seen: set[str] = set()
    next_number = 1

    for paragraph in document.paragraphs:
        candidates: list[tuple[int, str, str, str]] = []
        source_folded = paragraph.source_text.casefold()

        for entry in glossary_manager.find_terms(
            paragraph.source_text,
            domain=domain,
        ):
            position = source_folded.find(entry.source.casefold())
            candidates.append((position, entry.source, entry.target, "glossary"))

        for source, target in proper_nouns.items():
            needle = source.casefold()
            position = source_folded.find(needle)
            if position < 0:
                continue
            before = source_folded[position - 1] if position else ""
            end = position + len(needle)
            after = source_folded[end] if end < len(source_folded) else ""
            if (not before or not before.isalnum()) and (
                not after or not after.isalnum()
            ):
                candidates.append((position, source, target, "proper_noun"))

        for source, target in (extra_terms or {}).items():
            position = source_folded.find(source.casefold())
            if position >= 0:
                candidates.append((position, source, target, "persisted"))

        refs: list[dict[str, Any]] = []
        used_target_offsets: set[int] = set()
        for source_position, source, raw_target, kind in sorted(
            candidates,
            key=lambda item: (item[0], -len(item[1])),
        ):
            key = source.casefold().strip()
            if not key or key in seen:
                continue
            target = typographer.process(raw_target).strip()
            if not target:
                continue
            target_offsets = []
            offset = paragraph.translated_text.find(target)
            while offset >= 0:
                if offset not in used_target_offsets:
                    target_offsets.append(offset)
                offset = paragraph.translated_text.find(target, offset + len(target))
            if not target_offsets:
                continue
            expected_offset = int(
                max(source_position, 0)
                / max(len(paragraph.source_text), 1)
                * len(paragraph.translated_text)
            )
            target_offset = min(
                target_offsets,
                key=lambda value: abs(value - expected_offset),
            )

            note = {
                "number": next_number,
                "original": source,
                "transliteration": target,
                "kind": kind,
            }
            refs.append({
                "number": next_number,
                "target": target,
                "offset": target_offset,
            })
            notes.append(note)
            seen.add(key)
            used_target_offsets.add(target_offset)
            next_number += 1

        if refs:
            refs.sort(key=lambda ref: int(ref.get("offset", 0)))
            paragraph.metadata = dict(paragraph.metadata)
            paragraph.metadata["term_note_refs"] = refs

    if notes:
        document.metadata = dict(document.metadata)
        document.metadata["term_notes"] = notes
        document.metadata["term_notes_mode"] = mode
    return notes
