"""Build first-occurrence term-note metadata after translation assembly."""

from __future__ import annotations

from typing import Any

from tarjomeh.exporters.base import TranslatedDocument
from tarjomeh.glossary.manager import GlossaryManager
from tarjomeh.persian.typography import PersianTypographer


NOTE_CAPABLE_FORMATS = {"docx", "epub", "markdown"}


def effective_term_notes_mode(mode: str, output_format: str) -> str:
    """Fall back to inline originals when an exporter cannot render notes."""
    if mode != "inline" and output_format.lower() not in NOTE_CAPABLE_FORMATS:
        return "inline"
    return mode


def ensure_inline_proper_noun_originals(
    document: TranslatedDocument,
    proper_nouns: dict[str, str],
    typographer: PersianTypographer,
) -> int:
    """Deterministically retain first-occurrence English proper nouns."""
    inserted = 0
    seen: set[str] = set()
    for paragraph in document.paragraphs:
        source_folded = paragraph.source_text.casefold()
        candidates = []
        for source, raw_target in proper_nouns.items():
            key = source.casefold().strip()
            position = source_folded.find(key)
            if not key or key in seen or position < 0:
                continue
            target = typographer.process(raw_target).strip()
            if target:
                candidates.append((position, source, target, key))

        text = paragraph.translated_text
        for _, source, target, key in sorted(candidates):
            target_offset = text.find(target)
            if target_offset < 0:
                continue
            target_end = target_offset + len(target)
            insertion_at = target_end
            if insertion_at < len(text) and text[insertion_at] in "»”":
                insertion_at += 1
            nearby = text[insertion_at:insertion_at + len(source) + 8]
            if source.casefold() not in nearby.casefold():
                text = (
                    text[:insertion_at]
                    + f" ({source})"
                    + text[insertion_at:]
                )
                inserted += 1
            seen.add(key)
        paragraph.translated_text = text
    return inserted


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

    # Source and Persian word order can differ. Renumber by final marker order.
    ordered_refs = []
    for paragraph in document.paragraphs:
        ordered_refs.extend(paragraph.metadata.get("term_note_refs", []))
    old_to_new = {
        int(ref["number"]): number
        for number, ref in enumerate(ordered_refs, 1)
    }
    for ref in ordered_refs:
        ref["number"] = old_to_new[int(ref["number"])]
    for note in notes:
        note["number"] = old_to_new[int(note["number"])]
    notes.sort(key=lambda note: int(note["number"]))
    convert_numbers = bool(getattr(typographer, "_convert_numerals", True))
    for ref in ordered_refs:
        ref["display_number"] = (
            typographer.convert_numerals(str(ref["number"]))
            if convert_numbers else str(ref["number"])
        )
    for note in notes:
        note["display_number"] = next(
            ref["display_number"] for ref in ordered_refs
            if int(ref["number"]) == int(note["number"])
        )

    if notes:
        document.metadata = dict(document.metadata)
        document.metadata["term_notes"] = notes
        document.metadata["term_notes_mode"] = mode
    return notes
