"""Build first-occurrence term-note metadata after translation assembly."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from tarjomeh.exporters.base import TranslatedDocument
from tarjomeh.glossary.manager import GlossaryManager
from tarjomeh.persian.typography import PersianTypographer


NOTE_CAPABLE_FORMATS = {"docx", "epub", "markdown"}
_DIGIT_VARIANTS = {
    "0": "0\u06f0\u0660", "1": "1\u06f1\u0661", "2": "2\u06f2\u0662",
    "3": "3\u06f3\u0663", "4": "4\u06f4\u0664", "5": "5\u06f5\u0665",
    "6": "6\u06f6\u0666", "7": "7\u06f7\u0667", "8": "8\u06f8\u0668",
    "9": "9\u06f9\u0669",
}


def effective_term_notes_mode(mode: str, output_format: str) -> str:
    """Fall back to inline originals when an exporter cannot render notes."""
    if mode != "inline" and output_format.lower() not in NOTE_CAPABLE_FORMATS:
        return "inline"
    return mode


def _source_original_pattern(source: str) -> re.Pattern[str]:
    """Match a Latin original despite harmless spacing/digit normalization."""
    parts: list[str] = []
    previous_kind = ""
    for char in source.strip():
        if char.isspace():
            if not parts or parts[-1] != r"\s+":
                parts.append(r"\s+")
            previous_kind = "space"
            continue
        kind = "digit" if char in _DIGIT_VARIANTS else "letter" if char.isalpha() else "other"
        if {kind, previous_kind} == {"digit", "letter"}:
            parts.append(r"\s*")
        if kind == "digit":
            parts.append(f"[{_DIGIT_VARIANTS[char]}]")
        else:
            parts.append(re.escape(char))
        previous_kind = kind
    return re.compile(rf"(?<!\w){''.join(parts)}(?!\w)", re.IGNORECASE)


def _inside_parenthetical(text: str, offset: int) -> bool:
    """Return true when an offset is inside an unmatched round parenthesis."""
    return text.rfind("(", 0, offset) > text.rfind(")", 0, offset)


def ensure_inline_proper_noun_originals(
    document: TranslatedDocument,
    proper_nouns: dict[str, str],
    typographer: PersianTypographer,
    categories: dict[str, str] | None = None,
    *,
    aliases: dict[str, list[str]] | None = None,
    return_report: bool = False,
) -> int | dict[str, Any]:
    """Anchor one English original to its exact source occurrence."""
    inserted = 0
    repositioned = 0
    paired_repaired = 0
    seen: set[str] = set()
    anchors: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    missing_targets: list[dict[str, Any]] = []
    categories = categories or {}
    aliases = aliases or {}
    for paragraph in document.paragraphs:
        candidates = []
        for source, raw_target in proper_nouns.items():
            key = source.casefold().strip()
            if not key or key in seen:
                continue
            category = str(categories.get(source, "proper_noun")).lower()
            flags = 0 if category == "publication" else re.IGNORECASE
            source_match = re.search(
                rf"(?<!\w){re.escape(source)}(?!\w)",
                paragraph.source_text,
                flags=flags,
            )
            if source_match is None:
                continue
            target = typographer.process(raw_target).strip()
            target_variants = [target] + [
                typographer.process(value).strip()
                for value in aliases.get(source, [])
                if str(value).strip()
            ]
            target_variants = list(dict.fromkeys(
                value for value in target_variants if value
            ))
            if target_variants:
                candidates.append(
                    (
                        source_match.start(), source, target_variants,
                        key, category,
                    )
                )

        text = paragraph.translated_text
        for source_position, source, target_variants, key, category in sorted(candidates):
            target_matches = [
                (match.start(), target)
                for target in target_variants
                for match in re.finditer(re.escape(target), text)
            ]
            target_offsets = sorted({offset for offset, _target in target_matches})
            target = target_variants[0]
            if not target_offsets:
                bare_originals = [
                    match
                    for match in _source_original_pattern(source).finditer(text)
                    if not _inside_parenthetical(text, match.start())
                ]
                if len(bare_originals) == 1:
                    original_match = bare_originals[0]
                    text = (
                        text[:original_match.start()]
                        + f"{target} ({source})"
                        + text[original_match.end():]
                    )
                    paired_repaired += 1
                    target_offsets = [original_match.start()]
                    target_matches = [(original_match.start(), target)]
                elif len(bare_originals) > 1:
                    ambiguous.append({
                        "paragraph_index": paragraph.index,
                        "source": source,
                        "target": target,
                        "category": category,
                        "reason": "ambiguous_english_only_occurrence",
                    })
                    continue
                else:
                    missing_targets.append({
                        "paragraph_index": paragraph.index,
                        "source": source,
                        "target": target,
                        "category": category,
                        "reason": "known_target_not_found",
                    })
                    continue
            expected = int(
                source_position / max(len(paragraph.source_text), 1) * len(text)
            )
            ranked = sorted(target_offsets, key=lambda value: abs(value - expected))
            if (
                len(ranked) > 1
                and abs(abs(ranked[0] - expected) - abs(ranked[1] - expected)) <= 3
            ):
                ambiguous.append({
                    "paragraph_index": paragraph.index,
                    "source": source,
                    "target": target,
                    "category": category,
                    "reason": "ambiguous_target_occurrence",
                })
                continue
            target_offset = ranked[0]
            target = max(
                [
                    value for offset, value in target_matches
                    if offset == target_offset
                ],
                key=len,
            )
            target_end = target_offset + len(target)
            insertion_at = target_end
            if insertion_at < len(text) and text[insertion_at] in "»”":
                insertion_at += 1
            original_re = re.compile(
                rf"\s*\(\s*{re.escape(source)}\s*\)", re.IGNORECASE
            )
            existing = list(original_re.finditer(text))
            correctly_placed = any(
                abs(match.start() - insertion_at) <= 2 for match in existing
            )
            if not correctly_placed:
                if existing:
                    repositioned += 1
                    text = original_re.sub("", text)
                    target_offsets = [
                        match.start() for match in re.finditer(re.escape(target), text)
                    ]
                    if not target_offsets:
                        continue
                    target_offset = min(
                        target_offsets, key=lambda value: abs(value - expected)
                    )
                    insertion_at = target_offset + len(target)
                    if insertion_at < len(text) and text[insertion_at] in "»”":
                        insertion_at += 1
                text = (
                    text[:insertion_at]
                    + f" ({source})"
                    + text[insertion_at:]
                )
                inserted += 1
            elif len(existing) > 1:
                kept = False

                def dedupe(match: re.Match[str]) -> str:
                    nonlocal kept
                    if not kept and abs(match.start() - insertion_at) <= 2:
                        kept = True
                        return match.group(0)
                    return ""

                text = original_re.sub(dedupe, text)
            anchors.append({
                "paragraph_index": paragraph.index,
                "source": source,
                "target": target,
                "category": category,
                "source_offset": source_position,
                "target_offset": target_offset,
                "repositioned": bool(existing and not correctly_placed),
            })
            seen.add(key)
        paragraph.translated_text = text
    report = {
        "inserted_count": inserted,
        "repositioned_count": repositioned,
        "paired_repair_count": paired_repaired,
        "anchored_count": len(anchors),
        "ambiguous_count": len(ambiguous),
        "missing_target_count": len(missing_targets),
        "anchors": anchors,
        "ambiguous": ambiguous,
        "missing_targets": missing_targets,
    }
    return report if return_report else inserted


_ADJACENT_ORIGINAL_CITATION_RE = re.compile(
    r"\((?P<original>[A-Za-zÀ-ž][^()]{0,120}?)\)\s+"
    r"\((?P<citation>[^()]*\d[^()]*)\)"
)
_ORIGINAL_BARE_YEAR_RE = re.compile(
    r"\((?P<original>[A-Za-zÀ-ž][^()]{0,120}?)\)\s+"
    r"(?P<citation>(?:1[5-9]|20)\d{2}[a-z]?(?:\s*,\s*\d+(?:[-–]\d+)?)?)"
)


def normalize_adjacent_original_citations(
    document: TranslatedDocument,
    authorized_originals: dict[str, str],
) -> dict[str, Any]:
    """Merge only source-grounded adjacent original and citation spans."""
    authorized = {source.casefold().strip() for source in authorized_originals}
    changes: list[dict[str, Any]] = []
    for paragraph in document.paragraphs:
        source_folded = (paragraph.source_text or "").casefold()

        def replace(match: re.Match[str]) -> str:
            original = " ".join(match.group("original").split())
            citation = " ".join(match.group("citation").split())
            if original.casefold() not in authorized:
                return match.group(0)
            if original.casefold() not in source_folded:
                return match.group(0)
            if f"({citation})".casefold() not in source_folded:
                return match.group(0)
            combined = f"({original}, {citation})"
            changes.append({
                "paragraph_index": paragraph.index,
                "before": match.group(0),
                "after": combined,
                "original": original,
                "citation": citation,
            })
            return combined

        paragraph.translated_text = _ADJACENT_ORIGINAL_CITATION_RE.sub(
            replace, paragraph.translated_text or ""
        )

        def replace_bare_year(match: re.Match[str]) -> str:
            original = " ".join(match.group("original").split())
            citation = " ".join(match.group("citation").split())
            if original.casefold() not in authorized:
                return match.group(0)
            source_pattern = re.compile(
                rf"{re.escape(original)}\s+{re.escape(citation)}",
                re.IGNORECASE,
            )
            if not source_pattern.search(paragraph.source_text or ""):
                return match.group(0)
            combined = f"({original}, {citation})"
            changes.append({
                "paragraph_index": paragraph.index,
                "before": match.group(0),
                "after": combined,
                "original": original,
                "citation": citation,
            })
            return combined

        paragraph.translated_text = _ORIGINAL_BARE_YEAR_RE.sub(
            replace_bare_year, paragraph.translated_text or ""
        )
    return {"normalized_count": len(changes), "changes": changes}


_LATIN_PARENTHETICAL_RE = re.compile(r"\s*\(([^()\n]{1,160})\)")
_ORIGINAL_PUNCTUATION_MAP = str.maketrans({
    "\u060c": ",",
    "\u061b": ";",
    "\uff0c": ",",
    "\uff1a": ":",
    "\u2013": "-",
    "\u2014": "-",
})


def _original_identity(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = normalized.translate(_ORIGINAL_PUNCTUATION_MAP)
    normalized = re.sub(r"\s*([,:;\-])\s*", r"\1", normalized)
    return " ".join(normalized.casefold().split()).strip()


def audit_inline_english_originals(
    document: TranslatedDocument,
    authorized_originals: dict[str, str],
) -> dict[str, Any]:
    """Remove grounded unauthorized originals and deduplicate authorized ones.

    Numeric and source-authored parentheticals are treated as scholarly
    apparatus and are always preserved.
    """
    authorized = {
        _original_identity(source): source
        for source in authorized_originals
        if source.strip()
    }
    seen: set[str] = set()
    removed_unauthorized: list[dict[str, Any]] = []
    removed_duplicates: list[dict[str, Any]] = []
    preserved_citations = 0

    for paragraph in document.paragraphs:
        source_text = paragraph.source_text or ""
        source_folded = source_text.casefold()

        def replace(match: re.Match[str]) -> str:
            nonlocal preserved_citations
            content = " ".join(match.group(1).split()).strip()
            if not re.search(r"[A-Za-z]", content):
                return match.group(0)

            key = _original_identity(content)
            exact_source_parenthetical = f"({content})".casefold() in source_folded
            if any(char.isdigit() for char in content) or exact_source_parenthetical:
                leading_original = _original_identity(content.split(",", 1)[0])
                if leading_original in authorized:
                    seen.add(leading_original)
                preserved_citations += 1
                return match.group(0)

            if key in authorized:
                if key not in seen:
                    seen.add(key)
                    return match.group(0)
                removed_duplicates.append({
                    "paragraph_index": paragraph.index,
                    "original": content,
                })
                return ""

            grounded = re.search(
                rf"(?<!\w){re.escape(content)}(?!\w)",
                source_text,
                flags=re.IGNORECASE,
            )
            if grounded:
                removed_unauthorized.append({
                    "paragraph_index": paragraph.index,
                    "original": content,
                })
                return ""
            return match.group(0)

        cleaned = _LATIN_PARENTHETICAL_RE.sub(
            replace, paragraph.translated_text or ""
        )
        paragraph.translated_text = re.sub(r" {2,}", " ", cleaned).strip()

    return {
        "authorized_count": len(authorized),
        "kept_authorized": len(seen),
        "removed_unauthorized_count": len(removed_unauthorized),
        "removed_duplicate_count": len(removed_duplicates),
        "preserved_citation_count": preserved_citations,
        "removed_unauthorized": removed_unauthorized,
        "removed_duplicates": removed_duplicates,
    }


def apply_term_notes(
    document: TranslatedDocument,
    glossary_manager: GlossaryManager,
    proper_nouns: dict[str, str],
    typographer: PersianTypographer,
    *,
    domain: str = "",
    mode: str = "inline",
    extra_terms: dict[str, str] | None = None,
    aliases: dict[str, list[str]] | None = None,
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
            target_variants = [typographer.process(raw_target).strip()] + [
                typographer.process(value).strip()
                for value in (aliases or {}).get(source, [])
                if str(value).strip()
            ]
            target_variants = list(dict.fromkeys(
                value for value in target_variants if value
            ))
            if not target_variants:
                continue
            target_matches: list[tuple[int, str]] = []
            for target in target_variants:
                offset = paragraph.translated_text.find(target)
                while offset >= 0:
                    if offset not in used_target_offsets:
                        target_matches.append((offset, target))
                    offset = paragraph.translated_text.find(
                        target, offset + len(target)
                    )
            target_offsets = sorted({offset for offset, _target in target_matches})
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
            target = max(
                [
                    value for offset, value in target_matches
                    if offset == target_offset
                ],
                key=len,
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
