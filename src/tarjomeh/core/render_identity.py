"""Audited identity between canonical text, the rendered document and the DOCX.

The canonical text is what the database and memory hold. Rendering may still
change delivered text through a fixed set of guarded producers (inline
originals, citation merges, identifier restoration and so on). Every such
change is recorded per paragraph with the producer that made it, proven by
replaying that producer on the same input, and checked by a type-specific
rule. The written DOCX is then read back with the exporter's own mapping.
Anything unknown or unproven blocks publication.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tarjomeh.exporters.base import TranslatedDocument

_ZWNJ = chr(0x200C)
_COMBINING_MARKS = "".join(chr(code) for code in range(0x064B, 0x0653)) + chr(0x0654)
_LATIN_EXTENDED = chr(0x00C0) + "-" + chr(0x024F)
_CLOSING_QUOTES = chr(0x00BB) + chr(0x201D)
_EN_DASH = chr(0x2013)

_LATIN_PARENTHETICAL_RE = re.compile(r"\(([^()\n]*[A-Za-z][^()\n]*)\)")
_ANY_PARENTHETICAL_RE = re.compile(r"\([^()\n]*\)")
_YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})[a-z]?\b")
_PAGE_RE = re.compile(r"\d+\s*[-" + _EN_DASH + r"]\s*\d+|:\s*\d+")
_LATIN_WORD_RE = re.compile(r"[A-Za-z" + _LATIN_EXTENDED + r"][A-Za-z" + _LATIN_EXTENDED + r"'-]*")
_PERSIAN_NOISE_RE = re.compile(r"[\s" + _ZWNJ + _COMBINING_MARKS + r"]")
_PERSIAN_LETTERS_RE = re.compile(
    "[" + chr(0x0621) + "-" + chr(0x063A) + chr(0x0641) + "-" + chr(0x064A)
    + chr(0x066E) + "-" + chr(0x06D3) + chr(0x06FA) + "-" + chr(0x06FF) + "]"
)
# Citation framing may be translated by house style; it is not an author name.
_CITATION_FRAMING_WORDS = frozenset({"see", "also", "cf", "eg", "ie", "ibid", "and"})

PERMITTED_RENDER_PRODUCERS = (
    "protocol_sanitation",
    "inline_original_anchor",
    "citation_merge",
    "original_fragment_reconciliation",
    "english_original_cleanup",
    "render_language_repair",
    "identifier_restoration",
)
# How far back from an inserted original its Persian rendering may start
# (the anchor may extend over attached suffixes such as a plural).
_ANCHOR_WINDOW = 24
_ANCHOR_SUFFIX = "[" + _PERSIAN_LETTERS_RE.pattern[1:-1] + "]{0,6}"


def _texts(document: TranslatedDocument) -> list[str]:
    return [paragraph.translated_text for paragraph in document.paragraphs]


def _squash(text: str) -> str:
    return " ".join((text or "").split())


def _outside_parentheses(text: str) -> str:
    """Return the text outside parentheses with all spacing removed.

    Inserting ``" (x)"`` before punctuation shifts a space, so spacing is not
    compared; every non-space character outside parentheses is.
    """
    return _PERSIAN_NOISE_RE.sub("", _ANY_PARENTHETICAL_RE.sub(" ", text or ""))


def _latin_parentheticals(text: str) -> Counter[str]:
    return Counter(
        _squash(match.group(1)) for match in _LATIN_PARENTHETICAL_RE.finditer(text or "")
    )


def _persian_key(text: str) -> str:
    return _PERSIAN_NOISE_RE.sub("", text or "")


def _persian_letters(text: str) -> str:
    return "".join(_PERSIAN_LETTERS_RE.findall(text or ""))


def _word_count(source: str, phrase: str) -> int:
    if not phrase:
        return 0
    pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
    return len(pattern.findall(source or ""))


def _proof_protocol(source: str, before: str, after: str, context: dict[str, Any]) -> str:
    if not after.strip() or after.strip() not in before:
        return "not_a_pure_wrapper_removal"
    return ""


def _anchor_targets(original: str, context: dict[str, Any]) -> list[str]:
    authorized = context.get("authorized", {}) or {}
    aliases = context.get("aliases", {}) or {}
    key = next(
        (name for name in authorized if name.casefold() == original.casefold()),
        None,
    )
    if key is None:
        return []
    return [
        value for value in [authorized[key], *(aliases.get(key, []) or [])]
        if str(value).strip()
    ]


def _anchor_position_proven(after: str, start: int, targets: list[str]) -> bool:
    """An inserted original must directly follow its Persian rendering.

    Only an attached suffix of a few letters (a plural, an ezafe yeh), spacing
    and closing quotes may stand between the rendering and the parenthesis.
    """
    window = after[max(0, start - len(max(targets, key=len)) - _ANCHOR_WINDOW):start]
    window_key = _persian_key(window.rstrip(" " + _CLOSING_QUOTES))
    return any(
        _persian_key(target)
        and re.search(
            re.escape(_persian_key(target)) + _ANCHOR_SUFFIX + "$", window_key
        )
        for target in targets
    )


def _proof_anchor(source: str, before: str, after: str, context: dict[str, Any]) -> str:
    before_originals = _latin_parentheticals(before)
    after_originals = _latin_parentheticals(after)
    touched = set((after_originals - before_originals) + (before_originals - after_originals))
    paired_repair = _outside_parentheses(before) != _outside_parentheses(after)
    for original in touched:
        targets = _anchor_targets(original, context)
        if not targets:
            return "anchor_original_not_authorized"
        if original.casefold() not in (source or "").casefold():
            return "anchor_original_not_exact_source_span"
        if after_originals[original] > max(1, _word_count(source, original)):
            return "anchor_count_exceeds_source"
        if after_originals[original] == 0 and before_originals[original] > 0:
            return "anchor_original_removed"
        for match in _LATIN_PARENTHETICAL_RE.finditer(after):
            if _squash(match.group(1)) != original:
                continue
            if not _anchor_position_proven(after, match.start(), targets):
                return "anchor_position_unproven"
    if paired_repair:
        # A paired repair turns one bare English original into
        # "<Persian> (<original>)"; nothing else outside parentheses changes.
        for original in touched:
            for target in _anchor_targets(original, context):
                repaired = re.sub(
                    rf"(?<!\w){re.escape(original)}(?!\w)",
                    target,
                    _ANY_PARENTHETICAL_RE.sub(" ", before),
                    count=1,
                    flags=re.IGNORECASE,
                )
                if _outside_parentheses(repaired) == _outside_parentheses(after):
                    return ""
        return "anchor_changed_text_outside_parentheses"
    return ""


def _citation_names(text: str) -> set[str]:
    return {
        word for word in _LATIN_WORD_RE.findall(text or "")
        if word[:1].isupper() and word.casefold() not in _CITATION_FRAMING_WORDS
    }


def _proof_citation(source: str, before: str, after: str, context: dict[str, Any]) -> str:
    """Citation merge and house style may move or translate framing only.

    The producer itself is proven by replay. This check adds that Persian
    prose outside the parentheses, every year, every page reference and every
    author name are unchanged.
    """
    if _persian_letters(_outside_parentheses(before)) != _persian_letters(
        _outside_parentheses(after)
    ):
        return "citation_changed_persian_text"
    if Counter(_YEAR_RE.findall(before)) != Counter(_YEAR_RE.findall(after)):
        return "citation_years_changed"
    if Counter(_squash(value) for value in _PAGE_RE.findall(before)) != Counter(
        _squash(value) for value in _PAGE_RE.findall(after)
    ):
        return "citation_pages_changed"
    if _citation_names(before) != _citation_names(after):
        return "citation_names_changed"
    return ""


def _removed_parentheticals(before: str, after: str) -> Counter[str] | None:
    """Return removed Latin parentheticals, or None if anything else changed."""
    if _outside_parentheses(before) != _outside_parentheses(after):
        return None
    before_originals = _latin_parentheticals(before)
    after_originals = _latin_parentheticals(after)
    if after_originals - before_originals:
        return None
    return before_originals - after_originals


def _proof_fragment(source: str, before: str, after: str, context: dict[str, Any]) -> str:
    removed = _removed_parentheticals(before, after)
    if removed is None:
        return "fragment_change_is_not_a_removal"
    remaining = [set(_LATIN_WORD_RE.findall(value)) for value in _latin_parentheticals(after)]
    for fragment in removed:
        tokens = set(_LATIN_WORD_RE.findall(fragment))
        if not any(tokens < full for full in remaining):
            return "fragment_not_covered_by_remaining_original"
    return ""


def _proof_cleanup(source: str, before: str, after: str, context: dict[str, Any]) -> str:
    removed = _removed_parentheticals(before, after)
    if removed is None:
        return "cleanup_change_is_not_a_removal"
    kept = _latin_parentheticals(after)
    for original in removed:
        # A source-grounded original may only be removed as a duplicate.
        if original.casefold() in (source or "").casefold() and kept[original] < 1:
            return "source_grounded_original_deleted"
    return ""


def _proof_language_repair(source: str, before: str, after: str, context: dict[str, Any]) -> str:
    from tarjomeh.quality.integrity import repair_source_grounded_language_artifacts

    candidate, _report = repair_source_grounded_language_artifacts(
        source, before, structural_role=str(context.get("role", "body"))
    )
    return "" if candidate == after else "language_repair_replay_mismatch"


def _proof_identifier(source: str, before: str, after: str, context: dict[str, Any]) -> str:
    from tarjomeh.quality.integrity import (
        extract_identifiers,
        extract_labeled_identifier_surfaces,
    )

    added = extract_identifiers(after) - extract_identifiers(before)
    if any(value not in extract_identifiers(source) for value in added):
        return "identifier_not_in_source"
    added_labels = (
        extract_labeled_identifier_surfaces(after)
        - extract_labeled_identifier_surfaces(before)
    )
    if any(value not in extract_labeled_identifier_surfaces(source) for value in added_labels):
        return "identifier_label_not_in_source"
    return ""


_TYPE_PROOFS: dict[str, Callable[[str, str, str, dict[str, Any]], str]] = {
    "protocol_sanitation": _proof_protocol,
    "inline_original_anchor": _proof_anchor,
    "citation_merge": _proof_citation,
    "original_fragment_reconciliation": _proof_fragment,
    "english_original_cleanup": _proof_cleanup,
    "render_language_repair": _proof_language_repair,
    "identifier_restoration": _proof_identifier,
}


class RenderChangeLedger:
    """Record, replay and prove every render-time change to delivered text."""

    def __init__(self, document: TranslatedDocument) -> None:
        self.document = document
        self.canonical = _texts(document)
        self.changes: list[dict[str, Any]] = []
        self.authorized: dict[str, str] = {}
        self.aliases: dict[str, list[str]] = {}

    def set_authorized(
        self,
        authorized: dict[str, str],
        aliases: dict[str, list[str]] | None = None,
    ) -> None:
        self.authorized = dict(authorized or {})
        self.aliases = {key: list(value) for key, value in dict(aliases or {}).items()}

    def step(self, producer: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run one render producer on the document and prove what it changed."""
        before_document = copy.deepcopy(self.document)
        before = _texts(self.document)
        result = function(self.document, *args, **kwargs)
        after = _texts(self.document)
        changed = [
            index for index, (old, new) in enumerate(zip(before, after, strict=True))
            if old != new
        ]
        if not changed:
            return result
        function(before_document, *args, **kwargs)
        replayed = _texts(before_document)
        proof = _TYPE_PROOFS.get(producer)
        for index in changed:
            paragraph = self.document.paragraphs[index]
            context = {
                "authorized": self.authorized,
                "aliases": self.aliases,
                "role": str(paragraph.metadata.get("structure_role", "body")),
            }
            failure = "unknown_render_producer" if proof is None else proof(
                paragraph.source_text, before[index], after[index], context
            )
            if replayed[index] != after[index]:
                failure = failure or "producer_replay_mismatch"
            self.changes.append({
                "producer": producer,
                "position": index,
                "paragraph_index": paragraph.index,
                "before": before[index],
                "after": after[index],
                "replay_verified": replayed[index] == after[index],
                "proof_failure": failure,
            })
        return result

    def audit(self) -> dict[str, Any]:
        """Check that every canonical-to-rendered difference has a proven chain."""
        final = _texts(self.document)
        by_position: dict[int, list[dict[str, Any]]] = {}
        for change in self.changes:
            by_position.setdefault(change["position"], []).append(change)
        unexplained: list[dict[str, Any]] = []
        for position, (canonical, rendered) in enumerate(
            zip(self.canonical, final, strict=True)
        ):
            current = canonical
            for change in by_position.get(position, []):
                if change["before"] != current:
                    break
                current = change["after"]
            if current != rendered:
                unexplained.append({
                    "position": position,
                    "paragraph_index": self.document.paragraphs[position].index,
                    "reason": "difference_without_recorded_producer",
                })
        unproven = [
            {key: change[key] for key in (
                "producer", "position", "paragraph_index", "proof_failure",
            )}
            for change in self.changes
            if change["proof_failure"]
        ]
        return {
            "canonical_paragraph_count": len(self.canonical),
            "changed_paragraph_count": len(by_position),
            "change_count": len(self.changes),
            "changes_by_producer": dict(Counter(
                change["producer"] for change in self.changes
            )),
            "changes": [
                {key: change[key] for key in (
                    "producer", "paragraph_index", "before", "after",
                    "replay_verified", "proof_failure",
                )}
                for change in self.changes
            ],
            "unexplained": unexplained,
            "unproven": unproven,
            "passed": not unexplained and not unproven,
        }


def expected_delivered_units(
    document: TranslatedDocument,
    bilingual_mode: str,
) -> list[dict[str, Any]] | None:
    """Mirror the DOCX exporter's text units for every supported layout."""
    from tarjomeh.exporters.docx_exporter import NOTES_HEADING, contents_display_title
    from tarjomeh.exporters.term_notes import document_term_notes, paragraph_note_parts

    if bilingual_mode not in {"target_only", "inline", "side_by_side"}:
        return None

    def delivered(text: str, metadata: dict[str, Any]) -> str:
        return "".join(
            segment + (
                str(ref.get("display_number", ref["number"])) if ref is not None else ""
            )
            for segment, ref in paragraph_note_parts(text, metadata)
        )

    units: list[dict[str, Any]] = []
    paragraphs = document.paragraphs
    if bilingual_mode == "side_by_side":
        units.append({
            "kind": "table",
            "rows": [
                [paragraph.source_text, delivered(paragraph.translated_text, paragraph.metadata)]
                for paragraph in paragraphs
            ],
        })
    elif bilingual_mode == "inline":
        for paragraph in paragraphs:
            units.append({
                "kind": "paragraph",
                "paragraph_index": paragraph.index,
                "text": paragraph.source_text,
            })
            units.append({
                "kind": "paragraph",
                "paragraph_index": paragraph.index,
                "text": delivered(paragraph.translated_text, paragraph.metadata),
            })
    else:
        position = 0
        while position < len(paragraphs):
            paragraph = paragraphs[position]
            if (
                paragraph.metadata.get("suppress_empty_target_export")
                and not paragraph.translated_text.strip()
            ):
                position += 1
                continue
            if paragraph.metadata.get("structure_role") == "contents_entry":
                rows = []
                while (
                    position < len(paragraphs)
                    and paragraphs[position].metadata.get("structure_role") == "contents_entry"
                ):
                    entry = paragraphs[position]
                    rows.append([
                        delivered(
                            contents_display_title(entry.translated_text, entry.metadata),
                            entry.metadata,
                        ),
                        str(entry.metadata.get("toc_page_label", "")),
                    ])
                    position += 1
                units.append({"kind": "table", "rows": rows})
                continue
            units.append({
                "kind": "paragraph",
                "paragraph_index": paragraph.index,
                "text": delivered(paragraph.translated_text, paragraph.metadata),
            })
            position += 1
    notes = document_term_notes(document)
    if notes:
        units.append({"kind": "paragraph", "paragraph_index": None, "text": NOTES_HEADING})
        for note in notes:
            units.append({
                "kind": "paragraph",
                "paragraph_index": None,
                "text": (
                    f"{note.get('display_number', note['number'])}. {note['original']} "
                    f"({note['transliteration']})"
                ),
            })
    return units


def read_docx_units(path: Path) -> list[dict[str, Any]]:
    """Read body paragraphs and tables of a DOCX in document order."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    units: list[dict[str, Any]] = []
    for element in document.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            units.append({"kind": "paragraph", "text": Paragraph(element, document).text})
        elif tag == "tbl":
            table = Table(element, document)
            units.append({
                "kind": "table",
                "rows": [[cell.text for cell in row.cells] for row in table.rows],
            })
    return units


def docx_contents_evidence(path: Path, source_contents_rows: int) -> dict[str, Any]:
    """Inspect native table units; mixed-role chunks do not hide contents rows."""
    import xml.etree.ElementTree as ET
    import zipfile

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    prefix = "{" + namespace["w"] + "}"
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    units = read_docx_units(path)
    tables = [unit for unit in units if unit["kind"] == "table"]
    evidence = []
    for table, element in zip(tables, root.findall("w:body/w:tbl", namespace), strict=True):
        rows = table["rows"]
        properties = element.find("w:tblPr", namespace)
        if properties is None:
            continue
        bidi = properties.find("w:bidiVisual", namespace)
        alignment = properties.find("w:jc", namespace)
        layout = properties.find("w:tblLayout", namespace)
        width = properties.find("w:tblW", namespace)
        widths = [int(item.get(prefix + "w", "0")) for item in
                  element.findall("w:tblGrid/w:gridCol", namespace)]
        twips = int(width.get(prefix + "w", "0")) if width is not None else 0
        page_column = bool(rows) and all(
            len(row) == 2 and bool(re.fullmatch(
                r"[0-9\u06f0-\u06f9ivxlcdmIVXLCDM\s.\u2013-]+", row[1].strip()
            )) for row in rows
        )
        if not page_column:
            continue
        evidence.append({
            "rows": len(rows), "columns": 2,
            "rtl": bidi is not None and bidi.get(prefix + "val", "1") not in {"0", "false"},
            "right_aligned": alignment is not None and alignment.get(prefix + "val") == "right",
            "fixed_layout": layout is not None and layout.get(prefix + "type") == "fixed",
            "table_width_twips": twips, "grid_widths_twips": widths,
            "full_width": bool(twips >= 9000 and len(widths) == 2
                               and sum(widths) == twips and widths[0] > widths[1] * 4),
        })
    matched = bool(source_contents_rows and evidence
                   and sum(item["rows"] for item in evidence) == source_contents_rows
                   and all(item["rtl"] for item in evidence))
    return {
        "source_contents_rows": source_contents_rows, "native_page_tables": evidence,
        "docx_unit_count": len(units), "matched_native_rtl_rows": matched,
        "status": "PASS" if matched else "REVIEW" if source_contents_rows else "NOT_APPLICABLE",
        "scope": "table structure only; lexical identity uses the separate render identity gate",
    }


def compare_docx_to_rendered(
    document: TranslatedDocument,
    path: Path,
    bilingual_mode: str,
) -> dict[str, Any]:
    """Compare the written DOCX with the rendered document unit by unit."""
    expected = expected_delivered_units(document, bilingual_mode)
    if expected is None:
        return {
            "mapping": "unsupported_bilingual_mode",
            "bilingual_mode": bilingual_mode,
            "passed": False,
            "mismatches": [],
        }
    actual = read_docx_units(path)
    mismatches: list[dict[str, Any]] = []
    for index in range(max(len(expected), len(actual))):
        want = expected[index] if index < len(expected) else None
        got = actual[index] if index < len(actual) else None
        if want == got or (
            want is not None and got is not None
            and want["kind"] == got["kind"]
            and (
                want.get("text") == got.get("text")
                if want["kind"] == "paragraph"
                else want.get("rows") == got.get("rows")
            )
        ):
            continue
        mismatches.append({
            "unit": index,
            "expected": want,
            "actual": got,
        })
        if len(mismatches) >= 20:
            break
    return {
        "mapping": f"exporter_{bilingual_mode}",
        "verified": True,
        "expected_unit_count": len(expected),
        "actual_unit_count": len(actual),
        "mismatches": mismatches,
        "passed": not mismatches,
    }
