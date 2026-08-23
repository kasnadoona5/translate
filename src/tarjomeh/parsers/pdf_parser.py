"""PDF document parser using PyMuPDF (fitz) with optional DocLayout-YOLO backend.

Provides two concrete parsers:

* **PyMuPDFParser** — default; uses ``pymupdf`` for text extraction, TOC-based
  chapter detection, font-size heuristics for headings, and footnote detection.
* **DocLayoutParser** — optional; uses the *DocLayout-YOLO* model for
  layout-aware region classification.  Falls back to *PyMuPDFParser* when the
  extra dependency is not installed.
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from pathlib import Path
from typing import Any

import pymupdf as fitz

from tarjomeh.parsers.base import (
    BaseParser,
    Chapter,
    Document,
    Paragraph,
    Section,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / tunables
# ---------------------------------------------------------------------------

_MIN_TEXT_LENGTH_PER_PAGE = 40  # characters — below this the page is "scanned"
_SCANNED_PAGE_RATIO_THRESHOLD = 0.5  # >50 % scanned pages → warn
_HEADING_FONT_SIZE_FACTOR = 1.20  # 20 % larger than body → heading candidate
_FOOTNOTE_FONT_SIZE_FACTOR = 0.85  # ≤85 % of body → footnote candidate
_FOOTNOTE_Y_RATIO = 0.80  # bottom 20 % of the page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _median_font_size(blocks: list[dict[str, Any]]) -> float:
    """Return the median font size across *blocks* extracted from a page.

    Each block is a dict produced by :func:`_extract_page_blocks`.
    """
    sizes: list[float] = [
        b["font_size"] for b in blocks if b.get("font_size", 0) > 0
    ]
    if not sizes:
        return 12.0  # safe default
    return statistics.median(sizes)


_SOFT_HYPHEN = "­"
# Edge zone (fraction of page height) where short blocks are treated as
# running headers / page numbers and dropped.
_EDGE_ZONE = 0.12
_EDGE_MAX_CHARS = 100
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:\d+|[ivxlcdm]+)\s*$", re.IGNORECASE)
_FURNITURE_NUMBER_RE = re.compile(r"\b(?:\d+|[ivxlcdm]+)\b", re.IGNORECASE)


def _join_block_lines(
    lines: list[str],
    *,
    known_words: set[str] | None = None,
    dehyphenation_evidence: list[dict[str, str]] | None = None,
) -> str:
    """Join the visual lines of one PDF text block into flowing prose.

    Handles print-style hyphenation:
    * a line ending in a SOFT HYPHEN (U+00AD) is a pure typographic break —
      join with the next line directly and drop the marker ("win­/dow" → "window");
    * a line ending in an ASCII "-" keeps the hyphen but joins without a
      space ("absent-/minded" → "absent-minded");
    * otherwise lines are joined with a single space.
    Remaining stray soft hyphens are stripped (invisible junk that corrupts
    words for the translator).
    """
    out = ""
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if not out:
            out = line
        elif out.endswith(_SOFT_HYPHEN):
            out = out[: -len(_SOFT_HYPHEN)] + line.lstrip()
        elif out.endswith("-"):
            left = re.search(r"([A-Za-z]{2,})-$", out)
            right = re.match(r"([a-z]{2,})", line.lstrip())
            joined = (
                left.group(1) + right.group(1)
                if left is not None and right is not None else ""
            )
            if joined and known_words and joined.casefold() in known_words:
                before = f"{left.group(1)}-{right.group(1)}"
                out = out[:-1] + line.lstrip()
                if dehyphenation_evidence is not None:
                    dehyphenation_evidence.append({
                        "before": before,
                        "joined": joined,
                        "reason": "unhyphenated_form_observed_elsewhere_in_source",
                    })
            else:
                out = out + line.lstrip()
        else:
            out = out + " " + line.lstrip()
    return out.replace(_SOFT_HYPHEN, "").strip()


def _join_span_texts(spans: list[dict[str, Any]]) -> str:
    """Join styled PDF spans while restoring only geometry-evidenced spaces."""
    output = ""
    previous: dict[str, Any] | None = None
    for span in spans:
        value = str(span.get("text", ""))
        if not value:
            continue
        if output and previous is not None and not output[-1].isspace() and not value[0].isspace():
            previous_box = previous.get("bbox", (0, 0, 0, 0))
            current_box = span.get("bbox", (0, 0, 0, 0))
            gap = float(current_box[0]) - float(previous_box[2])
            font_size = min(
                float(previous.get("size", 0.0) or 0.0),
                float(span.get("size", 0.0) or 0.0),
            )
            threshold = max(0.8, font_size * 0.12)
            if (
                gap >= threshold
                and re.search(r"[A-Za-z0-9)\]}]$", output)
                and re.match(r"[A-Za-z0-9]", value)
            ):
                output += " "
        output += value
        previous = span
    return output.strip()


_SUPERSCRIPT_MARKER_RE = re.compile(
    r"(?:[0-9\u0660-\u0669\u06f0-\u06f9]{1,3}|"
    r"[\u00b9\u00b2\u00b3\u2070-\u2079*\u2020\u2021])"
)


def _line_superscript_markers(
    spans: list[dict[str, Any]],
    line_text: str,
) -> list[dict[str, Any]]:
    """Retain source-confirmed note-marker positions without inferring them."""
    markers: list[dict[str, Any]] = []
    cursor = 0
    superscript_flag = int(getattr(fitz, "TEXT_FONT_SUPERSCRIPT", 1))
    for span in spans:
        value = str(span.get("text", "")).strip()
        if (
            not value
            or not int(span.get("flags", 0)) & superscript_flag
            or not _SUPERSCRIPT_MARKER_RE.fullmatch(value)
        ):
            continue
        offset = line_text.find(value, cursor)
        if offset < 0:
            offset = line_text.find(value)
        if offset < 0:
            continue
        cursor = offset + len(value)
        markers.append({
            "text": value,
            "relative_position": round(
                (offset + len(value) / 2) / max(1, len(line_text)), 6
            ),
        })
    return markers


def _extract_page_blocks(
    page: fitz.Page,
    *,
    geometry_order: bool = True,
) -> list[dict[str, Any]]:
    """Extract text blocks with font metadata from a single PDF page.

    One entry per PDF text BLOCK (visual paragraph) — the block's lines are
    retained with line geometry so document-level recurrence analysis can
    distinguish running furniture from legitimate headings.

    Returns a list of dicts, each with keys:
    ``text``, ``font_size``, ``bbox``, ``font_name``.
    """
    blocks: list[dict[str, Any]] = []
    # Keep ASCII hyphens verbatim: PyMuPDF's blanket dehyphenation also turns
    # legitimate compounds such as "well-defined" into "welldefined".
    # _join_block_lines still removes unambiguous soft-hyphen line breaks.
    raw_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    page_height = page.rect.height or 1.0
    page_width = page.rect.width or 1.0

    for raw_index, block in enumerate(raw_dict.get("blocks", [])):
        if block.get("type") != 0:  # 0 = text block
            continue

        line_texts: list[str] = []
        line_records: list[dict[str, Any]] = []
        sizes: list[float] = []
        font_names: list[str] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text_parts: list[str] = []
            for span in spans:
                t = span.get("text", "")
                if t.strip():
                    text_parts.append(t)
                    sizes.append(span.get("size", 0.0))
                    font_names.append(span.get("font", ""))
            if text_parts:
                line_text = _join_span_texts([
                    span for span in spans if str(span.get("text", "")).strip()
                ])
                line_texts.append(line_text)
                line_sizes = [
                    float(span.get("size", 0.0))
                    for span in spans if str(span.get("text", "")).strip()
                ]
                line_records.append({
                    "text": line_text,
                    "bbox": tuple(line.get("bbox", block["bbox"])),
                    "direction": tuple(line.get("dir", (1.0, 0.0))),
                    "font_size": (
                        sum(line_sizes) / len(line_sizes) if line_sizes else 0.0
                    ),
                    "has_superscript": any(
                        int(span.get("flags", 0))
                        & int(getattr(fitz, "TEXT_FONT_SUPERSCRIPT", 1))
                        for span in spans
                        if str(span.get("text", "")).strip()
                    ),
                    "superscript_markers": _line_superscript_markers(
                        spans, line_text
                    ),
                    "span_count": len([
                        span for span in spans
                        if str(span.get("text", "")).strip()
                    ]),
                    "large_gap_count": sum(
                        1
                        for left, right in zip(spans, spans[1:], strict=False)
                        if (
                            str(left.get("text", "")).strip()
                            and str(right.get("text", "")).strip()
                            and float(right.get("bbox", (0, 0, 0, 0))[0])
                            - float(left.get("bbox", (0, 0, 0, 0))[2]) >= 18
                        )
                    ),
                })

        merged_text = _join_block_lines(line_texts)
        if not merged_text:
            continue

        x0, y0, x1, y1 = block["bbox"]
        avg_size = sum(sizes) / len(sizes) if sizes else 0.0
        raw_line_chars = sum(len(value) for value in line_texts) + max(
            0, len(line_texts) - 1
        )
        superscript_markers: list[dict[str, Any]] = []
        consumed = 0
        for line_record in line_records:
            line_text = str(line_record.get("text", ""))
            for marker in line_record.get("superscript_markers", []) or []:
                local = float(marker.get("relative_position", 0.0))
                absolute = consumed + local * len(line_text)
                superscript_markers.append({
                    "text": str(marker.get("text", "")),
                    "relative_position": round(
                        absolute / max(1, raw_line_chars), 6
                    ),
                })
            consumed += len(line_text) + 1
        blocks.append(
            {
                "text": merged_text,
                "font_size": avg_size,
                "font_name": font_names[0] if font_names else "",
                "bbox": block["bbox"],  # (x0, y0, x1, y1)
                "lines": line_records,
                "page_height": page_height,
                "page_width": page_width,
                "raw_index": raw_index,
                "orientation": (
                    "horizontal"
                    if all(
                        abs(float(line.get("direction", (1.0, 0.0))[1])) < 0.25
                        for line in line_records
                    )
                    else "rotated"
                ),
                "has_superscript": any(
                    bool(line.get("has_superscript")) for line in line_records
                ),
                "superscript_markers": superscript_markers,
            }
        )
    return (
        _sort_page_blocks_reading_order(blocks, page_width)
        if geometry_order else blocks
    )


def _sort_page_blocks_reading_order(
    blocks: list[dict[str, Any]],
    page_width: float,
) -> list[dict[str, Any]]:
    """Return geometry-based reading order without scrambling rotated tables.

    PDF content-stream order is not visual reading order. Most book pages are
    single-column, while indexes and reference material may use two columns.
    Full-width blocks split the page into bands; narrow blocks inside a band
    are read down the left column and then down the right column. Pages whose
    text is predominantly rotated retain source order because their geometry
    usually represents a landscape table embedded in a portrait page.
    """
    if len(blocks) < 2:
        return blocks

    rotated = sum(block.get("orientation") == "rotated" for block in blocks)
    if rotated / len(blocks) >= 0.5:
        for order, block in enumerate(blocks):
            block["reading_order"] = order
            block["reading_order_mode"] = "source_order_rotated"
        return blocks

    width = max(float(page_width), 1.0)
    midpoint = width / 2.0
    gutter = width * 0.035

    def bbox(block: dict[str, Any]) -> tuple[float, float, float, float]:
        return tuple(float(value) for value in block["bbox"])  # type: ignore[return-value]

    def geometric(blocks_to_sort: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(blocks_to_sort, key=lambda item: (bbox(item)[1], bbox(item)[0]))

    left = [block for block in blocks if bbox(block)[2] <= midpoint - gutter]
    right = [block for block in blocks if bbox(block)[0] >= midpoint + gutter]
    has_two_columns = len(left) >= 2 and len(right) >= 2

    if not has_two_columns:
        ordered = geometric(blocks)
        for order, block in enumerate(ordered):
            block["reading_order"] = order
            block["reading_order_mode"] = "geometry_single_column"
        return ordered

    separators = [
        block for block in blocks
        if (
            bbox(block)[0] < midpoint - gutter
            and bbox(block)[2] > midpoint + gutter
        )
    ]
    separators = geometric(separators)
    remaining = [block for block in blocks if block not in separators]

    def column_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        band_left = geometric([
            item for item in items if bbox(item)[2] <= midpoint + gutter
        ])
        band_right = geometric([
            item for item in items if bbox(item)[0] > midpoint - gutter
        ])
        assigned = {id(item) for item in band_left + band_right}
        other = geometric([item for item in items if id(item) not in assigned])
        return band_left + band_right + other

    ordered: list[dict[str, Any]] = []
    lower_bound = float("-inf")
    for separator in separators:
        separator_y = bbox(separator)[1]
        band = [
            item for item in remaining
            if lower_bound <= bbox(item)[1] < separator_y
        ]
        ordered.extend(column_order(band))
        ordered.append(separator)
        lower_bound = bbox(separator)[3]
    ordered.extend(column_order([
        item for item in remaining if bbox(item)[1] >= lower_bound
    ]))

    # Defensive completion for unusual overlapping geometry.
    present = {id(block) for block in ordered}
    ordered.extend(geometric([block for block in blocks if id(block) not in present]))
    for order, block in enumerate(ordered):
        block["reading_order"] = order
        block["reading_order_mode"] = "geometry_two_column"
    return ordered


def _normalise_furniture(text: str) -> str:
    """Return a recurrence signature for a possible running header/footer."""
    folded = " ".join(text.casefold().split())
    folded = _FURNITURE_NUMBER_RE.sub("#", folded)
    return folded.strip(" -–—|·")


def _edge_line(line: dict[str, Any], page_height: float) -> bool:
    text = str(line.get("text", "")).strip()
    if not text or len(text) > _EDGE_MAX_CHARS:
        return False
    _x0, y0, _x1, y1 = line.get("bbox", (0, 0, 0, 0))
    return y1 <= _EDGE_ZONE * page_height or y0 >= (1 - _EDGE_ZONE) * page_height


def _strip_embedded_furniture(
    text: str,
    recurring_labels: set[str],
) -> tuple[str, bool]:
    """Strip a recurrent header plus page number fused to body text."""
    clean = " ".join(text.split())
    for label in sorted(recurring_labels, key=len, reverse=True):
        if not label or label == "#":
            continue
        escaped = re.escape(label).replace(r"\#", r"(?:\d+|[ivxlcdm]+)")
        patterns = (
            (rf"^(?:{escaped})\s+(?=[a-z])",)
            if "#" in label else (
                rf"^(?:{escaped})\s+(?:\d+|[ivxlcdm]+)\s+(?=[a-z])",
                rf"^(?:\d+|[ivxlcdm]+)\s+(?:{escaped})\s+(?=[a-z])",
            )
        )
        for pattern in patterns:
            updated = re.sub(pattern, "", clean, count=1, flags=re.IGNORECASE)
            if updated != clean:
                return updated.strip(), True
    return clean, False


def _looks_table_like(block: dict[str, Any]) -> bool:
    """Cheap table signal based on repeated, widely separated text spans."""
    lines = list(block.get("lines", []) or [])
    if len(lines) < 2:
        return False
    tabular_lines = sum(
        1 for line in lines
        if int(line.get("span_count", 0)) >= 2
        and int(line.get("large_gap_count", 0)) >= 1
    )
    return tabular_lines >= 2 and tabular_lines / len(lines) >= 0.5


_INDENTED_PARAGRAPH_START_RE = re.compile(r'^[\s\u201c\u2018"\'\(\[]*[A-Z]')
_PARAGRAPH_BOUNDARY_RE = re.compile(
    r'[.!?\u2026][\u201d\u2019"\'\)\]\u00bb]*(?:\d+)?$'
)


def _split_indented_paragraph_lines(
    lines: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Recover conservative paragraph starts hidden inside one PDF block.

    Recent PyMuPDF releases can group multiple print paragraphs into one text
    block. A genuine new body paragraph normally combines three independent
    signals: a first-line indent, a complete preceding sentence, and an
    uppercase prose start. Lists, tables, rotated text, and ambiguous lines are
    intentionally left unchanged.
    """
    if len(lines) < 3:
        return [lines]
    if any(
        abs(float(line.get("direction", (1.0, 0.0))[1])) >= 0.25
        for line in lines
    ):
        return [lines]
    first_text = str(lines[0].get("text", "")).strip()
    if not _INDENTED_PARAGRAPH_START_RE.match(first_text):
        return [lines]

    x_positions = [float(line.get("bbox", (0, 0, 0, 0))[0]) for line in lines]
    buckets: dict[float, int] = {}
    for value in x_positions:
        key = round(value, 1)
        buckets[key] = buckets.get(key, 0) + 1
    highest_frequency = max(buckets.values())
    dominant_left = min(
        key for key, frequency in buckets.items()
        if frequency == highest_frequency
    )
    sizes = [
        float(line.get("font_size", 0.0))
        for line in lines if float(line.get("font_size", 0.0)) > 0
    ]
    body_size = statistics.median(sizes) if sizes else 10.0
    minimum_indent = max(6.0, body_size * 0.65)

    groups: list[list[dict[str, Any]]] = [[]]
    for index, line in enumerate(lines):
        text = str(line.get("text", "")).strip()
        previous_text = _join_block_lines([
            str(item.get("text", "")) for item in groups[-1]
        ])
        line_size = float(line.get("font_size", body_size) or body_size)
        starts_paragraph = bool(
            index > 0
            and groups[-1]
            and x_positions[index] - dominant_left >= minimum_indent
            and 0.8 * body_size <= line_size <= 1.2 * body_size
            and _INDENTED_PARAGRAPH_START_RE.match(text)
            and _PARAGRAPH_BOUNDARY_RE.search(previous_text)
            and not previous_text.endswith(("-", _SOFT_HYPHEN))
        )
        if starts_paragraph:
            groups.append([])
        groups[-1].append(line)
    return groups


def _line_group_bbox(
    lines: list[dict[str, Any]],
    fallback: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if not lines:
        return fallback
    boxes = [tuple(float(value) for value in line.get("bbox", fallback)) for line in lines]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _prepare_document_blocks(
    doc: fitz.Document,
    *,
    structure_version: int = 3,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Extract all pages, remove only recurrent edge furniture, and classify blocks."""
    pages: list[list[dict[str, Any]]] = []
    edge_occurrences: dict[str, set[int]] = {}
    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = _extract_page_blocks(
            page, geometry_order=structure_version >= 2
        )
        for block in blocks:
            for line in block.get("lines", []):
                if _edge_line(line, float(block.get("page_height", 1.0))):
                    signature = _normalise_furniture(str(line.get("text", "")))
                    if signature:
                        edge_occurrences.setdefault(signature, set()).add(page_num)
        pages.append(blocks)

    recurrence_min = max(
        3, min(8, math.ceil(max(doc.page_count, 1) * 0.02))
    )
    recurrent = {
        signature
        for signature, page_numbers in edge_occurrences.items()
        if len(page_numbers) >= recurrence_min
    }
    source_words = {
        match.group().casefold()
        for blocks in pages
        for block in blocks
        for line in block.get("lines", [])
        for match in re.finditer(
            r"(?<![A-Za-z-])[A-Za-z]{4,}(?![A-Za-z-])",
            str(line.get("text", "")),
        )
    }
    removed: list[dict[str, Any]] = []
    table_blocks = 0
    rotated_table_pages: list[int] = []
    reading_order_modes: dict[str, int] = {}
    internal_paragraph_splits: list[dict[str, Any]] = []
    ascii_dehyphenation_evidence: list[dict[str, str]] = []
    prepared: list[list[dict[str, Any]]] = []
    for page_num, blocks in enumerate(pages):
        rotated_count = sum(
            block.get("orientation") == "rotated" for block in blocks
        )
        rotated_table_page = bool(
            structure_version >= 2
            and
            len(blocks) >= 6 and rotated_count / max(len(blocks), 1) >= 0.5
        )
        if rotated_table_page:
            rotated_table_pages.append(page_num + 1)
        page_blocks: list[dict[str, Any]] = []
        for block in blocks:
            kept_lines: list[dict[str, Any]] = []
            block_removed = []
            for line in block.get("lines", []):
                text = str(line.get("text", "")).strip()
                signature = _normalise_furniture(text)
                is_edge = _edge_line(line, float(block.get("page_height", 1.0)))
                if is_edge and (
                    signature in recurrent or _PAGE_NUMBER_RE.fullmatch(text)
                ):
                    block_removed.append(text)
                else:
                    kept_lines.append(line)
            table_like = bool(
                rotated_table_page
                or _looks_table_like({**block, "lines": kept_lines})
            )
            line_groups = (
                _split_indented_paragraph_lines(kept_lines)
                if structure_version >= 3 and not table_like
                else [kept_lines]
            )
            embedded_removed_any = False
            if len(line_groups) > 1:
                internal_paragraph_splits.append({
                    "page": page_num + 1,
                    "raw_block": int(block.get("raw_index", 0)),
                    "paragraph_count": len(line_groups),
                    "starts": [
                        str(group[0].get("text", ""))[:120]
                        for group in line_groups if group
                    ],
                })
            for part_index, line_group in enumerate(line_groups):
                text = _join_block_lines(
                    [str(line.get("text", "")) for line in line_group],
                    known_words=source_words,
                    dehyphenation_evidence=ascii_dehyphenation_evidence,
                )
                text, embedded_removed = _strip_embedded_furniture(text, recurrent)
                embedded_removed_any = embedded_removed_any or embedded_removed
                if not text:
                    continue
                updated = dict(block)
                updated["text"] = text
                updated["lines"] = line_group
                updated["bbox"] = _line_group_bbox(
                    line_group, tuple(float(value) for value in block["bbox"])
                )
                group_sizes = [
                    float(line.get("font_size", 0.0))
                    for line in line_group
                    if float(line.get("font_size", 0.0)) > 0
                ]
                if group_sizes:
                    updated["font_size"] = sum(group_sizes) / len(group_sizes)
                updated["_page"] = page_num
                fragment_id = (
                    f"pg{page_num + 1:04d}.b{int(block.get('raw_index', 0)):04d}"
                )
                if len(line_groups) > 1:
                    fragment_id += f".p{part_index + 1:02d}"
                updated["source_fragment_id"] = fragment_id
                updated["is_table"] = table_like
                mode = str(updated.get("reading_order_mode", "source_order"))
                reading_order_modes[mode] = reading_order_modes.get(mode, 0) + 1
                if updated["is_table"]:
                    table_blocks += 1
                page_blocks.append(updated)
            if block_removed or embedded_removed_any:
                removed.append({
                    "page": page_num + 1,
                    "text": " | ".join(block_removed),
                    "embedded": embedded_removed_any,
                })
        prepared.append(page_blocks)
    return prepared, {
        "furniture_signatures": sorted(recurrent),
        "removed_furniture": removed,
        "removed_furniture_count": len(removed),
        "table_block_count": table_blocks,
        "rotated_table_pages": rotated_table_pages,
        "reading_order_modes": reading_order_modes,
        "internal_paragraph_split_count": len(internal_paragraph_splits),
        "internal_paragraph_splits": internal_paragraph_splits,
        "recurrence_minimum_pages": recurrence_min,
        "dehyphenation_mode": "soft_hyphen_plus_repeated_source_evidence",
        "ascii_dehyphenation_count": len(ascii_dehyphenation_evidence),
        "ascii_dehyphenation_evidence": ascii_dehyphenation_evidence[:200],
        "structure_version": structure_version,
    }


_SENTENCE_END_CHARS = '.?!:;"”»…'


def _merge_continuation_paragraphs(paragraphs: list[Paragraph]) -> list[Paragraph]:
    """Stitch body paragraphs that continue across blocks/columns/pages.

    Print paragraphs frequently break at page boundaries. When a body
    paragraph does not end with sentence-final punctuation and the next body
    paragraph starts with a lowercase letter (or the previous ends with a
    hyphen), they are two halves of one logical paragraph — merge them.
    Headings and footnotes are never merged.
    """
    merged: list[Paragraph] = []
    for para in paragraphs:
        prev = merged[-1] if merged else None
        is_body = not para.metadata.get("heading_level") and not para.metadata.get("is_footnote")
        prev_is_body = (
            prev is not None
            and not prev.metadata.get("heading_level")
            and not prev.metadata.get("is_footnote")
        )
        same_role = bool(para.metadata.get("is_table")) == bool(
            prev.metadata.get("is_table") if prev else False
        )
        prev_page = int(prev.metadata.get("page", 0)) if prev else 0
        page = int(para.metadata.get("page", 0))
        prev_bbox = prev.metadata.get("bbox", (0, 0, 0, 0)) if prev else (0, 0, 0, 0)
        bbox = para.metadata.get("bbox", (0, 0, 0, 0))
        prev_height = float(prev.metadata.get("page_height", 1.0)) if prev else 1.0
        page_height = float(para.metadata.get("page_height", 1.0))
        crosses_page = page == prev_page + 1
        near_page_boundary = (
            float(prev_bbox[3]) >= 0.65 * prev_height
            and float(bbox[1]) <= 0.35 * page_height
        )
        legacy_without_geometry = bool(
            prev is not None
            and "page" not in prev.metadata
            and "page" not in para.metadata
        )
        if (
            prev is not None
            and is_body
            and prev_is_body
            and same_role
            and not para.metadata.get("is_table")
            and (
                legacy_without_geometry
                or (crosses_page and near_page_boundary)
            )
            and prev.text
            and para.text
            and prev.text[-1] not in _SENTENCE_END_CHARS
            and (para.text[0].islower() or prev.text.endswith("-"))
        ):
            joiner = "" if prev.text.endswith("-") else " "
            prev.text = prev.text + joiner + para.text
            prev.metadata["end_page"] = page
            prev.metadata["cross_page_join"] = True
            fragments = list(prev.metadata.get("source_fragment_ids", []) or [])
            fragments.extend(para.metadata.get("source_fragment_ids", []) or [])
            prev.metadata["source_fragment_ids"] = list(dict.fromkeys(fragments))
        else:
            merged.append(para)
    return merged


def _merge_table_interrupted_continuations(
    paragraphs: list[Paragraph],
) -> list[Paragraph]:
    """Rejoin one prose paragraph split by intervening table blocks.

    A print table may occupy the next page while the surrounding sentence
    resumes on the following page. Keeping the two prose fragments separate
    causes each to be translated as an incomplete sentence. When geometry,
    syntax, and structural roles all agree, move the preserved table block
    after the completed prose paragraph and retain provenance in metadata.
    """
    result: list[Paragraph] = []
    index = 0
    while index < len(paragraphs):
        left = paragraphs[index]
        table_end = index + 1
        while (
            table_end < len(paragraphs)
            and paragraphs[table_end].metadata.get("is_table")
        ):
            table_end += 1

        if table_end == index + 1 or table_end >= len(paragraphs):
            result.append(left)
            index += 1
            continue

        right = paragraphs[table_end]
        left_body = bool(
            not left.metadata.get("is_table")
            and not left.metadata.get("heading_level")
            and not left.metadata.get("is_footnote")
        )
        right_body = bool(
            not right.metadata.get("is_table")
            and not right.metadata.get("heading_level")
            and not right.metadata.get("is_footnote")
        )
        right_start = right.text.lstrip(" \t\"'\u2018\u201c(")[:1]
        left_page = int(left.metadata.get("page", 0) or 0)
        right_page = int(right.metadata.get("page", 0) or 0)
        pages_advance = bool(left_page and right_page and right_page > left_page)
        continuation = bool(
            left_body
            and right_body
            and pages_advance
            and left.text
            and right.text
            and left.text[-1] not in _SENTENCE_END_CHARS
            and (right_start.islower() or left.text.endswith("-"))
        )
        if not continuation:
            result.append(left)
            index += 1
            continue

        tables = paragraphs[index + 1:table_end]
        joiner = "" if left.text.endswith("-") else " "
        left.text = left.text + joiner + right.text
        left.metadata["end_page"] = right_page
        left.metadata["cross_table_join"] = True
        left.metadata["intervening_table_blocks"] = len(tables)
        fragments = list(left.metadata.get("source_fragment_ids", []) or [])
        fragments.extend(right.metadata.get("source_fragment_ids", []) or [])
        left.metadata["source_fragment_ids"] = list(dict.fromkeys(fragments))
        result.append(left)
        for table in tables:
            table.metadata["relocated_after_continuation"] = True
            result.append(table)
        index = table_end + 1
    return result


_HEADING_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_ROMAN_TOKEN_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)


def _heading_tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _HEADING_TOKEN_RE.findall(text or ""))


def _is_numbering_token(token: str) -> bool:
    return bool(token.isdigit() or _ROMAN_TOKEN_RE.fullmatch(token))


def _heading_variants(text: str) -> set[tuple[str, ...]]:
    tokens = _heading_tokens(text)
    variants = {tokens} if tokens else set()
    if len(tokens) > 1 and _is_numbering_token(tokens[0]):
        variants.add(tokens[1:])
    if len(tokens) > 1 and _is_numbering_token(tokens[-1]):
        variants.add(tokens[:-1])
    return {variant for variant in variants if variant}


def _matches_chapter_heading(candidate: str, chapter_title: str) -> bool:
    return bool(_heading_variants(candidate) & _heading_variants(chapter_title))


def _canonicalize_chapter_heading(
    chapter_title: str,
    paragraphs: list[Paragraph],
) -> tuple[list[Paragraph], dict[str, Any]]:
    """Create one stable chapter heading and remove matching running headers.

    The TOC supplies the canonical title. Printed books often split a chapter
    number and title into separate PDF blocks and repeat the title beside page
    numbers on following pages. This routine consolidates only geometry-near
    title blocks that match the TOC, leaving ordinary prose untouched.
    """
    if not paragraphs or not chapter_title.strip() or chapter_title == "Untitled":
        return paragraphs, {"canonicalized": False, "duplicates_removed": 0}

    first_page = min(int(p.metadata.get("page", 1)) for p in paragraphs)
    first_page_indices = [
        index for index, paragraph in enumerate(paragraphs)
        if int(paragraph.metadata.get("page", first_page)) == first_page
        and float(paragraph.metadata.get("bbox", (0, 0, 0, 0))[1])
        <= 0.55 * float(paragraph.metadata.get("page_height", 1.0))
    ][:8]

    matched_indices: list[int] = []
    for start_offset in range(len(first_page_indices)):
        for width in range(1, min(4, len(first_page_indices) - start_offset) + 1):
            indices = first_page_indices[start_offset:start_offset + width]
            if indices != list(range(indices[0], indices[0] + len(indices))):
                continue
            combined = " ".join(paragraphs[index].text for index in indices)
            if _matches_chapter_heading(combined, chapter_title):
                matched_indices = indices
                break
        if matched_indices:
            break

    if matched_indices:
        source_parts = [paragraphs[index] for index in matched_indices]
        metadata = dict(source_parts[0].metadata)
        bboxes = [part.metadata.get("bbox", (0, 0, 0, 0)) for part in source_parts]
        metadata["bbox"] = (
            min(float(box[0]) for box in bboxes),
            min(float(box[1]) for box in bboxes),
            max(float(box[2]) for box in bboxes),
            max(float(box[3]) for box in bboxes),
        )
        fragments: list[str] = []
        for part in source_parts:
            fragments.extend(part.metadata.get("source_fragment_ids", []) or [])
        insert_at = matched_indices[0]
        retained = [
            paragraph for index, paragraph in enumerate(paragraphs)
            if index not in set(matched_indices)
        ]
    else:
        metadata = dict(paragraphs[0].metadata)
        fragments = []
        insert_at = 0
        retained = list(paragraphs)

    metadata.update({
        "heading_level": 1,
        "structure_role": "heading",
        "chapter_heading": True,
        "canonical_chapter_title": True,
        "source_fragment_ids": list(dict.fromkeys(fragments)),
    })
    canonical = Paragraph(text=chapter_title.strip(), metadata=metadata)
    retained.insert(insert_at, canonical)

    deduplicated: list[Paragraph] = []
    removed = 0
    canonical_seen = False
    for paragraph in retained:
        if paragraph is canonical:
            canonical_seen = True
            deduplicated.append(paragraph)
            continue
        page_height = float(paragraph.metadata.get("page_height", 1.0))
        y0 = float(paragraph.metadata.get("bbox", (0, 0, 0, 0))[1])
        at_running_edge = y0 <= 0.14 * page_height
        if (
            canonical_seen
            and at_running_edge
            and _matches_chapter_heading(paragraph.text, chapter_title)
        ):
            removed += 1
            continue
        deduplicated.append(paragraph)

    return deduplicated, {
        "canonicalized": True,
        "source_blocks_consolidated": len(matched_indices),
        "duplicates_removed": removed,
        "inserted_from_toc": not bool(matched_indices),
    }


def _is_heading(
    block: dict[str, Any],
    median_size: float,
) -> bool:
    """Heuristic: the block is a heading if its font is significantly larger."""
    return block["font_size"] >= median_size * _HEADING_FONT_SIZE_FACTOR


def _is_footnote(
    block: dict[str, Any],
    median_size: float,
    page_height: float,
) -> bool:
    """Heuristic: small font near the bottom of the page → footnote."""
    if block["font_size"] > median_size * _FOOTNOTE_FONT_SIZE_FACTOR:
        return False
    _, y0, _, _ = block["bbox"]
    return y0 / page_height >= _FOOTNOTE_Y_RATIO if page_height > 0 else False


def _heading_level_from_size(font_size: float, median_size: float) -> int:
    """Map a font size to a heading level (1-3) based on its ratio to body text."""
    ratio = font_size / median_size if median_size > 0 else 1.0
    if ratio >= 1.8:
        return 1
    if ratio >= 1.4:
        return 2
    return 3


# ---------------------------------------------------------------------------
# PyMuPDFParser
# ---------------------------------------------------------------------------


class PyMuPDFParser(BaseParser):
    """Parse PDF documents using *PyMuPDF* (``fitz``).

    Features:
    * TOC-based chapter boundary detection via ``doc.get_toc()``.
    * Font-size heuristics for heading and footnote detection.
    * Warning when a large fraction of pages appear scanned (image-only).
    """

    def __init__(self, structure_version: int = 3) -> None:
        self.structure_version = max(1, int(structure_version))

    def parse(self, file_path: Path) -> Document:
        """Parse *file_path* and return a :class:`Document`."""
        file_path = self._ensure_file(file_path)

        doc = fitz.open(str(file_path))
        try:
            toc = doc.get_toc()  # list of [level, title, page]
            title = self._extract_title(doc, toc)
            metadata = self._extract_metadata(doc)
            metadata["format_type"] = "pdf"

            # Warn about scanned pages
            self._check_scanned(doc)

            page_blocks, structure_audit = _prepare_document_blocks(
                doc, structure_version=self.structure_version
            )
            metadata["pdf_structure_audit"] = structure_audit

            if toc:
                chapters = self._parse_with_toc(doc, toc, page_blocks)
            else:
                chapters = self._parse_without_toc(doc, page_blocks)

            self._annotate_chapters(chapters)
            heading_audits = [
                chapter.metadata.get("heading_audit", {}) for chapter in chapters
            ]
            structure_audit["canonical_chapter_headings"] = sum(
                bool(item.get("canonicalized")) for item in heading_audits
            )
            structure_audit["chapter_heading_duplicates_removed"] = sum(
                int(item.get("duplicates_removed", 0)) for item in heading_audits
            )
            structure_audit["chapter_headings_inserted_from_toc"] = sum(
                bool(item.get("inserted_from_toc")) for item in heading_audits
            )
            structure_audit["cross_table_continuation_joins"] = sum(
                bool(paragraph.metadata.get("cross_table_join"))
                for chapter in chapters
                for paragraph in chapter.all_paragraphs
            )

            raw_toc = [entry[1] for entry in toc] if toc else None

            return Document(
                title=title,
                chapters=chapters,
                metadata=metadata,
                raw_toc=raw_toc,
            )
        finally:
            doc.close()

    # ------------------------------------------------------------------
    # Internal: metadata & title
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_title(doc: fitz.Document, toc: list[list[Any]]) -> str:
        """Best-effort title extraction: metadata → first TOC entry → filename."""
        md = doc.metadata or {}
        if md.get("title"):
            return md["title"]
        if toc:
            return toc[0][1]
        return Path(doc.name).stem

    @staticmethod
    def _extract_metadata(doc: fitz.Document) -> dict[str, Any]:
        md = doc.metadata or {}
        return {
            "author": md.get("author", ""),
            "language": md.get("language", ""),
            "producer": md.get("producer", ""),
            "page_count": doc.page_count,
        }

    # ------------------------------------------------------------------
    # Scanned-PDF detection
    # ------------------------------------------------------------------

    def _check_scanned(self, doc: fitz.Document) -> None:
        """Log a warning if the PDF looks scanned (images with little text)."""
        if doc.page_count == 0:
            return

        scanned_pages = 0
        sample_size = min(doc.page_count, 20)  # sample up to 20 pages

        for page_num in range(sample_size):
            page = doc[page_num]
            text = page.get_text("text").strip()
            images = page.get_images(full=False)
            if len(text) < _MIN_TEXT_LENGTH_PER_PAGE and len(images) > 0:
                scanned_pages += 1

        ratio = scanned_pages / sample_size
        if ratio >= _SCANNED_PAGE_RATIO_THRESHOLD:
            logger.warning(
                "PDF appears to be scanned (%.0f%% of sampled pages contain "
                "images but very little text). Consider using --enable-ocr to "
                "run OCR preprocessing first.",
                ratio * 100,
            )

    # ------------------------------------------------------------------
    # TOC-based chapter parsing
    # ------------------------------------------------------------------

    def _parse_with_toc(
        self,
        doc: fitz.Document,
        toc: list[list[Any]],
        page_blocks: list[list[dict[str, Any]]] | None = None,
    ) -> list[Chapter]:
        """Split the document into chapters using TOC entries."""
        chapters: list[Chapter] = []

        # Build page ranges for each top-level (level=1) TOC entry
        level1_entries: list[tuple[str, int, int]] = []  # (title, start_page, end_page)
        for idx, entry in enumerate(toc):
            level, title, page_num = entry[0], entry[1], entry[2]
            if level != 1:
                continue
            start_page = max(page_num - 1, 0)  # 0-indexed
            # End page = start of next level-1 entry (exclusive), or last page
            end_page = doc.page_count
            for future in toc[idx + 1 :]:
                if future[0] == 1:
                    end_page = max(future[2] - 1, 0)
                    break
            level1_entries.append((title, start_page, end_page))

        # If no level-1 entries, treat every TOC entry as a chapter
        if not level1_entries:
            for idx, entry in enumerate(toc):
                _level, title, page_num = entry[0], entry[1], entry[2]
                start_page = max(page_num - 1, 0)
                end_page = doc.page_count
                if idx + 1 < len(toc):
                    end_page = max(toc[idx + 1][2] - 1, 0)
                level1_entries.append((title, start_page, end_page))

        for chap_idx, (title, start_page, end_page) in enumerate(level1_entries, 1):
            paragraphs = self._extract_pages(
                doc, start_page, end_page, page_blocks
            )
            if self.structure_version >= 2:
                paragraphs, heading_audit = _canonicalize_chapter_heading(
                    title, paragraphs
                )
                paragraphs = _merge_table_interrupted_continuations(
                    _merge_continuation_paragraphs(paragraphs)
                )
            else:
                heading_audit = {"canonicalized": False, "legacy_resume": True}
            section = Section(title="", level=2, paragraphs=paragraphs)
            chapters.append(
                Chapter(
                    title=title,
                    number=chap_idx,
                    sections=[section],
                    metadata={
                        "start_page": start_page + 1,
                        "end_page": end_page,
                        "heading_audit": heading_audit,
                    },
                )
            )

        return chapters

    # ------------------------------------------------------------------
    # Fallback parsing (no TOC)
    # ------------------------------------------------------------------

    def _parse_without_toc(
        self,
        doc: fitz.Document,
        page_blocks: list[list[dict[str, Any]]] | None = None,
    ) -> list[Chapter]:
        """Parse page-by-page, using font-size heuristics to detect chapters."""
        all_blocks: list[dict[str, Any]] = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            current_page_blocks = (
                page_blocks[page_num]
                if page_blocks is not None else _extract_page_blocks(page)
            )
            page_height = page.rect.height
            median = _median_font_size(current_page_blocks)
            for blk in current_page_blocks:
                blk["_page"] = page_num
                blk["_page_height"] = page_height
                blk["_median_size"] = median
            all_blocks.extend(current_page_blocks)

        if not all_blocks:
            return [
                Chapter(
                    title="Document",
                    number=1,
                    sections=[Section(title="", level=2, paragraphs=[])],
                )
            ]

        # Split into chapters by heading-level-1 blocks
        chapters: list[Chapter] = []
        current_paragraphs: list[Paragraph] = []
        current_title = ""
        chap_num = 0

        for blk in all_blocks:
            median = blk["_median_size"]
            page_h = blk["_page_height"]

            if _is_heading(blk, median):
                level = _heading_level_from_size(blk["font_size"], median)
                if level == 1 and (current_paragraphs or current_title):
                    # Flush previous chapter
                    chap_num += 1
                    chapters.append(
                        Chapter(
                            title=current_title,
                            number=chap_num,
                            sections=[
                                Section(
                                    title="",
                                    level=2,
                                    paragraphs=current_paragraphs,
                                )
                            ],
                        )
                    )
                    current_paragraphs = []
                    current_title = blk["text"].strip()
                    continue

                if level == 1:
                    current_title = blk["text"].strip()
                    continue

                # Level 2-3 headings become paragraph with heading metadata
                current_paragraphs.append(
                    Paragraph(
                        text=blk["text"].strip(),
                        metadata={
                            "heading_level": level,
                            "structure_role": "heading",
                            "font_size": blk["font_size"],
                            "page": blk["_page"] + 1,
                            "bbox": tuple(blk["bbox"]),
                            "page_height": page_h,
                            "has_superscript": bool(blk.get("has_superscript")),
                            "superscript_markers": list(
                                blk.get("superscript_markers", []) or []
                            ),
                            "source_fragment_ids": [
                                str(blk.get("source_fragment_id", ""))
                            ] if blk.get("source_fragment_id") else [],
                        },
                    )
                )
            elif _is_footnote(blk, median, page_h):
                current_paragraphs.append(
                    Paragraph(
                        text=blk["text"].strip(),
                        metadata={
                            "is_footnote": True,
                            "structure_role": "footnote",
                            "font_size": blk["font_size"],
                            "page": blk["_page"] + 1,
                            "bbox": tuple(blk["bbox"]),
                            "page_height": page_h,
                            "has_superscript": bool(blk.get("has_superscript")),
                            "superscript_markers": list(
                                blk.get("superscript_markers", []) or []
                            ),
                            "source_fragment_ids": [
                                str(blk.get("source_fragment_id", ""))
                            ] if blk.get("source_fragment_id") else [],
                        },
                    )
                )
            else:
                current_paragraphs.append(
                        Paragraph(
                            text=blk["text"].strip(),
                            metadata={
                                "font_size": blk["font_size"],
                                "page": blk["_page"] + 1,
                                "bbox": tuple(blk["bbox"]),
                                "page_height": page_h,
                                "is_table": bool(blk.get("is_table")),
                                "structure_role": (
                                    "table" if blk.get("is_table") else "body"
                                ),
                                "has_superscript": bool(
                                    blk.get("has_superscript")
                                ),
                                "superscript_markers": list(
                                    blk.get("superscript_markers", []) or []
                                ),
                                "orientation": str(
                                    blk.get("orientation", "horizontal")
                                ),
                                "reading_order": int(
                                    blk.get("reading_order", 0)
                                ),
                                "source_fragment_ids": [
                                    str(blk.get("source_fragment_id", ""))
                                ] if blk.get("source_fragment_id") else [],
                            },
                        )
                )

        # Flush last chapter
        chap_num += 1
        chapters.append(
            Chapter(
                title=current_title or "Untitled",
                number=chap_num,
                sections=[
                    Section(title="", level=2, paragraphs=current_paragraphs)
                ],
            )
        )

        # Stitch cross-block/page continuations within each chapter section.
        for chapter in chapters:
            if self.structure_version >= 2:
                canonical, heading_audit = _canonicalize_chapter_heading(
                    chapter.title, chapter.all_paragraphs
                )
                chapter.sections = [Section(
                    title="",
                    level=2,
                    paragraphs=_merge_table_interrupted_continuations(
                        _merge_continuation_paragraphs(canonical)
                    ),
                )]
                chapter.metadata["heading_audit"] = heading_audit
            else:
                for section in chapter.sections:
                    section.paragraphs = _merge_continuation_paragraphs(
                        section.paragraphs
                    )
                chapter.metadata["heading_audit"] = {
                    "canonicalized": False,
                    "legacy_resume": True,
                }

        return chapters

    # ------------------------------------------------------------------
    # Page-range extraction
    # ------------------------------------------------------------------

    def _extract_pages(
        self,
        doc: fitz.Document,
        start_page: int,
        end_page: int,
        page_blocks: list[list[dict[str, Any]]] | None = None,
    ) -> list[Paragraph]:
        """Extract paragraphs from *start_page* (inclusive) to *end_page* (exclusive)."""
        paragraphs: list[Paragraph] = []
        for page_num in range(start_page, min(end_page, doc.page_count)):
            page = doc[page_num]
            blocks = (
                page_blocks[page_num]
                if page_blocks is not None else _extract_page_blocks(page)
            )
            median = _median_font_size(blocks)
            page_height = page.rect.height

            for blk in blocks:
                text = blk["text"].strip()
                if not text:
                    continue

                meta: dict[str, Any] = {
                    "font_size": blk["font_size"],
                    "page": page_num + 1,
                    "bbox": tuple(blk["bbox"]),
                    "page_height": page_height,
                    "is_table": bool(blk.get("is_table")),
                    "structure_role": (
                        "table" if blk.get("is_table") else "body"
                    ),
                    "has_superscript": bool(blk.get("has_superscript")),
                    "superscript_markers": list(
                        blk.get("superscript_markers", []) or []
                    ),
                    "orientation": str(blk.get("orientation", "horizontal")),
                    "reading_order": int(blk.get("reading_order", 0)),
                    "reading_order_mode": str(
                        blk.get("reading_order_mode", "source_order")
                    ),
                    "source_fragment_ids": [
                        str(blk.get("source_fragment_id", ""))
                    ] if blk.get("source_fragment_id") else [],
                }

                if _is_heading(blk, median):
                    meta["heading_level"] = _heading_level_from_size(
                        blk["font_size"], median
                    )
                    meta["structure_role"] = "heading"
                elif _is_footnote(blk, median, page_height):
                    meta["is_footnote"] = True
                    meta["structure_role"] = "footnote"

                paragraphs.append(Paragraph(text=text, metadata=meta))

        # Chapter-heading normalization runs before continuation stitching so a
        # repeated running title cannot interrupt a genuine cross-page paragraph.
        return (
            paragraphs
            if self.structure_version >= 2
            else _merge_continuation_paragraphs(paragraphs)
        )

    @staticmethod
    def _annotate_chapters(chapters: list[Chapter]) -> None:
        """Attach stable chapter identity to every paragraph for export/resume."""
        source_order = 0
        for position, chapter in enumerate(chapters, 1):
            chapter.metadata["tarjomeh_chapter_position"] = position
            first = True
            for section in chapter.sections:
                for paragraph in section.paragraphs:
                    paragraph.metadata["paragraph_id"] = f"p{source_order:07d}"
                    paragraph.metadata["source_order"] = source_order
                    paragraph.metadata["chapter_position"] = position
                    paragraph.metadata["chapter_number"] = chapter.number
                    paragraph.metadata["chapter_title"] = chapter.title
                    if first:
                        paragraph.metadata["chapter_start"] = True
                        first = False
                    source_order += 1


# ---------------------------------------------------------------------------
# DocLayoutParser (optional — requires doclayout-yolo)
# ---------------------------------------------------------------------------


class DocLayoutParser(BaseParser):
    """Layout-aware PDF parser using the *DocLayout-YOLO* model.

    Falls back to :class:`PyMuPDFParser` when the ``doclayout_yolo`` package is
    not installed.

    Region categories detected: ``text``, ``header``, ``footnote``,
    ``figure_caption``, ``table``, ``figure``.
    """

    _REGION_LABELS: dict[int, str] = {
        0: "text",
        1: "header",
        2: "footnote",
        3: "figure_caption",
        4: "table",
        5: "figure",
    }

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._model: Any = None
        self._available: bool | None = None

    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        """Attempt to lazily import and load the YOLO model.

        Returns *True* on success, *False* if the dependency is missing.
        """
        if self._available is not None:
            return self._available

        try:
            from doclayout_yolo import YOLOv10  # type: ignore[import-untyped]
        except ImportError:
            logger.info(
                "doclayout-yolo is not installed; falling back to PyMuPDFParser. "
                "Install with: pip install tarjomeh[pdf-layout]"
            )
            self._available = False
            return False

        model_path = self._model_path or "doclayout_yolo_docstructbench_imgsz1024.pt"
        try:
            self._model = YOLOv10(model_path)
            self._available = True
        except Exception:
            logger.warning(
                "Failed to load DocLayout-YOLO model from '%s'; "
                "falling back to PyMuPDFParser.",
                model_path,
                exc_info=True,
            )
            self._available = False

        return self._available

    # ------------------------------------------------------------------

    def parse(self, file_path: Path) -> Document:
        """Parse *file_path* using layout analysis, with PyMuPDF fallback."""
        file_path = self._ensure_file(file_path)

        if not self._load_model():
            return PyMuPDFParser().parse(file_path)

        return self._parse_with_layout(file_path)

    # ------------------------------------------------------------------

    def _parse_with_layout(self, file_path: Path) -> Document:
        """Run YOLO layout detection on each page and build a Document."""
        import pymupdf as _fitz

        doc = _fitz.open(str(file_path))
        try:
            chapters: list[Chapter] = []
            current_paragraphs: list[Paragraph] = []
            current_title = ""
            chap_num = 0

            for page_num in range(doc.page_count):
                page = doc[page_num]
                # Render page to pixmap for YOLO inference
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")

                regions = self._detect_regions(img_bytes, page)

                for region in regions:
                    label = region["label"]
                    text = region["text"].strip()
                    if not text:
                        continue

                    if label == "header":
                        # Flush previous chapter
                        if current_paragraphs or current_title:
                            chap_num += 1
                            chapters.append(
                                Chapter(
                                    title=current_title,
                                    number=chap_num,
                                    sections=[
                                        Section(
                                            title="",
                                            level=2,
                                            paragraphs=current_paragraphs,
                                        )
                                    ],
                                )
                            )
                            current_paragraphs = []
                        current_title = text
                    else:
                        meta: dict[str, Any] = {
                            "region_type": label,
                            "page": page_num + 1,
                        }
                        if label == "footnote":
                            meta["is_footnote"] = True
                        current_paragraphs.append(
                            Paragraph(text=text, metadata=meta)
                        )

            # Flush last chapter
            chap_num += 1
            chapters.append(
                Chapter(
                    title=current_title or "Untitled",
                    number=chap_num,
                    sections=[
                        Section(title="", level=2, paragraphs=current_paragraphs)
                    ],
                )
            )

            metadata = PyMuPDFParser._extract_metadata(doc)  # noqa: SLF001
            metadata["format_type"] = "pdf"
            metadata["parser"] = "doclayout-yolo"
            title = PyMuPDFParser._extract_title(  # noqa: SLF001
                doc, doc.get_toc()
            )

            return Document(
                title=title,
                chapters=chapters,
                metadata=metadata,
                raw_toc=[e[1] for e in doc.get_toc()] or None,
            )
        finally:
            doc.close()

    def _detect_regions(
        self,
        img_bytes: bytes,
        page: fitz.Page,
    ) -> list[dict[str, Any]]:
        """Run YOLO model on *img_bytes* and extract text from detected regions."""
        import io

        from PIL import Image  # type: ignore[import-untyped]

        img = Image.open(io.BytesIO(img_bytes))
        results = self._model.predict(img, imgsz=1024, conf=0.2, iou=0.45)  # type: ignore[union-attr]

        regions: list[dict[str, Any]] = []
        if not results or len(results) == 0:
            return regions

        result = results[0]
        boxes = result.boxes
        page_width, page_height = page.rect.width, page.rect.height
        img_w, img_h = img.size

        for box in boxes:
            cls_id = int(box.cls[0])
            label = self._REGION_LABELS.get(cls_id, "text")

            # Scale bounding box from image coords to PDF coords
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            pdf_rect = fitz.Rect(
                x0 * page_width / img_w,
                y0 * page_height / img_h,
                x1 * page_width / img_w,
                y1 * page_height / img_h,
            )
            text = page.get_text("text", clip=pdf_rect).strip()
            regions.append({"label": label, "text": text, "bbox": pdf_rect})

        # Sort top-to-bottom, then left-to-right
        regions.sort(key=lambda r: (r["bbox"].y0, r["bbox"].x0))
        return regions
