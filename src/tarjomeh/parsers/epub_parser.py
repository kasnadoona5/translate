"""EPUB document parser using *ebooklib*.

Reads EPUB 2/3 files, extracts spine-ordered chapter content, strips HTML
while preserving paragraph boundaries and heading hierarchy, and maps
everything onto the canonical :class:`~tarjomeh.parsers.base.Document` model.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from typing import Any

import ebooklib  # type: ignore[import-untyped]
from ebooklib import epub  # type: ignore[import-untyped]

from tarjomeh.parsers.base import (
    BaseParser,
    Chapter,
    Document,
    Paragraph,
    Section,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"<(h[1-6])\b[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_PARAGRAPH_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_tags(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _heading_level(tag: str) -> int:
    """Convert ``'h1'``…``'h6'`` to an integer 1–6."""
    return int(tag[1])


# ---------------------------------------------------------------------------
# EpubParser
# ---------------------------------------------------------------------------


class EpubParser(BaseParser):
    """Parse EPUB files using *ebooklib*.

    Spine ordering is respected so chapters appear in reading order.
    HTML ``<h1>``–``<h6>`` tags are mapped to heading levels and
    ``<p>`` tags become individual :class:`Paragraph` instances.
    """

    def parse(self, file_path: Path) -> Document:
        file_path = self._ensure_file(file_path)

        book = epub.read_epub(str(file_path), options={"ignore_ncx": False})

        title = self._get_metadata_field(book, "title") or file_path.stem
        author = self._get_metadata_field(book, "creator") or ""
        language = self._get_metadata_field(book, "language") or ""

        chapters = self._extract_chapters(book)

        # Build raw TOC from the NCX / navigation document
        raw_toc = self._extract_toc(book)

        return Document(
            title=title,
            chapters=chapters,
            metadata={
                "author": author,
                "language": language,
                "format_type": "epub",
            },
            raw_toc=raw_toc or None,
        )

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_metadata_field(book: epub.EpubBook, field: str) -> str:
        """Return first value for a Dublin Core *field*, or empty string."""
        values = book.get_metadata("DC", field)
        if values:
            # Each entry is (value, attrs) or just value
            val = values[0]
            if isinstance(val, tuple):
                return str(val[0])
            return str(val)
        return ""

    # ------------------------------------------------------------------
    # Chapter extraction (spine order)
    # ------------------------------------------------------------------

    def _extract_chapters(self, book: epub.EpubBook) -> list[Chapter]:
        """Walk the spine and convert each document item into a Chapter."""
        chapters: list[Chapter] = []
        chap_num = 0

        # Spine gives (item_id, linear) tuples
        spine_ids: list[str] = [
            item_id for item_id, _linear in book.spine
        ]

        items_by_id: dict[str, epub.EpubItem] = {
            item.get_id(): item for item in book.get_items()
        }

        for item_id in spine_ids:
            item = items_by_id.get(item_id)
            if item is None:
                continue
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            content = item.get_body_content()
            if content is None:
                continue
            body_html = content.decode("utf-8", errors="replace")

            sections = self._parse_html_content(body_html)
            if not sections:
                continue

            # Use first heading (if any) as chapter title
            chapter_title = ""
            for sec in sections:
                if sec.title:
                    chapter_title = sec.title
                    break

            chap_num += 1
            chapters.append(
                Chapter(
                    title=chapter_title,
                    number=chap_num,
                    sections=sections,
                    metadata={"spine_id": item_id},
                )
            )

        # Fallback: if no chapters produced, make a single-chapter document
        if not chapters:
            chapters.append(
                Chapter(
                    title="Document",
                    number=1,
                    sections=[Section(title="", level=2, paragraphs=[])],
                )
            )

        return chapters

    # ------------------------------------------------------------------
    # HTML → Sections / Paragraphs
    # ------------------------------------------------------------------

    def _parse_html_content(self, body_html: str) -> list[Section]:
        """Convert an HTML body string into a list of :class:`Section` objects.

        Headings create section boundaries; ``<p>`` tags create paragraphs.
        """
        sections: list[Section] = []
        current_section = Section(title="", level=2, paragraphs=[])

        # Tokenize: find all headings and paragraphs in document order
        tokens: list[tuple[str, str, int | None]] = []  # (type, text, heading_level)

        # Process headings
        for m in _HEADING_RE.finditer(body_html):
            tag = m.group(1).lower()
            text = _strip_tags(m.group(2))
            if text:
                tokens.append(("heading", text, _heading_level(tag)))

        # Process paragraphs
        for m in _PARAGRAPH_RE.finditer(body_html):
            text = _strip_tags(m.group(1))
            if text:
                tokens.append(("paragraph", text, None))

        # If regex extraction produced nothing, fall back to full text
        if not tokens:
            full_text = _strip_tags(body_html)
            if full_text:
                current_section.paragraphs.append(Paragraph(text=full_text))
                sections.append(current_section)
            return sections

        # Sort tokens by their position in the original HTML
        # We re-scan to get match positions
        all_positions: list[tuple[int, str, str, int | None]] = []
        for m in _HEADING_RE.finditer(body_html):
            text = _strip_tags(m.group(2))
            if text:
                all_positions.append(
                    (m.start(), "heading", text, _heading_level(m.group(1).lower()))
                )
        for m in _PARAGRAPH_RE.finditer(body_html):
            text = _strip_tags(m.group(1))
            if text:
                all_positions.append((m.start(), "paragraph", text, None))

        all_positions.sort(key=lambda t: t[0])

        for _, kind, text, h_level in all_positions:
            if kind == "heading":
                # Flush current section if it has content
                if current_section.paragraphs or current_section.title:
                    sections.append(current_section)
                current_section = Section(
                    title=text,
                    level=h_level or 2,
                    paragraphs=[],
                )
                # Also store heading as a paragraph for downstream use
                current_section.paragraphs.append(
                    Paragraph(
                        text=text,
                        metadata={"heading_level": h_level},
                    )
                )
            else:
                current_section.paragraphs.append(Paragraph(text=text))

        # Flush remaining
        if current_section.paragraphs or current_section.title:
            sections.append(current_section)

        return sections

    # ------------------------------------------------------------------
    # TOC extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_toc(book: epub.EpubBook) -> list[str]:
        """Extract flat list of TOC entry titles."""
        toc_entries: list[str] = []

        def _walk(items: Any) -> None:
            if isinstance(items, tuple) and len(items) == 2:
                # (Section, children) tuple
                section, children = items
                if hasattr(section, "title") and section.title:
                    toc_entries.append(section.title)
                _walk(children)
            elif isinstance(items, (list, tuple)):
                for item in items:
                    _walk(item)
            elif isinstance(items, epub.Link):
                if items.title:
                    toc_entries.append(items.title)

        _walk(book.toc)
        return toc_entries
