"""Plain-text file parser with heuristic structure detection.

Reads UTF-8 text files and attempts to infer document structure using
simple heuristics:

* Lines matching ``Chapter N`` / ``CHAPTER N`` patterns → chapter boundaries.
* Double blank lines → section separators.
* ALL-CAPS short lines → potential headings.

Falls back to a single-chapter document when no structure can be detected.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from tarjomeh.parsers.base import (
    BaseParser,
    Chapter,
    Document,
    Paragraph,
    Section,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

# Matches lines like "Chapter 1", "CHAPTER IV", "Chapter 12: Title"
_CHAPTER_RE = re.compile(
    r"^\s*(?:chapter|CHAPTER)\s+"
    r"(\d+|[IVXLCDM]+)"
    r"(?:\s*[:.\-—–]\s*(.+))?\s*$",
    re.IGNORECASE,
)

# Matches purely numeric chapter markers like "1", "12"
_NUMERIC_CHAPTER_RE = re.compile(r"^\s*(\d{1,3})\s*$")

# All-caps lines (at least 4 chars, no lower-case) that may be headings
_ALL_CAPS_RE = re.compile(r"^[A-Z\s\d\-:,.!?]{4,}$")

_MAX_HEADING_LENGTH = 120  # chars — longer lines are not headings


def _is_all_caps_heading(line: str) -> bool:
    """Return *True* if *line* looks like an all-caps heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_LENGTH:
        return False
    # Must contain at least one letter
    if not any(c.isalpha() for c in stripped):
        return False
    return bool(_ALL_CAPS_RE.match(stripped))


def _parse_chapter_number(raw: str) -> int | None:
    """Convert chapter number string (digits or roman numerals) to int."""
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    # Simple Roman numeral conversion
    roman_map = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    upper = raw.upper()
    if all(c in roman_map for c in upper):
        total = 0
        prev = 0
        for c in reversed(upper):
            val = roman_map[c]
            if val < prev:
                total -= val
            else:
                total += val
            prev = val
        return total if total > 0 else None
    return None


# ---------------------------------------------------------------------------
# TxtParser
# ---------------------------------------------------------------------------


class TxtParser(BaseParser):
    """Parse plain UTF-8 text files with heuristic structure detection."""

    def parse(self, file_path: Path) -> Document:
        file_path = self._ensure_file(file_path)

        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        if not lines:
            return Document(
                title=file_path.stem,
                chapters=[
                    Chapter(
                        title="Empty",
                        number=1,
                        sections=[Section(title="", level=2, paragraphs=[])],
                    )
                ],
                metadata={"format_type": "txt", "author": "", "language": ""},
            )

        chapters = self._detect_chapters(lines)

        # Fallback: no structure found → single chapter
        if not chapters:
            chapters = [self._single_chapter(lines)]

        return Document(
            title=self._guess_title(file_path, chapters),
            chapters=chapters,
            metadata={"format_type": "txt", "author": "", "language": ""},
        )

    # ------------------------------------------------------------------
    # Chapter detection
    # ------------------------------------------------------------------

    def _detect_chapters(self, lines: list[str]) -> list[Chapter]:
        """Attempt to split *lines* into chapters based on heuristics."""
        # First pass: look for "Chapter N" patterns
        chapter_starts: list[tuple[int, str, int | None]] = []
        for idx, line in enumerate(lines):
            m = _CHAPTER_RE.match(line)
            if m:
                num = _parse_chapter_number(m.group(1))
                title = m.group(2).strip() if m.group(2) else ""
                chapter_starts.append((idx, title, num))

        if chapter_starts:
            return self._split_by_markers(lines, chapter_starts)

        # Second pass: all-caps lines could be chapter headings
        # Only use if there are a reasonable number of them (2-50)
        caps_starts: list[tuple[int, str, int | None]] = []
        for idx, line in enumerate(lines):
            if _is_all_caps_heading(line):
                caps_starts.append((idx, line.strip(), None))

        if 2 <= len(caps_starts) <= 50:
            return self._split_by_markers(lines, caps_starts)

        return []

    def _split_by_markers(
        self,
        lines: list[str],
        markers: list[tuple[int, str, int | None]],
    ) -> list[Chapter]:
        """Split *lines* into chapters at the given *markers*.

        Each marker is ``(line_index, title, chapter_number_or_None)``.
        """
        chapters: list[Chapter] = []

        # If there's content before the first marker, add as "Preface"
        first_marker_idx = markers[0][0]
        if first_marker_idx > 0:
            pre_lines = lines[:first_marker_idx]
            pre_paragraphs = self._lines_to_paragraphs(pre_lines)
            if pre_paragraphs:
                chapters.append(
                    Chapter(
                        title="Preface",
                        number=None,
                        sections=[
                            Section(title="", level=2, paragraphs=pre_paragraphs)
                        ],
                    )
                )

        for i, (start_idx, title, chap_num) in enumerate(markers):
            end_idx = markers[i + 1][0] if i + 1 < len(markers) else len(lines)
            # Skip the marker line itself
            body_lines = lines[start_idx + 1 : end_idx]
            sections = self._lines_to_sections(body_lines)

            chapters.append(
                Chapter(
                    title=title or f"Chapter {chap_num or i + 1}",
                    number=chap_num or i + 1,
                    sections=sections,
                )
            )

        return chapters

    # ------------------------------------------------------------------
    # Section detection (double blank lines)
    # ------------------------------------------------------------------

    def _lines_to_sections(self, lines: list[str]) -> list[Section]:
        """Split *lines* into sections at double blank lines."""
        sections: list[Section] = []
        current_lines: list[str] = []
        blank_count = 0
        for line in lines:
            if not line.strip():
                blank_count += 1
                if blank_count >= 2 and current_lines:
                    paragraphs = self._lines_to_paragraphs(current_lines)
                    if paragraphs:
                        sections.append(
                            Section(title="", level=2, paragraphs=paragraphs)
                        )
                    current_lines = []
                    blank_count = 0
                else:
                    current_lines.append("")
                continue
            blank_count = 0
            current_lines.append(line)

        # Flush remaining
        if current_lines:
            paragraphs = self._lines_to_paragraphs(current_lines)
            if paragraphs:
                sections.append(
                    Section(title="", level=2, paragraphs=paragraphs)
                )

        if not sections:
            sections.append(Section(title="", level=2, paragraphs=[]))

        return sections

    # ------------------------------------------------------------------
    # Paragraph assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _lines_to_paragraphs(lines: list[str]) -> list[Paragraph]:
        """Group consecutive non-blank lines into paragraphs.

        Single blank lines separate paragraphs.
        """
        paragraphs: list[Paragraph] = []
        current: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    paragraphs.append(Paragraph(text=" ".join(current)))
                    current = []
            else:
                current.append(stripped)

        if current:
            paragraphs.append(Paragraph(text=" ".join(current)))

        return paragraphs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _single_chapter(lines: list[str]) -> Chapter:
        """Wrap all *lines* into a single chapter with one section."""
        paragraphs: list[Paragraph] = []
        current: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    paragraphs.append(Paragraph(text=" ".join(current)))
                    current = []
            else:
                current.append(stripped)

        if current:
            paragraphs.append(Paragraph(text=" ".join(current)))

        return Chapter(
            title="Document",
            number=1,
            sections=[Section(title="", level=2, paragraphs=paragraphs)],
        )

    @staticmethod
    def _guess_title(file_path: Path, chapters: list[Chapter]) -> str:
        """Best-effort title: first chapter title or filename stem."""
        for ch in chapters:
            if ch.title and ch.title not in ("Preface", "Document"):
                return ch.title
        return file_path.stem
