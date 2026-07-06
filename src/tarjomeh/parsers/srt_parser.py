"""SRT subtitle file parser.

Parses ``.srt`` (SubRip) subtitle files into the canonical document model.
Each subtitle cue becomes a :class:`~tarjomeh.parsers.base.Paragraph` whose
``metadata`` carries the original *index* and *timecode*.  Cues are grouped
into a single chapter for short content, or split at hour boundaries for
feature-length films.
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
# SRT format regexes
# ---------------------------------------------------------------------------

# Matches a timecode line: "00:01:23,456 --> 00:01:25,789"
_TIMECODE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
)


def _parse_timecode_hour(tc: str) -> int:
    """Extract the hour component from a timecode like ``'01:23:45,678'``."""
    return int(tc.split(":")[0])


# ---------------------------------------------------------------------------
# SrtParser
# ---------------------------------------------------------------------------


class SrtParser(BaseParser):
    """Parse SRT subtitle files.

    Each subtitle block is converted to a :class:`Paragraph` with:

    * ``text`` — the dialogue / subtitle text (translatable).
    * ``metadata['index']`` — the 1-based cue index.
    * ``metadata['timecode']`` — the full timecode string
      (``'00:01:23,456 --> 00:01:25,789'``).
    * ``metadata['format']`` — always ``'srt'``.

    For films longer than one hour, cues are split into per-hour chapters
    to keep translation chunks manageable.
    """

    def parse(self, file_path: Path) -> Document:
        file_path = self._ensure_file(file_path)

        raw = file_path.read_text(encoding="utf-8-sig", errors="replace")
        cues = self._parse_cues(raw)

        if not cues:
            return Document(
                title=file_path.stem,
                chapters=[
                    Chapter(
                        title="Subtitles",
                        number=1,
                        sections=[Section(title="", level=2, paragraphs=[])],
                    )
                ],
                metadata={"format_type": "srt", "author": "", "language": ""},
            )

        chapters = self._group_into_chapters(cues, file_path.stem)

        return Document(
            title=file_path.stem,
            chapters=chapters,
            metadata={
                "format_type": "srt",
                "author": "",
                "language": "",
                "cue_count": len(cues),
            },
        )

    # ------------------------------------------------------------------
    # Cue parsing
    # ------------------------------------------------------------------

    def _parse_cues(self, raw_text: str) -> list[Paragraph]:
        """Parse raw SRT text into a list of :class:`Paragraph` cues."""
        cues: list[Paragraph] = []

        # Split on blank lines to get blocks
        blocks = re.split(r"\n\s*\n", raw_text.strip())

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = block.splitlines()
            if len(lines) < 2:
                continue

            # Line 0: cue index (integer)
            index_line = lines[0].strip()
            if not index_line.isdigit():
                # Sometimes the index is missing; try to recover
                logger.debug("Unexpected SRT index line: %r", index_line)
                continue

            cue_index = int(index_line)

            # Line 1: timecode
            tc_match = _TIMECODE_RE.match(lines[1].strip())
            if not tc_match:
                logger.debug(
                    "Invalid timecode in cue %d: %r", cue_index, lines[1]
                )
                continue

            timecode = lines[1].strip()

            # Lines 2+: subtitle text
            text_lines = [ln.strip() for ln in lines[2:] if ln.strip()]
            text = "\n".join(text_lines)

            if not text:
                continue

            cues.append(
                Paragraph(
                    text=text,
                    metadata={
                        "index": cue_index,
                        "timecode": timecode,
                        "format": "srt",
                        "start_tc": tc_match.group(1),
                        "end_tc": tc_match.group(2),
                    },
                )
            )

        return cues

    # ------------------------------------------------------------------
    # Chapter grouping
    # ------------------------------------------------------------------

    def _group_into_chapters(
        self,
        cues: list[Paragraph],
        doc_stem: str,
    ) -> list[Chapter]:
        """Group cues into chapters — one per hour for long content."""
        # Find the maximum hour in the cues
        max_hour = 0
        for cue in cues:
            start_tc = cue.metadata.get("start_tc", "00:00:00,000")
            hour = _parse_timecode_hour(start_tc)
            if hour > max_hour:
                max_hour = hour

        if max_hour == 0:
            # All cues within the first hour → single chapter
            return [
                Chapter(
                    title=doc_stem,
                    number=1,
                    sections=[Section(title="", level=2, paragraphs=cues)],
                )
            ]

        # Split by hour boundaries
        hour_buckets: dict[int, list[Paragraph]] = {}
        for cue in cues:
            start_tc = cue.metadata.get("start_tc", "00:00:00,000")
            hour = _parse_timecode_hour(start_tc)
            hour_buckets.setdefault(hour, []).append(cue)

        chapters: list[Chapter] = []
        for hour in sorted(hour_buckets):
            chap_title = f"Hour {hour}" if hour > 0 else "Opening"
            chapters.append(
                Chapter(
                    title=chap_title,
                    number=hour + 1,
                    sections=[
                        Section(
                            title="",
                            level=2,
                            paragraphs=hour_buckets[hour],
                        )
                    ],
                )
            )

        return chapters
