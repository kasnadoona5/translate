"""Markdown document parser (regex-based, no external dependencies).

Maps heading levels to the canonical document model:

* ``#`` → Chapter
* ``##`` → Section
* ``###`` – ``######`` → Paragraph with ``heading_level`` metadata

Special block types are preserved in metadata so that downstream stages
can skip or treat them differently:

* Fenced code blocks (``````` … ```````) → ``metadata['code_block'] = True``
* Blockquotes (``> …``) → ``metadata['blockquote'] = True``
* YAML frontmatter (``--- … ---``) → stored in ``Document.metadata``
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from tarjomeh.parsers.base import (
    BaseParser,
    Chapter,
    Document,
    Paragraph,
    Section,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})", re.MULTILINE)
_BLOCKQUOTE_LINE_RE = re.compile(r"^>\s?(.*)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_yaml_frontmatter(raw: str) -> dict[str, str]:
    """Minimal YAML frontmatter parser (key: value lines only).

    We deliberately avoid pulling in PyYAML for this simple use case.
    """
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("\"'")
    return meta


# ---------------------------------------------------------------------------
# MarkdownParser
# ---------------------------------------------------------------------------


class MarkdownParser(BaseParser):
    """Parse Markdown files using regex-based heading / block detection."""

    def parse(self, file_path: Path) -> Document:
        file_path = self._ensure_file(file_path)

        raw = file_path.read_text(encoding="utf-8", errors="replace")

        # Extract YAML frontmatter
        doc_meta: dict[str, Any] = {"format_type": "markdown"}
        fm_match = _FRONTMATTER_RE.match(raw)
        if fm_match:
            fm_data = _parse_yaml_frontmatter(fm_match.group(1))
            doc_meta.update(fm_data)
            raw = raw[fm_match.end() :]

        title = str(doc_meta.get("title", file_path.stem))
        author = str(doc_meta.get("author", ""))
        doc_meta.setdefault("author", author)
        doc_meta.setdefault("language", "")

        chapters = self._build_structure(raw)

        if not chapters:
            chapters = [
                Chapter(
                    title=title,
                    number=1,
                    sections=[Section(title="", level=2, paragraphs=[])],
                )
            ]

        return Document(
            title=title,
            chapters=chapters,
            metadata=doc_meta,
        )

    # ------------------------------------------------------------------
    # Structure builder
    # ------------------------------------------------------------------

    def _build_structure(self, text: str) -> list[Chapter]:
        """Parse *text* into a list of :class:`Chapter` objects."""
        # Tokenize into blocks: headings, code fences, blockquotes, text
        tokens = self._tokenize(text)

        chapters: list[Chapter] = []
        current_chapter_title = ""
        current_chapter_num: int | None = None
        current_section = Section(title="", level=2, paragraphs=[])
        current_sections: list[Section] = [current_section]
        chap_count = 0

        for token in tokens:
            kind = token["kind"]

            if kind == "heading" and token["level"] == 1:
                # Flush previous chapter
                if current_sections and (
                    any(s.paragraphs for s in current_sections)
                    or current_chapter_title
                ):
                    chap_count += 1
                    chapters.append(
                        Chapter(
                            title=current_chapter_title,
                            number=current_chapter_num or chap_count,
                            sections=current_sections,
                        )
                    )
                current_chapter_title = token["text"]
                current_chapter_num = chap_count + 1
                current_section = Section(title="", level=2, paragraphs=[])
                current_sections = [current_section]

            elif kind == "heading" and token["level"] == 2:
                # New section
                current_section = Section(
                    title=token["text"], level=2, paragraphs=[]
                )
                current_sections.append(current_section)

            elif kind == "heading":
                # Sub-heading (level 3-6)
                current_section.paragraphs.append(
                    Paragraph(
                        text=token["text"],
                        metadata={"heading_level": token["level"]},
                    )
                )

            elif kind == "code_block":
                current_section.paragraphs.append(
                    Paragraph(
                        text=token["text"],
                        metadata={"code_block": True},
                    )
                )

            elif kind == "blockquote":
                current_section.paragraphs.append(
                    Paragraph(
                        text=token["text"],
                        metadata={"blockquote": True},
                    )
                )

            elif kind == "text":
                # Normal text paragraph(s)
                for para_text in self._split_paragraphs(token["text"]):
                    current_section.paragraphs.append(
                        Paragraph(text=para_text)
                    )

        # Flush last chapter
        if current_sections and (
            any(s.paragraphs for s in current_sections)
            or current_chapter_title
        ):
            chap_count += 1
            chapters.append(
                Chapter(
                    title=current_chapter_title or "Untitled",
                    number=current_chapter_num or chap_count,
                    sections=current_sections,
                )
            )

        return chapters

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[dict[str, Any]]:
        """Split Markdown *text* into typed tokens.

        Returns dicts with keys ``kind`` (``'heading'``, ``'code_block'``,
        ``'blockquote'``, ``'text'``) and ``text``, plus ``level`` for headings.
        """
        tokens: list[dict[str, Any]] = []
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # Fenced code block
            fence_match = _FENCE_RE.match(line)
            if fence_match:
                fence_marker = fence_match.group(1)
                fence_char = fence_marker[0]
                fence_len = len(fence_marker)
                code_lines: list[str] = []
                i += 1
                while i < len(lines):
                    if (
                        lines[i].startswith(fence_char * fence_len)
                        and lines[i].strip() == fence_char * fence_len
                    ):
                        i += 1
                        break
                    code_lines.append(lines[i])
                    i += 1
                tokens.append(
                    {"kind": "code_block", "text": "\n".join(code_lines)}
                )
                continue

            # Heading
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match:
                level = len(heading_match.group(1))
                h_text = heading_match.group(2).strip()
                tokens.append(
                    {"kind": "heading", "text": h_text, "level": level}
                )
                i += 1
                continue

            # Blockquote (consecutive > lines)
            if _BLOCKQUOTE_LINE_RE.match(line):
                bq_lines: list[str] = []
                while i < len(lines):
                    bq_match = _BLOCKQUOTE_LINE_RE.match(lines[i])
                    if bq_match:
                        bq_lines.append(bq_match.group(1))
                        i += 1
                    else:
                        break
                tokens.append(
                    {"kind": "blockquote", "text": "\n".join(bq_lines)}
                )
                continue

            # Blank line — skip
            if not line.strip():
                i += 1
                continue

            # Normal text — accumulate until next special line
            text_lines: list[str] = []
            while i < len(lines):
                l = lines[i]
                if (
                    not l.strip()
                    or _FENCE_RE.match(l)
                    or re.match(r"^#{1,6}\s+", l)
                    or _BLOCKQUOTE_LINE_RE.match(l)
                ):
                    break
                text_lines.append(l)
                i += 1
            if text_lines:
                tokens.append({"kind": "text", "text": "\n".join(text_lines)})

        return tokens

    # ------------------------------------------------------------------
    # Paragraph splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split a text block into paragraphs on blank-line boundaries."""
        paragraphs: list[str] = []
        current: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
            else:
                current.append(stripped)
        if current:
            paragraphs.append(" ".join(current))
        return [p for p in paragraphs if p]
