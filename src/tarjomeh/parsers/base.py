"""Base data structures and abstract parser for document ingestion.

Defines the canonical document model used across the entire Tarjomeh pipeline:
Paragraph → Section → Chapter → Document.  Every format-specific parser
converts its native structure into this common representation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Paragraph:
    """Smallest translatable unit — a single paragraph or subtitle cue.

    Attributes:
        text: The paragraph text content (may contain inline markdown/HTML).
        metadata: Flexible dict for format-specific info, e.g.
            ``font_size``, ``is_footnote``, ``timecode``, ``heading_level``,
            ``code_block``, ``blockquote``.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Convenience helpers --------------------------------------------------

    @property
    def is_translatable(self) -> bool:
        """Return *False* for code blocks, image-only paragraphs, etc."""
        if self.metadata.get("code_block", False):
            return False
        if not self.text.strip():
            return False
        return True

    @property
    def is_footnote(self) -> bool:
        return bool(self.metadata.get("is_footnote", False))

    @property
    def heading_level(self) -> int | None:
        """Return heading level (1-6) if this paragraph is a heading, else *None*."""
        return self.metadata.get("heading_level")


@dataclass
class Section:
    """A logical section inside a chapter (maps to ``## …`` or *Heading 2*).

    Attributes:
        title: Section heading text; may be empty for untitled sections.
        level: Nesting depth (2 = section, 3 = subsection, …).
        paragraphs: Ordered content paragraphs within this section.
    """

    title: str
    level: int
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class Chapter:
    """A chapter — the primary high-level division of a document.

    Attributes:
        title: Chapter title; empty string when unknown.
        number: 1-based chapter number, or *None* if unnumbered.
        sections: Ordered list of sections within the chapter.
        metadata: Extra per-chapter information (e.g. original page range).
    """

    title: str
    number: int | None = None
    sections: list[Section] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Convenience -----------------------------------------------------------

    @property
    def all_paragraphs(self) -> list[Paragraph]:
        """Flatten all paragraphs across every section."""
        return [p for sec in self.sections for p in sec.paragraphs]


@dataclass
class Document:
    """Top-level container produced by every parser.

    Attributes:
        title: Document / book title.
        chapters: Ordered list of chapters.
        metadata: Global metadata such as *author*, *language*,
            *format_type* (``"pdf"``, ``"epub"``, …).
        raw_toc: Optional table-of-contents lines extracted verbatim from
            the source file (useful for PDF bookmarks, EPUB NCX, etc.).
    """

    title: str
    chapters: list[Chapter] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_toc: list[str] | None = None

    # Convenience -----------------------------------------------------------

    @property
    def all_paragraphs(self) -> list[Paragraph]:
        """Flatten every paragraph in the entire document."""
        return [p for ch in self.chapters for p in ch.all_paragraphs]

    @property
    def format_type(self) -> str:
        """Shortcut for ``metadata['format_type']``."""
        return self.metadata.get("format_type", "unknown")

    @property
    def author(self) -> str:
        return self.metadata.get("author", "")


# ---------------------------------------------------------------------------
# Extension → parser mapping
# ---------------------------------------------------------------------------

# Maps lower-cased file suffix (with dot) to the module attribute path used by
# ``get_parser`` in ``__init__.py``.  This lives here so that every parser
# module can import it without circular deps.

EXTENSION_PARSER_MAP: dict[str, str] = {
    ".pdf": "tarjomeh.parsers.pdf_parser.PyMuPDFParser",
    ".epub": "tarjomeh.parsers.epub_parser.EpubParser",
    ".docx": "tarjomeh.parsers.docx_parser.DocxParser",
    ".txt": "tarjomeh.parsers.txt_parser.TxtParser",
    ".text": "tarjomeh.parsers.txt_parser.TxtParser",
    ".srt": "tarjomeh.parsers.srt_parser.SrtParser",
    ".md": "tarjomeh.parsers.markdown_parser.MarkdownParser",
    ".markdown": "tarjomeh.parsers.markdown_parser.MarkdownParser",
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(EXTENSION_PARSER_MAP)


# ---------------------------------------------------------------------------
# Abstract base parser
# ---------------------------------------------------------------------------


class BaseParser(ABC):
    """Abstract base for all format-specific document parsers.

    Subclasses **must** implement :meth:`parse`.  Optionally they may
    override :meth:`can_handle` for finer-grained format detection beyond
    file extension.
    """

    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """Parse *file_path* and return a :class:`Document`.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            ValueError: If the file cannot be parsed by this parser.
        """

    # Optional overrides ----------------------------------------------------

    def can_handle(self, file_path: Path) -> bool:
        """Return *True* if this parser supports the given file.

        The default implementation checks the file extension against
        :data:`EXTENSION_PARSER_MAP`.  Override for sniffing magic bytes, etc.
        """
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS

    # Shared helpers --------------------------------------------------------

    @staticmethod
    def _ensure_file(file_path: Path) -> Path:
        """Resolve *file_path* and raise if it does not exist."""
        resolved = file_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"File not found: {resolved}")
        return resolved
