"""Abstract base class for all Tarjomeh exporters.

Every concrete exporter (PDF, EPUB, DOCX, TXT, SRT) must subclass
:class:`BaseExporter` and implement the :meth:`export` method.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ── Bilingual mode type alias ────────────────────────────────────────
BilingualMode = Literal["target_only", "inline", "side_by_side"]


@dataclass
class TranslatedParagraph:
    """A single translated paragraph with associated metadata.

    Attributes
    ----------
    index : int
        Zero-based paragraph index within its parent chapter/section.
    source_text : str
        Original English text.
    translated_text : str
        Persian translation.
    heading_level : int | None
        ``1`` for chapter, ``2`` for section, ``3`` for subsection, etc.
        ``None`` for body paragraphs.
    metadata : dict[str, Any]
        Arbitrary metadata — e.g. ``timecode_start``, ``timecode_end``
        for SRT subtitles.
    """

    index: int
    source_text: str
    translated_text: str
    heading_level: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslatedDocument:
    """Container for a fully-translated document.

    Attributes
    ----------
    title : str
        Document title.
    author : str
        Author name(s).
    language : str
        BCP-47 language tag for the translation (default ``"fa"``).
    paragraphs : list[TranslatedParagraph]
        Ordered sequence of translated paragraphs.
    metadata : dict[str, Any]
        Document-level metadata (ISBN, publisher, etc.).
    """

    title: str = ""
    author: str = ""
    language: str = "fa"
    paragraphs: list[TranslatedParagraph] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseExporter(ABC):
    """Abstract base for format-specific exporters.

    Subclasses **must** implement :meth:`export`.

    Parameters
    ----------
    config : dict[str, Any] | None
        Optional exporter-specific configuration (font paths, page
        size overrides, etc.).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}

    @abstractmethod
    def export(
        self,
        document: TranslatedDocument,
        output_path: str | Path,
        bilingual_mode: BilingualMode = "target_only",
    ) -> Path:
        """Export *document* to *output_path* in the subclass's format.

        Parameters
        ----------
        document:
            Fully-translated document to export.
        output_path:
            Destination file path.
        bilingual_mode:
            ``"target_only"`` — Persian text only.
            ``"inline"``      — alternating source/translation paragraphs.
            ``"side_by_side"`` — parallel two-column layout (where
            applicable).

        Returns
        -------
        Path
            Resolved path to the written output file.
        """

    # ── shared helpers ───────────────────────────────────────────────

    @staticmethod
    def _resolve_font_path(font_name: str = "Vazirmatn-Regular.ttf") -> Path | None:
        """Attempt to locate a bundled font file.

        Looks for the font in the ``persian/fonts/`` directory shipped
        alongside this package.

        Returns
        -------
        Path | None
            Absolute path to the font file, or ``None`` if not found.
        """
        # Bundled path: src/tarjomeh/persian/fonts/
        bundled = (
            Path(__file__).resolve().parent.parent / "persian" / "fonts" / font_name
        )
        if bundled.is_file():
            return bundled

        logger.debug("Font %r not found at %s", font_name, bundled)
        return None

    @staticmethod
    def _shape_persian_text(text: str) -> str:
        """Reshape and apply bidi algorithm for correct RTL rendering.

        Uses ``arabic_reshaper`` + ``python-bidi`` — required for PDF
        backends that do not natively handle RTL shaping.

        Returns
        -------
        str
            Display-ready RTL text.
        """
        try:
            import arabic_reshaper  # type: ignore[import-untyped]
            from bidi.algorithm import get_display  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "arabic-reshaper and python-bidi are required for RTL "
                "text shaping. Install with:  "
                "pip install arabic-reshaper python-bidi"
            ) from exc

        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
