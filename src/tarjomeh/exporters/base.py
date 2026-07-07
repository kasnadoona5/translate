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
        alongside this package. Downloads it automatically from GitHub
        if not found.

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

        # Automatic download fallback
        try:
            bundled.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://github.com/rastikerdar/vazirmatn/raw/master/fonts/ttf/{font_name}"
            logger.info("Font %s not found locally. Downloading from %s...", font_name, url)
            import urllib.request
            urllib.request.urlretrieve(url, str(bundled))
            if bundled.is_file():
                logger.info("Successfully downloaded %s to %s", font_name, bundled)
                return bundled
        except Exception as e:
            logger.error("Failed to automatically download font %s: %s", font_name, e)

        logger.debug("Font %r not found at %s", font_name, bundled)
        return None

    @staticmethod
    def _shape_persian_text(text: str) -> str:
        """Reshape and apply bidi algorithm for correct RTL rendering.

        Uses ``arabic_reshaper`` + ``python-bidi``.

        .. warning::
            Only suitable for text that renders on a SINGLE line. Applying
            ``get_display`` to a multi-line block before the PDF engine wraps
            it reverses the visual line order (paragraphs read bottom-to-top).
            For wrapped paragraphs use :meth:`_shape_persian_lines` instead.

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

    @staticmethod
    def _shape_persian_lines(
        text: str,
        font_name: str,
        font_size: float,
        avail_width: float,
        escape_xml: bool = True,
    ) -> str:
        """Wrap-first-then-bidi shaping for correct MULTI-LINE RTL in ReportLab.

        The Unicode bidi algorithm is defined per rendered line, so the text
        must be wrapped to the target width FIRST and only then visually
        reordered line by line:

        1. Reshape the whole paragraph (contextual glyph forms — word-internal,
           unaffected by line breaks).
        2. Wrap the shaped string with ReportLab's own ``simpleSplit`` so the
           measured widths match exactly what will be drawn.
        3. Apply ``get_display`` to EACH line individually.
        4. Join with ``\\n`` for use with ``XPreformatted`` (which honours the
           line breaks and never re-wraps — re-wrapping would push each line's
           logical start onto an orphan line).

        Verified empirically: this is the only recipe that renders Persian
        paragraphs top-to-bottom with correct embedded Latin/number runs.

        Returns
        -------
        str
            Newline-joined, display-ready RTL lines (XML-escaped by default).
        """
        try:
            import arabic_reshaper  # type: ignore[import-untyped]
            from bidi.algorithm import get_display  # type: ignore[import-untyped]
            from reportlab.lib.utils import simpleSplit  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "arabic-reshaper, python-bidi and reportlab are required for "
                "RTL PDF export. Install with:  "
                "pip install arabic-reshaper python-bidi reportlab"
            ) from exc

        if not text or not text.strip():
            return ""

        shaped = arabic_reshaper.reshape(text)
        lines = simpleSplit(shaped, font_name, font_size, avail_width)
        visual = [get_display(line) for line in lines]
        if escape_xml:
            from xml.sax.saxutils import escape
            visual = [escape(v) for v in visual]
        return "\n".join(visual)
