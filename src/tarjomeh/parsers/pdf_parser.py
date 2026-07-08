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
import statistics
from pathlib import Path
from typing import Any

import fitz  # pymupdf

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
_EDGE_ZONE = 0.07
_EDGE_MAX_CHARS = 40


def _join_block_lines(lines: list[str]) -> str:
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
            out = out + line.lstrip()
        else:
            out = out + " " + line.lstrip()
    return out.replace(_SOFT_HYPHEN, "").strip()


def _extract_page_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    """Extract text blocks with font metadata from a single PDF page.

    One entry per PDF text BLOCK (visual paragraph) — the block's lines are
    merged into flowing, dehyphenated prose. Short blocks hugging the very
    top/bottom edge of the page (running headers, page numbers) are dropped.

    Returns a list of dicts, each with keys:
    ``text``, ``font_size``, ``bbox``, ``font_name``.
    """
    blocks: list[dict[str, Any]] = []
    raw_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    page_height = page.rect.height or 1.0

    for block in raw_dict.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block
            continue

        line_texts: list[str] = []
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
                line_texts.append("".join(text_parts))

        merged_text = _join_block_lines(line_texts)
        if not merged_text:
            continue

        # Drop running headers / page numbers: short blocks entirely within
        # the top or bottom edge zone of the page.
        x0, y0, x1, y1 = block["bbox"]
        if len(merged_text) <= _EDGE_MAX_CHARS and (
            y1 < _EDGE_ZONE * page_height or y0 > (1 - _EDGE_ZONE) * page_height
        ):
            continue

        avg_size = sum(sizes) / len(sizes) if sizes else 0.0
        blocks.append(
            {
                "text": merged_text,
                "font_size": avg_size,
                "font_name": font_names[0] if font_names else "",
                "bbox": block["bbox"],  # (x0, y0, x1, y1)
            }
        )
    return blocks


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
        if (
            prev is not None
            and is_body
            and prev_is_body
            and prev.text
            and para.text
            and prev.text[-1] not in _SENTENCE_END_CHARS
            and (para.text[0].islower() or prev.text.endswith("-"))
        ):
            joiner = "" if prev.text.endswith("-") else " "
            prev.text = prev.text + joiner + para.text
        else:
            merged.append(para)
    return merged


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

            if toc:
                chapters = self._parse_with_toc(doc, toc)
            else:
                chapters = self._parse_without_toc(doc)

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
            paragraphs = self._extract_pages(doc, start_page, end_page)
            section = Section(title="", level=2, paragraphs=paragraphs)
            chapters.append(
                Chapter(
                    title=title,
                    number=chap_idx,
                    sections=[section],
                    metadata={"start_page": start_page + 1, "end_page": end_page},
                )
            )

        return chapters

    # ------------------------------------------------------------------
    # Fallback parsing (no TOC)
    # ------------------------------------------------------------------

    def _parse_without_toc(self, doc: fitz.Document) -> list[Chapter]:
        """Parse page-by-page, using font-size heuristics to detect chapters."""
        all_blocks: list[dict[str, Any]] = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_blocks = _extract_page_blocks(page)
            page_height = page.rect.height
            median = _median_font_size(page_blocks)
            for blk in page_blocks:
                blk["_page"] = page_num
                blk["_page_height"] = page_height
                blk["_median_size"] = median
            all_blocks.extend(page_blocks)

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
                            "font_size": blk["font_size"],
                        },
                    )
                )
            elif _is_footnote(blk, median, page_h):
                current_paragraphs.append(
                    Paragraph(
                        text=blk["text"].strip(),
                        metadata={
                            "is_footnote": True,
                            "font_size": blk["font_size"],
                        },
                    )
                )
            else:
                current_paragraphs.append(
                    Paragraph(
                        text=blk["text"].strip(),
                        metadata={"font_size": blk["font_size"]},
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
            for section in chapter.sections:
                section.paragraphs = _merge_continuation_paragraphs(section.paragraphs)

        return chapters

    # ------------------------------------------------------------------
    # Page-range extraction
    # ------------------------------------------------------------------

    def _extract_pages(
        self,
        doc: fitz.Document,
        start_page: int,
        end_page: int,
    ) -> list[Paragraph]:
        """Extract paragraphs from *start_page* (inclusive) to *end_page* (exclusive)."""
        paragraphs: list[Paragraph] = []
        for page_num in range(start_page, min(end_page, doc.page_count)):
            page = doc[page_num]
            blocks = _extract_page_blocks(page)
            median = _median_font_size(blocks)
            page_height = page.rect.height

            for blk in blocks:
                text = blk["text"].strip()
                if not text:
                    continue

                meta: dict[str, Any] = {
                    "font_size": blk["font_size"],
                    "page": page_num + 1,
                }

                if _is_heading(blk, median):
                    meta["heading_level"] = _heading_level_from_size(
                        blk["font_size"], median
                    )
                elif _is_footnote(blk, median, page_height):
                    meta["is_footnote"] = True

                paragraphs.append(Paragraph(text=text, metadata=meta))

        # Stitch paragraphs that continue across blocks / pages.
        return _merge_continuation_paragraphs(paragraphs)


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
        import fitz as _fitz  # local alias to avoid confusion

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
