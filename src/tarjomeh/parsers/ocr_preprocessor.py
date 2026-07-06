"""OCR pre-processor for scanned PDFs.

Uses *ocrmypdf* (which wraps Tesseract) to add a text layer to image-only
PDF pages so that :class:`~tarjomeh.parsers.pdf_parser.PyMuPDFParser` can
extract text afterwards.

The ``ocrmypdf`` dependency is **optional** — install with::

    pip install tarjomeh[ocr]
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Default Tesseract language string: English + Persian
_DEFAULT_LANGUAGES = "eng+fas"


class OCRPreprocessor:
    """Add a text layer to scanned PDFs via *ocrmypdf*.

    Usage::

        preprocessor = OCRPreprocessor()
        text_pdf = preprocessor.preprocess(Path("scanned.pdf"))
        doc = PyMuPDFParser().parse(text_pdf)

    Parameters:
        languages: Tesseract language codes separated by ``+``
            (default ``eng+fas``).
        dpi: Resolution hint for Tesseract when images lack DPI metadata.
        jobs: Number of parallel OCR worker processes (``0`` = auto).
        output_dir: Directory for the output PDF.  If *None* a temporary
            directory is used.
    """

    def __init__(
        self,
        languages: str = _DEFAULT_LANGUAGES,
        dpi: int = 300,
        jobs: int = 0,
        output_dir: Path | None = None,
    ) -> None:
        self.languages = languages
        self.dpi = dpi
        self.jobs = jobs
        self.output_dir = output_dir
        self._ocrmypdf: Any = None

    # ------------------------------------------------------------------
    # Lazy import
    # ------------------------------------------------------------------

    def _ensure_ocrmypdf(self) -> Any:
        """Import *ocrmypdf* lazily; raise a clear error if missing."""
        if self._ocrmypdf is not None:
            return self._ocrmypdf

        try:
            import ocrmypdf  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "ocrmypdf is required for OCR preprocessing but is not "
                "installed.  Install it with:  pip install tarjomeh[ocr]"
            ) from exc

        self._ocrmypdf = ocrmypdf
        return ocrmypdf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(
        self,
        input_pdf_path: Path,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> Path:
        """Run OCR on *input_pdf_path* and return the path to the output PDF.

        The output PDF contains the original images **plus** an invisible text
        layer that can be extracted by any PDF text-extraction tool.

        Args:
            input_pdf_path: Path to the scanned PDF.
            progress_callback: Optional callable receiving status messages.

        Returns:
            Path to the OCR-processed PDF file.

        Raises:
            FileNotFoundError: If *input_pdf_path* does not exist.
            ImportError: If *ocrmypdf* is not installed.
            RuntimeError: If the OCR process fails.
        """
        input_pdf_path = Path(input_pdf_path).resolve()
        if not input_pdf_path.is_file():
            raise FileNotFoundError(f"Input PDF not found: {input_pdf_path}")

        ocrmypdf = self._ensure_ocrmypdf()

        output_path = self._resolve_output_path(input_pdf_path)
        self._report(progress_callback, f"Starting OCR on {input_pdf_path.name}…")

        try:
            exit_code = ocrmypdf.ocr(
                input_file=str(input_pdf_path),
                output_file=str(output_path),
                language=self.languages,
                deskew=True,
                rotate_pages=True,
                skip_text=True,  # don't re-OCR pages that already have text
                image_dpi=self.dpi,
                jobs=self.jobs if self.jobs > 0 else None,
                progress_bar=False,  # we handle progress ourselves
            )
        except self._ocrmypdf.exceptions.PriorOcrFoundError:
            # The PDF already has a text layer — just copy / return original
            logger.info(
                "PDF already contains a text layer; skipping OCR."
            )
            self._report(progress_callback, "PDF already has text layer — skipping.")
            return input_pdf_path
        except Exception as exc:
            raise RuntimeError(
                f"OCR processing failed for {input_pdf_path.name}: {exc}"
            ) from exc

        if exit_code != 0:
            raise RuntimeError(
                f"ocrmypdf exited with code {exit_code} for {input_pdf_path.name}"
            )

        self._report(
            progress_callback,
            f"OCR complete -> {output_path.name} "
            f"({output_path.stat().st_size / 1_048_576:.1f} MiB)",
        )
        logger.info("OCR output saved to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_output_path(self, input_path: Path) -> Path:
        """Compute the output path for the OCR'd PDF."""
        stem = input_path.stem
        suffix = input_path.suffix  # typically ".pdf"
        out_name = f"{stem}_ocr{suffix}"

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return self.output_dir / out_name

        # Use a temp directory that persists until the process ends
        tmp = Path(tempfile.mkdtemp(prefix="tarjomeh_ocr_"))
        return tmp / out_name

    @staticmethod
    def _report(
        callback: Callable[[str], None] | None,
        message: str,
    ) -> None:
        """Send *message* to the progress callback and the logger."""
        logger.info(message)
        if callback is not None:
            callback(message)
