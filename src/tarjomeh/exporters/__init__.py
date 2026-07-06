"""Export format registry and factory for Tarjomeh.

Provides ``get_exporter(format)`` to obtain the appropriate
:class:`BaseExporter` subclass for a given output format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tarjomeh.exporters.base import BaseExporter

if TYPE_CHECKING:
    pass  # forward refs only

# ── Supported format → module/class mapping ──────────────────────────
_EXPORTER_REGISTRY: dict[str, tuple[str, str]] = {
    "pdf":  ("tarjomeh.exporters.pdf_exporter",  "PdfExporter"),
    "epub": ("tarjomeh.exporters.epub_exporter", "EpubExporter"),
    "docx": ("tarjomeh.exporters.docx_exporter", "DocxExporter"),
    "txt":  ("tarjomeh.exporters.txt_exporter",  "TxtExporter"),
    "srt":  ("tarjomeh.exporters.srt_exporter",  "SrtExporter"),
}


def get_exporter(fmt: str) -> type[BaseExporter]:
    """Return the :class:`BaseExporter` subclass for *fmt*.

    Parameters
    ----------
    fmt:
        Output format string — one of ``"pdf"``, ``"epub"``, ``"docx"``,
        ``"txt"``, ``"srt"`` (case-insensitive).

    Returns
    -------
    type[BaseExporter]
        The exporter **class** (not an instance).  Instantiate it
        yourself with any required constructor arguments.

    Raises
    ------
    ValueError
        If *fmt* is not a recognised export format.
    """
    key = fmt.strip().lower()
    if key not in _EXPORTER_REGISTRY:
        supported = ", ".join(sorted(_EXPORTER_REGISTRY))
        raise ValueError(
            f"Unknown export format {fmt!r}. "
            f"Supported formats: {supported}"
        )

    module_path, class_name = _EXPORTER_REGISTRY[key]

    # Lazy import — only load the exporter module when requested.
    import importlib

    module = importlib.import_module(module_path)
    exporter_cls: type[BaseExporter] = getattr(module, class_name)
    return exporter_cls


__all__ = ["BaseExporter", "get_exporter"]
