"""Document parsers and the canonical document model.

Every format-specific parser converts its native structure into the shared
:class:`~tarjomeh.parsers.base.Document` model.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from tarjomeh.parsers.base import (
    EXTENSION_PARSER_MAP,
    SUPPORTED_EXTENSIONS,
    BaseParser,
    Chapter,
    Document,
    Paragraph,
    Section,
)

__all__ = [
    "BaseParser",
    "Chapter",
    "Document",
    "EXTENSION_PARSER_MAP",
    "Paragraph",
    "SUPPORTED_EXTENSIONS",
    "Section",
    "get_parser",
]


def get_parser(file_path: str | Path) -> BaseParser:
    """Return a parser instance for *file_path*, chosen by extension.

    Raises:
        ValueError: If no parser is registered for the file's suffix.
    """
    suffix = Path(file_path).suffix.lower()
    target = EXTENSION_PARSER_MAP.get(suffix)
    if target is None:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported extensions: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
    module_path, class_name = target.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)()
