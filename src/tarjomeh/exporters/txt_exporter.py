"""TXT exporter stub for Tarjomeh."""

from __future__ import annotations

from pathlib import Path
from tarjomeh.exporters.base import BaseExporter, TranslatedDocument, BilingualMode


class TxtExporter(BaseExporter):
    """Simple TXT exporter for writing plain-text output files."""

    def export(
        self,
        document: TranslatedDocument,
        output_path: str | Path,
        bilingual_mode: BilingualMode = "target_only",
    ) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.write(f"# {document.title}\n")
            if document.author:
                f.write(f"Author: {document.author}\n")
            f.write("\n")
            for p in document.paragraphs:
                if bilingual_mode == "target_only":
                    f.write(p.translated_text + "\n\n")
                elif bilingual_mode == "inline":
                    f.write(p.source_text + "\n" + p.translated_text + "\n\n")
                else:  # side_by_side fallback
                    f.write(f"[EN]: {p.source_text}\n[FA]: {p.translated_text}\n\n")
        return out
