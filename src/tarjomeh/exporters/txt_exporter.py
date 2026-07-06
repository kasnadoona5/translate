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
        RLM = "\u200f"
        with out.open("w", encoding="utf-8") as f:
            f.write(f"# {document.title}\n")
            if document.author:
                f.write(f"Author: {document.author}\n")
            f.write("\n")
            for p in document.paragraphs:
                translated_rlm = f"{RLM}{p.translated_text}{RLM}"
                if bilingual_mode == "target_only":
                    f.write(translated_rlm + "\n\n")
                elif bilingual_mode == "inline":
                    f.write(p.source_text + "\n" + translated_rlm + "\n\n")
                else:  # side_by_side fallback
                    f.write(f"[EN]: {p.source_text}\n[FA]: {translated_rlm}\n\n")
        return out
