"""SRT (SubRip) subtitle file exporter for Tarjomeh."""

from __future__ import annotations

from pathlib import Path
from tarjomeh.exporters.base import BaseExporter, TranslatedDocument, BilingualMode


class SrtExporter(BaseExporter):
    """SRT subtitle exporter.

    Reconstructs translated subtitles, preserving timing/indices.
    """

    def export(
        self,
        document: TranslatedDocument,
        output_path: str | Path,
        bilingual_mode: BilingualMode = "target_only",
    ) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with out.open("w", encoding="utf-8-sig") as f:
            for idx, p in enumerate(document.paragraphs, 1):
                # Retrieve index and timecode from metadata, or fallback
                cue_index = p.metadata.get("index", idx)
                timecode = p.metadata.get("timecode", "00:00:00,000 --> 00:00:00,000")

                f.write(f"{cue_index}\n")
                f.write(f"{timecode}\n")

                # Format text based on bilingual mode
                if bilingual_mode == "target_only":
                    text = p.translated_text
                elif bilingual_mode == "inline":
                    text = f"{p.source_text}\n{p.translated_text}"
                else:  # side_by_side
                    text = f"{p.source_text} | {p.translated_text}"

                f.write(f"{text}\n\n")

        return out
