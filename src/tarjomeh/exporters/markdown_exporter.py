"""Markdown exporter with native first-occurrence footnotes."""

from __future__ import annotations

from pathlib import Path

from tarjomeh.exporters.base import BaseExporter, BilingualMode, TranslatedDocument
from tarjomeh.exporters.term_notes import document_term_notes, paragraph_note_parts


class MarkdownExporter(BaseExporter):
    """Write readable RTL-friendly Markdown without changing translated text."""

    def export(
        self,
        document: TranslatedDocument,
        output_path: str | Path,
        bilingual_mode: BilingualMode = "target_only",
    ) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {document.title}", ""]

        for paragraph in document.paragraphs:
            target = self._target_with_notes(
                paragraph.translated_text,
                paragraph.metadata,
            )
            level = min(paragraph.heading_level or 0, 6)
            if level:
                target = f"{'#' * level} {target}"
            if bilingual_mode == "target_only":
                lines.extend([target, ""])
            elif bilingual_mode == "inline":
                lines.extend([f"> {paragraph.source_text}", "", target, ""])
            else:
                lines.extend([
                    f"**English:** {paragraph.source_text}",
                    "",
                    f"**Persian:** {target}",
                    "",
                ])

        notes = document_term_notes(document)
        if notes:
            lines.extend(["## یادداشت‌ها", ""])
            for note in notes:
                lines.append(
                    f"[^{note['number']}]: {note['original']} "
                    f"({note['transliteration']})"
                )

        out.write_text(chr(10).join(lines).rstrip() + chr(10), encoding="utf-8")
        return out

    @staticmethod
    def _target_with_notes(text: str, metadata: dict) -> str:
        rendered = []
        for segment, ref in paragraph_note_parts(text, metadata):
            rendered.append(segment)
            if ref is not None:
                rendered.append(f"[^{ref['number']}]")
        return "".join(rendered)
