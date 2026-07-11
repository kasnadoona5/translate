"""DOCX document exporter for Tarjomeh."""

from __future__ import annotations

import logging
from pathlib import Path
from tarjomeh.exporters.base import BaseExporter, TranslatedDocument, BilingualMode
from tarjomeh.exporters.term_notes import document_term_notes, paragraph_note_parts

logger = logging.getLogger(__name__)

try:
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


class DocxExporter(BaseExporter):
    """DOCX exporter.

    Generates RTL-formatted DOCX files with optional bilingual layouts.
    """

    def export(
        self,
        document: TranslatedDocument,
        output_path: str | Path,
        bilingual_mode: BilingualMode = "target_only",
    ) -> Path:
        if not HAS_DOCX:
            raise ImportError(
                "python-docx is required for DOCX export. "
                "Install it with:  pip install python-docx"
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        doc = docx.Document()
        doc.core_properties.title = document.title
        doc.core_properties.author = document.author

        # Helpers to set RTL direction and Vazirmatn font
        def make_paragraph_rtl(p_obj) -> None:
            p_obj.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            pPr = p_obj._element.get_or_add_pPr()
            bidi = pPr.find(qn('w:bidi'))
            if bidi is None:
                bidi_el = OxmlElement('w:bidi')
                bidi_el.set(qn('w:val'), '1')
                pPr.append(bidi_el)

        def make_run_rtl(r_obj) -> None:
            rPr = r_obj._r.get_or_add_rPr()
            rtl = rPr.find(qn('w:rtl'))
            if rtl is None:
                rtl_el = OxmlElement('w:rtl')
                rtl_el.set(qn('w:val'), '1')
                rPr.append(rtl_el)
            
            # Set complex script font to Vazirmatn
            rFonts = rPr.find(qn('w:rFonts'))
            if rFonts is None:
                rFonts_el = OxmlElement('w:rFonts')
                rFonts_el.set(qn('w:cs'), 'Vazirmatn')
                rPr.append(rFonts_el)
            else:
                rFonts.set(qn('w:cs'), 'Vazirmatn')
            r_obj.font.name = 'Vazirmatn'

        def add_target_runs(p_obj, text: str, metadata: dict) -> None:
            for segment, ref in paragraph_note_parts(text, metadata):
                if segment:
                    run = p_obj.add_run(segment)
                    make_run_rtl(run)
                if ref is not None:
                    marker = p_obj.add_run(str(ref.get("display_number", ref["number"])))
                    marker.font.superscript = True
                    make_run_rtl(marker)

        if bilingual_mode == "side_by_side":
            table = doc.add_table(rows=0, cols=2)
            table.autofit = False
            
            for p in document.paragraphs:
                row = table.add_row()
                cell_en, cell_fa = row.cells
                
                # Left Column: English (LTR)
                p_en = cell_en.paragraphs[0]
                if p.heading_level is not None:
                    p_en.style = doc.styles[f'Heading {min(p.heading_level, 9)}']
                run_en = p_en.add_run(p.source_text)
                
                # Right Column: Persian (RTL)
                p_fa = cell_fa.paragraphs[0]
                if p.heading_level is not None:
                    p_fa.style = doc.styles[f'Heading {min(p.heading_level, 9)}']
                add_target_runs(p_fa, p.translated_text, p.metadata)
                
                make_paragraph_rtl(p_fa)
                
        else:
            for p in document.paragraphs:
                # Add Heading or Paragraph
                if bilingual_mode == "target_only":
                    if p.heading_level is not None:
                        p_fa = doc.add_heading(level=min(p.heading_level, 9))
                    else:
                        p_fa = doc.add_paragraph()
                    
                    add_target_runs(p_fa, p.translated_text, p.metadata)
                    make_paragraph_rtl(p_fa)
                    
                elif bilingual_mode == "inline":
                    # English paragraph (LTR)
                    if p.heading_level is not None:
                        p_en = doc.add_heading(level=min(p.heading_level, 9))
                    else:
                        p_en = doc.add_paragraph()
                    p_en.add_run(p.source_text)
                    
                    # Persian paragraph (RTL)
                    if p.heading_level is not None:
                        p_fa = doc.add_heading(level=min(p.heading_level, 9))
                    else:
                        p_fa = doc.add_paragraph()
                    add_target_runs(p_fa, p.translated_text, p.metadata)
                    make_paragraph_rtl(p_fa)

        notes = document_term_notes(document)
        if notes:
            heading = doc.add_heading("یادداشت‌ها", level=1)
            make_paragraph_rtl(heading)
            for note in notes:
                note_para = doc.add_paragraph()
                note_run = note_para.add_run(
                    f"{note.get('display_number', note['number'])}. {note['original']} "
                    f"({note['transliteration']})"
                )
                make_paragraph_rtl(note_para)
                make_run_rtl(note_run)

        doc.save(str(out))
        return out
