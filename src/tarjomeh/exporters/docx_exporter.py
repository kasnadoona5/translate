"""DOCX document exporter for Tarjomeh."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from tarjomeh.exporters.base import BaseExporter, TranslatedDocument, BilingualMode
from tarjomeh.exporters.term_notes import document_term_notes, paragraph_note_parts

logger = logging.getLogger(__name__)

try:
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.shared import Cm, Pt
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

        for section in doc.sections:
            section.top_margin = Cm(2.2)
            section.bottom_margin = Cm(2.2)
            section.left_margin = Cm(2.2)
            section.right_margin = Cm(2.2)

        normal_style = doc.styles["Normal"]
        normal_style.font.name = "Vazirmatn"
        normal_style.font.size = Pt(11)
        normal_style.paragraph_format.space_after = Pt(6)
        normal_style.paragraph_format.line_spacing = 1.15

        latin_parenthetical = re.compile(r"(\([^()\n]*[A-Za-z][^()\n]*\))")

        def set_on_off(parent, tag: str, enabled: bool) -> None:
            element = parent.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                parent.append(element)
            element.set(qn("w:val"), "1" if enabled else "0")

        def set_run_fonts(r_obj, *, rtl: bool) -> None:
            rPr = r_obj._r.get_or_add_rPr()
            set_on_off(rPr, "w:rtl", rtl)
            rFonts = rPr.find(qn("w:rFonts"))
            if rFonts is None:
                rFonts = OxmlElement("w:rFonts")
                rPr.append(rFonts)
            if rtl:
                rFonts.set(qn("w:cs"), "Vazirmatn")
                rFonts.set(qn("w:eastAsia"), "Vazirmatn")
                r_obj.font.name = "Vazirmatn"
            else:
                rFonts.set(qn("w:ascii"), "Times New Roman")
                rFonts.set(qn("w:hAnsi"), "Times New Roman")
                r_obj.font.name = "Times New Roman"
            r_obj.font.size = Pt(11)

        def make_paragraph_rtl(p_obj, *, heading: bool = False) -> None:
            p_obj.alignment = (
                WD_ALIGN_PARAGRAPH.RIGHT
                if heading else WD_ALIGN_PARAGRAPH.JUSTIFY
            )
            pPr = p_obj._element.get_or_add_pPr()
            set_on_off(pPr, "w:bidi", True)
            fmt = p_obj.paragraph_format
            fmt.line_spacing = 1.15
            fmt.space_after = Pt(6)
            if heading:
                fmt.keep_with_next = True
                fmt.space_before = Pt(10)
                fmt.space_after = Pt(4)
            else:
                fmt.first_line_indent = Cm(0.5)
                fmt.widow_control = True

        def add_target_runs(p_obj, text: str, metadata: dict) -> None:
            for segment, ref in paragraph_note_parts(text, metadata):
                if segment:
                    for part in latin_parenthetical.split(segment):
                        if not part:
                            continue
                        run = p_obj.add_run(part)
                        set_run_fonts(
                            run,
                            rtl=not bool(latin_parenthetical.fullmatch(part)),
                        )
                if ref is not None:
                    marker = p_obj.add_run(
                        str(ref.get("display_number", ref["number"]))
                    )
                    marker.font.superscript = True
                    set_run_fonts(marker, rtl=True)

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
                
                make_paragraph_rtl(p_fa, heading=p.heading_level is not None)
                
        else:
            for p in document.paragraphs:
                # Add Heading or Paragraph
                if bilingual_mode == "target_only":
                    if p.heading_level is not None:
                        p_fa = doc.add_heading(level=min(p.heading_level, 9))
                    else:
                        p_fa = doc.add_paragraph()
                    
                    add_target_runs(p_fa, p.translated_text, p.metadata)
                    make_paragraph_rtl(p_fa, heading=p.heading_level is not None)
                    
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
                    make_paragraph_rtl(p_fa, heading=p.heading_level is not None)

        notes = document_term_notes(document)
        if notes:
            heading = doc.add_heading("یادداشت‌ها", level=1)
            make_paragraph_rtl(heading, heading=True)
            for note in notes:
                note_para = doc.add_paragraph()
                note_run = note_para.add_run(
                    f"{note.get('display_number', note['number'])}. {note['original']} "
                    f"({note['transliteration']})"
                )
                make_paragraph_rtl(note_para)
                set_run_fonts(note_run, rtl=True)

        doc.save(str(out))
        return out
