"""PDF document exporter for Tarjomeh."""

from __future__ import annotations

import logging
from pathlib import Path
from tarjomeh.exporters.base import BaseExporter, TranslatedDocument, BilingualMode

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class PdfExporter(BaseExporter):
    """PDF exporter.

    Generates publication-quality RTL PDFs using ReportLab.
    """

    def export(
        self,
        document: TranslatedDocument,
        output_path: str | Path,
        bilingual_mode: BilingualMode = "target_only",
    ) -> Path:
        if not HAS_REPORTLAB:
            raise ImportError(
                "reportlab is required for PDF export. "
                "Install it with:  pip install reportlab"
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # 1. Font registration
        reg_path = self._resolve_font_path("Vazirmatn-Regular.ttf")
        bold_path = self._resolve_font_path("Vazirmatn-Bold.ttf")

        if not reg_path or not reg_path.is_file() or not bold_path or not bold_path.is_file():
            raise FileNotFoundError(
                "Vazirmatn font files (Vazirmatn-Regular.ttf and Vazirmatn-Bold.ttf) are required to render Persian text in PDF. "
                "Please place them in src/tarjomeh/persian/fonts/."
            )

        try:
            pdfmetrics.registerFont(TTFont("Vazirmatn", str(reg_path)))
            pdfmetrics.registerFont(TTFont("Vazirmatn-Bold", str(bold_path)))
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to register Vazirmatn font: {e}. "
                "Ensure that valid TrueType font files are placed in src/tarjomeh/persian/fonts/."
            ) from e

        font_family = "Vazirmatn"
        bold_font_family = "Vazirmatn-Bold"

        # 2. Build Document styles
        styles = getSampleStyleSheet()
        normal_style = styles["Normal"]

        title_style = ParagraphStyle(
            "DocTitle",
            parent=normal_style,
            fontName=bold_font_family,
            fontSize=24,
            leading=28,
            alignment=1,  # Center
            spaceAfter=20
        )

        heading_style_fa = ParagraphStyle(
            "HeadingStyleFa",
            parent=normal_style,
            fontName=bold_font_family,
            fontSize=16,
            leading=20,
            alignment=2,  # Right
            spaceBefore=14,
            spaceAfter=8
        )

        heading_style_en = ParagraphStyle(
            "HeadingStyleEn",
            parent=normal_style,
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            alignment=0,  # Left
            spaceBefore=14,
            spaceAfter=8
        )

        body_style_fa = ParagraphStyle(
            "BodyStyleFa",
            parent=normal_style,
            fontName=font_family,
            fontSize=11,
            leading=16,
            alignment=2  # Right
        )

        body_style_en = ParagraphStyle(
            "BodyStyleEn",
            parent=normal_style,
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=0  # Left
        )

        # 3. Assemble document story flowables
        doc_template = SimpleDocTemplate(
            str(out),
            pagesize=letter,
            rightMargin=54,
            leftMargin=54,
            topMargin=54,
            bottomMargin=54
        )
        story = []

        # Renders the Document Title
        shaped_title = self._shape_persian_text(document.title)
        story.append(Paragraph(shaped_title, title_style))
        story.append(Spacer(1, 10))

        if bilingual_mode == "side_by_side":
            table_data = []
            for p in document.paragraphs:
                # Add left cell (English LTR) and right cell (Persian RTL)
                style_en = heading_style_en if p.heading_level is not None else body_style_en
                style_fa = heading_style_fa if p.heading_level is not None else body_style_fa
                
                # Shape Persian translation
                shaped_text = self._shape_persian_text(p.translated_text)
                
                p_en = Paragraph(p.source_text, style_en)
                p_fa = Paragraph(shaped_text, style_fa)
                
                table_data.append([p_en, p_fa])
            
            # Letter page printable width is 612 - 54 * 2 = 504 points
            col_width = 504 / 2
            table = Table(table_data, colWidths=[col_width, col_width])
            table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(table)
            
        else:
            for p in document.paragraphs:
                if bilingual_mode == "target_only":
                    style = heading_style_fa if p.heading_level is not None else body_style_fa
                    shaped_text = self._shape_persian_text(p.translated_text)
                    story.append(Paragraph(shaped_text, style))
                    story.append(Spacer(1, 10))
                    
                elif bilingual_mode == "inline":
                    # English paragraph (LTR)
                    style_en = heading_style_en if p.heading_level is not None else body_style_en
                    story.append(Paragraph(p.source_text, style_en))
                    
                    # Persian paragraph (RTL)
                    style_fa = heading_style_fa if p.heading_level is not None else body_style_fa
                    shaped_text = self._shape_persian_text(p.translated_text)
                    story.append(Paragraph(shaped_text, style_fa))
                    story.append(Spacer(1, 12))

        doc_template.build(story)
        return out
