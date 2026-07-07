"""Unit tests for Phase 5: Document Exporters Subsystem."""

from __future__ import annotations

import sys
import tempfile
import zipfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Inject fallback mocks into sys.modules if real modules are not present on the host.
# This prevents ModuleNotFoundError on import in tests/exporters.
try:
    import docx
    HAS_DOCX = not isinstance(docx, MagicMock)
except ImportError:
    HAS_DOCX = False
    mock_docx = MagicMock()
    sys.modules["docx"] = mock_docx
    sys.modules["docx.enum.text"] = MagicMock()
    sys.modules["docx.oxml"] = MagicMock()
    sys.modules["docx.oxml.ns"] = MagicMock()

try:
    import reportlab
    HAS_REPORTLAB = not isinstance(reportlab, MagicMock)
except ImportError:
    HAS_REPORTLAB = False
    mock_rl = MagicMock()
    sys.modules["reportlab"] = mock_rl
    sys.modules["reportlab.lib.pagesizes"] = MagicMock()
    sys.modules["reportlab.lib.styles"] = MagicMock()
    sys.modules["reportlab.platypus"] = MagicMock()
    sys.modules["reportlab.lib"] = MagicMock()
    sys.modules["reportlab.pdfbase"] = MagicMock()
    sys.modules["reportlab.pdfbase.ttfonts"] = MagicMock()

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_BIDI = not isinstance(arabic_reshaper, MagicMock)
except ImportError:
    HAS_BIDI = False
    sys.modules["arabic_reshaper"] = MagicMock()
    sys.modules["bidi"] = MagicMock()
    sys.modules["bidi.algorithm"] = MagicMock()

from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.txt_exporter import TxtExporter
from tarjomeh.exporters.srt_exporter import SrtExporter
from tarjomeh.exporters.docx_exporter import DocxExporter
from tarjomeh.exporters.epub_exporter import EpubExporter
from tarjomeh.exporters.pdf_exporter import PdfExporter


class TestExporters(unittest.TestCase):
    """Test all exporters subclasses of BaseExporter."""

    def setUp(self) -> None:
        self.doc = TranslatedDocument(
            title="Test Document Title",
            author="Test Author",
            paragraphs=[
                TranslatedParagraph(
                    index=0,
                    source_text="Welcome to the book.",
                    translated_text="به کتاب خوش آمدید.",
                    heading_level=1,
                ),
                TranslatedParagraph(
                    index=1,
                    source_text="This is paragraph one.",
                    translated_text="این پاراگراف اول است.",
                    heading_level=None,
                ),
            ],
        )

    def test_txt_exporter(self) -> None:
        exporter = TxtExporter()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
            temp_path = Path(tf.name)

        try:
            # 1. Target Only
            exporter.export(self.doc, temp_path, bilingual_mode="target_only")
            content = temp_path.read_text(encoding="utf-8")
            self.assertIn("# Test Document Title", content)
            self.assertIn("به کتاب خوش آمدید.", content)
            self.assertNotIn("Welcome to the book.", content)

            # 2. Inline
            exporter.export(self.doc, temp_path, bilingual_mode="inline")
            content = temp_path.read_text(encoding="utf-8")
            self.assertIn("Welcome to the book.", content)
            self.assertIn("به کتاب خوش آمدید.", content)
        finally:
            temp_path.unlink()

    def test_srt_exporter(self) -> None:
        sub_doc = TranslatedDocument(
            title="Subtitle Film",
            paragraphs=[
                TranslatedParagraph(
                    index=0,
                    source_text="Hello world.",
                    translated_text="سلام دنیا.",
                    metadata={"index": 1, "timecode": "00:00:01,000 --> 00:00:03,500"},
                )
            ],
        )
        exporter = SrtExporter()
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tf:
            temp_path = Path(tf.name)

        try:
            exporter.export(sub_doc, temp_path, bilingual_mode="target_only")
            # Read with utf-8-sig to automatically strip the BOM
            content = temp_path.read_text(encoding="utf-8-sig")
            self.assertTrue(content.startswith("1\n"))
            self.assertIn("00:00:01,000 --> 00:00:03,500", content)
            self.assertIn("سلام دنیا.", content)
        finally:
            temp_path.unlink()

    def test_epub_exporter(self) -> None:
        exporter = EpubExporter()
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tf:
            temp_path = Path(tf.name)

        try:
            exporter.export(self.doc, temp_path, bilingual_mode="side_by_side")
            
            self.assertTrue(zipfile.is_zipfile(temp_path))
            with zipfile.ZipFile(temp_path, "r") as zf:
                files = zf.namelist()
                self.assertIn("mimetype", files)
                self.assertIn("META-INF/container.xml", files)
                self.assertIn("OEBPS/content.opf", files)
                self.assertIn("OEBPS/style.css", files)
                self.assertIn("OEBPS/nav.xhtml", files)
                self.assertIn("OEBPS/chapter_1.xhtml", files)
                self.assertEqual(zf.read("mimetype"), b"application/epub+zip")
        finally:
            temp_path.unlink()

    def test_docx_exporter(self) -> None:
        exporter = DocxExporter()
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tf:
            temp_path = Path(tf.name)

        try:
            if HAS_DOCX:
                exporter.export(self.doc, temp_path, bilingual_mode="target_only")
                self.assertTrue(temp_path.exists())
                self.assertTrue(temp_path.stat().st_size > 0)
            else:
                # If mocked, check mock interaction
                from tarjomeh.exporters.docx_exporter import docx as target_docx
                exporter.export(self.doc, temp_path, bilingual_mode="target_only")
                target_docx.Document.assert_called()
        finally:
            temp_path.unlink()

    def test_pdf_exporter(self) -> None:
        exporter = PdfExporter()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            temp_path = Path(tf.name)

        try:
            if HAS_REPORTLAB and HAS_BIDI:
                reg = exporter._resolve_font_path("Vazirmatn-Regular.ttf")
                bold = exporter._resolve_font_path("Vazirmatn-Bold.ttf")
                if reg and reg.is_file() and bold and bold.is_file():
                    exporter.export(self.doc, temp_path, bilingual_mode="inline")
                    self.assertTrue(temp_path.exists())
                    self.assertTrue(temp_path.stat().st_size > 0)
                else:
                    with self.assertRaises(FileNotFoundError):
                        exporter.export(self.doc, temp_path, bilingual_mode="inline")
            else:
                # If mocked, check mock interaction
                from unittest.mock import patch
                from tarjomeh.exporters.pdf_exporter import SimpleDocTemplate
                with patch.object(exporter, "_resolve_font_path", return_value=Path("dummy.ttf")), \
                     patch("pathlib.Path.is_file", return_value=True):
                    exporter.export(self.doc, temp_path, bilingual_mode="inline")
                    SimpleDocTemplate.assert_called()
        finally:
            if temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    unittest.main()
