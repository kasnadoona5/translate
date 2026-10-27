"""Deterministic fixes pass real canonicalization/export, never a bypass."""
import copy
import hashlib
from unittest.mock import MagicMock

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import TranslationPipeline
from tarjomeh.core.render_identity import RenderChangeLedger, compare_docx_to_rendered
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.docx_exporter import DocxExporter
from tarjomeh.quality.integrity import unexpected_latin_prose


@pytest.mark.parametrize("mode", ["target_only", "inline", "side_by_side"])
@pytest.mark.parametrize("tamper", [False, True])
def test_numeral_and_etymon_survive_canonicalization_and_export(tmp_path, mode, tamper):
    source = "The term derives from Latin civitas. The researcher identifies three ways."
    target = "این اصطلاح از civitas لاتین گرفته شده است. پژوهشگر سه راه را شناسایی می‌کند."
    pipeline = object.__new__(TranslationPipeline)
    pipeline.config = TarjomehConfig()
    pipeline.db = MagicMock()
    pipeline.llm_client = MagicMock()
    canonical = pipeline._canonicalize_final_translation("offline", 0, Chunk(0, source, "", ""), target)
    assert "سه راه" in canonical and "سه‌راه" not in canonical
    assert not unexpected_latin_prose(source, canonical)
    assert pipeline._canonicalize_final_translation(
        "offline", 0, Chunk(0, source, "", ""), canonical) == canonical
    pipeline.llm_client.assert_not_called()
    assert not pipeline.llm_client.mock_calls
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(0, source, canonical)])
    before = copy.deepcopy(document)
    canonical_hash = hashlib.sha256(canonical.encode()).hexdigest()
    ledger = RenderChangeLedger(document)
    if tamper:
        document.paragraphs[0].translated_text += " تحریف"
    pipeline.config.output.bilingual_mode = mode
    output = tmp_path / "review.docx"
    if tamper:
        with pytest.raises(RuntimeError, match="Export blocked"):
            pipeline._export_with_render_identity("offline", document, DocxExporter(), output, ledger, "docx")
        assert not output.exists()
    else:
        pipeline._export_with_render_identity("offline", document, DocxExporter(), output, ledger, "docx")
        assert compare_docx_to_rendered(document, output, mode)["passed"]
        assert document == before
        assert hashlib.sha256(document.paragraphs[0].translated_text.encode()).hexdigest() == canonical_hash
