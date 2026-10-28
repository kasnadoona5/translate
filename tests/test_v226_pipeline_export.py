"""New parses and saved-version parses keep the same admission/export gates."""
import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pymupdf
import pytest

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import TranslationPipeline
from tarjomeh.core.render_identity import RenderChangeLedger, compare_docx_to_rendered
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.docx_exporter import DocxExporter
from tarjomeh.parsers.pdf_parser import PyMuPDFParser


def source_pdf(tmp_path, toc):
    path = tmp_path / 'source.pdf'
    with pymupdf.open() as document:
        page = document.new_page(width=432, height=648)
        page.insert_text((210, 95), '2', fontsize=22)
        page.insert_text((80, 140), 'A Chapter', fontsize=18)
        page.insert_text((55, 290), 'This is ordinary body text.', fontsize=10.5)
        page.insert_text((55, 550), 'This is continuing body text.', fontsize=10.5)
        if toc:
            document.set_toc([[1, '2 A Chapter', 1]])
        document.save(path)
    return path


@pytest.mark.parametrize('toc', [False, True])
def test_real_pdf_and_explicit_saved_version_routing(tmp_path, toc):
    path = source_pdf(tmp_path, toc)
    pipeline = object.__new__(TranslationPipeline)
    pipeline.config = TarjomehConfig()
    pipeline.llm_client = SimpleNamespace(count_tokens=lambda text: len(text.split()))
    old, _ = pipeline._parse_and_chunk(path, structure_version=4)
    new, _ = pipeline._parse_and_chunk(path)
    assert old.all_paragraphs[-1].metadata['is_footnote'] is True
    assert new.all_paragraphs[-1].metadata.get('is_footnote') is not True
    assert [p.text for p in old.all_paragraphs] == [p.text for p in new.all_paragraphs]
    for version in (1, 2, 3, 4):
        explicit, _ = pipeline._parse_and_chunk(path, structure_version=version)
        assert explicit == PyMuPDFParser(structure_version=version).parse(path)


@pytest.mark.parametrize('mode', ['target_only', 'inline', 'side_by_side'])
@pytest.mark.parametrize('tamper', [False, True])
def test_new_parser_metadata_neither_blocks_export_nor_waives_identity(tmp_path, mode, tamper):
    parsed = PyMuPDFParser().parse(source_pdf(tmp_path, True))
    paragraphs = []
    for i, p in enumerate(parsed.all_paragraphs):
        target = ('\u06f2 \u06cc\u06a9 \u0641\u0635\u0644' if p.heading_level else
                  '\u0627\u06cc\u0646 \u0645\u062a\u0646 \u0645\u0639\u0645\u0648\u0644\u06cc \u0627\u0633\u062a.')
        paragraphs.append(TranslatedParagraph(i, p.text, target, heading_level=p.heading_level,
                                              metadata=dict(p.metadata)))
    document = TranslatedDocument(paragraphs=paragraphs)
    before = copy.deepcopy(document)
    ledger = RenderChangeLedger(document)
    pipeline = SimpleNamespace(config=SimpleNamespace(output=SimpleNamespace(bilingual_mode=mode)),
                               db=MagicMock())
    path = tmp_path / 'checkpoint.docx'
    if tamper:
        document.paragraphs[-1].translated_text += ' UNEXPLAINED'
        with pytest.raises(RuntimeError, match='Export blocked'):
            TranslationPipeline._export_with_render_identity(
                pipeline, 'offline-v226', document, DocxExporter(), path, ledger, 'docx')
        assert not path.exists()
    else:
        TranslationPipeline._export_with_render_identity(
            pipeline, 'offline-v226', document, DocxExporter(), path, ledger, 'docx')
        assert compare_docx_to_rendered(document, path, mode)['passed']
        assert document == before


def test_faithful_body_admission_does_not_add_a_repair_call(tmp_path):
    from test_v20812_pipeline_export import run_chunk

    parsed = PyMuPDFParser().parse(source_pdf(tmp_path, True))
    paragraph = parsed.all_paragraphs[-1]
    target = '\u0627\u06cc\u0646 \u0645\u062a\u0646 \u0627\u062f\u0627\u0645\u0647 \u062f\u0627\u0631\u062f.'
    pipeline, kwargs = run_chunk(paragraph.text, target)
    kwargs['chunk'].metadata['structural_roles'] = [paragraph.metadata['structure_role']]
    assert pipeline._translate_single_chunk(**kwargs)
    assert [call.kwargs.get('_operation') for call in pipeline.llm_client.complete.call_args_list] == ['translation']
