"""R235: the exported document must be the state proved by its render ledger."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tarjomeh.core.pipeline import TranslationPipeline
from tarjomeh.core.render_identity import RenderChangeLedger
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.docx_exporter import DocxExporter


MODES = ('target_only', 'inline', 'side_by_side')


def document():
    return TranslatedDocument(title='Synthetic source', author='Author', paragraphs=[
        TranslatedParagraph(0, 'The argument is provisional.',
                            '\u0627\u06cc\u0646 \u0627\u0633\u062a\u062f\u0644\u0627\u0644 \u0645\u0648\u0642\u062a\u06cc \u0627\u0633\u062a.'),
        TranslatedParagraph(1, 'The second claim is disputed.',
                            '\u0627\u062f\u0639\u0627\u06cc \u062f\u0648\u0645 \u0645\u062d\u0644 \u0627\u062e\u062a\u0644\u0627\u0641 \u0627\u0633\u062a.'),
    ])


def pipeline(mode):
    artifacts = {}
    db = MagicMock()
    db.save_job_artifact.side_effect = lambda job, name, value: artifacts.update({name: value})
    return SimpleNamespace(db=db, config=SimpleNamespace(output=SimpleNamespace(bilingual_mode=mode))), artifacts


def export(instance, doc, ledger, path, exporter=None, fmt='docx'):
    TranslationPipeline._export_with_render_identity(
        instance, 'synthetic-binding', doc, exporter or DocxExporter(), path, ledger, fmt)


@pytest.mark.parametrize('mode', MODES)
def test_changed_detached_document_cannot_replace_previous_output(tmp_path, mode):
    original = document()
    ledger = RenderChangeLedger(original)
    detached = deepcopy(original)
    detached.paragraphs[0].translated_text += ' UNEXPLAINED'
    instance, artifacts = pipeline(mode)
    path = tmp_path / 'previous.docx'
    path.write_bytes(b'previous verified output')
    exporter = MagicMock(wraps=DocxExporter())
    with pytest.raises(RuntimeError, match='canonical-to-render proof'):
        export(instance, detached, ledger, path, exporter)
    assert path.read_bytes() == b'previous verified output'
    assert not list(tmp_path.glob('.*.tmp.docx'))
    assert not artifacts['render_identity_audit']['passed']
    assert artifacts['render_identity_audit']['render']['export_document_matches_ledger'] is False
    exporter.export.assert_not_called()
    assert ledger.audit()['passed']


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('change', [
    'source', 'index', 'order', 'heading', 'paragraph_metadata', 'note_geometry',
    'document_metadata', 'title', 'author', 'language', 'drop', 'append',
])
def test_binding_includes_order_source_identity_and_render_metadata(tmp_path, mode, change):
    original = document()
    ledger = RenderChangeLedger(original)
    detached = deepcopy(original)
    if change == 'source':
        detached.paragraphs[0].source_text += ' A different source.'
    elif change == 'index':
        detached.paragraphs[0].index = 50
    elif change == 'order':
        detached.paragraphs.reverse()
    elif change == 'heading':
        detached.paragraphs[0].heading_level = 1
    elif change == 'paragraph_metadata':
        detached.paragraphs[0].metadata['structure_role'] = 'footnote'
    elif change == 'note_geometry':
        detached.paragraphs[0].metadata['superscript_markers'] = [{'text': '4', 'offset': 1}]
    elif change == 'document_metadata':
        detached.metadata['saved_prefix_export'] = {'finished_chunks': 1, 'total_selected_chunks': 3}
    elif change == 'title':
        detached.title = 'Another title'
    elif change == 'author':
        detached.author = 'Another author'
    elif change == 'language':
        detached.language = 'en'
    elif change == 'drop':
        detached.paragraphs.pop()
    elif change == 'append':
        detached.paragraphs.append(deepcopy(detached.paragraphs[0]))
    instance, artifacts = pipeline(mode)
    path = tmp_path / 'not-published.docx'
    with pytest.raises(RuntimeError, match='canonical-to-render proof'):
        export(instance, detached, ledger, path)
    assert not path.exists() and not list(tmp_path.glob('.*.tmp.docx'))
    assert artifacts['render_identity_audit']['render']['export_document_matches_ledger'] is False


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('use_copy', [False, True])
def test_unchanged_same_or_equal_copy_still_exports(tmp_path, mode, use_copy):
    original = document()
    ledger = RenderChangeLedger(original)
    instance, artifacts = pipeline(mode)
    path = tmp_path / 'normal.docx'
    export(instance, deepcopy(original) if use_copy else original, ledger, path)
    assert path.is_file() and artifacts['render_identity_audit']['passed']
    assert artifacts['render_identity_audit']['render']['export_document_matches_ledger']


@pytest.mark.parametrize('mode', MODES)
def test_unchanged_copy_cannot_bypass_unproved_ledger_change(tmp_path, mode):
    original = document()
    ledger = RenderChangeLedger(original)
    original.paragraphs[0].translated_text += ' UNEXPLAINED'
    instance, artifacts = pipeline(mode)
    with pytest.raises(RuntimeError, match='canonical-to-render proof'):
        export(instance, deepcopy(original), ledger, tmp_path / 'bad.docx')
    assert not artifacts['render_identity_audit']['passed']
    assert artifacts['render_identity_audit']['render']['export_document_matches_ledger']


@pytest.mark.parametrize('mode', MODES)
def test_exporter_mutation_of_detached_metadata_is_rechecked_before_publication(tmp_path, mode):
    original = document()
    detached = deepcopy(original)
    ledger = RenderChangeLedger(original)
    instance, artifacts = pipeline(mode)
    real = DocxExporter()

    def mutate_after_write(**kwargs):
        result = real.export(**kwargs)
        kwargs['document'].title = 'Unexpected exporter mutation'
        return result

    exporter = MagicMock()
    exporter.export.side_effect = mutate_after_write
    path = tmp_path / 'previous.docx'
    path.write_bytes(b'previous verified output')
    with pytest.raises(RuntimeError, match='canonical-to-render proof'):
        export(instance, detached, ledger, path, exporter)
    assert path.read_bytes() == b'previous verified output'
    assert not list(tmp_path.glob('.*.tmp.docx'))
    assert not artifacts['render_identity_audit']['passed']


def test_other_export_formats_require_the_same_binding(tmp_path):
    original = document()
    ledger = RenderChangeLedger(original)
    detached = deepcopy(original)
    detached.paragraphs[0].translated_text += ' UNEXPLAINED'
    instance, artifacts = pipeline('target_only')
    exporter = MagicMock()
    with pytest.raises(RuntimeError, match='canonical-to-render proof'):
        export(instance, detached, ledger, tmp_path / 'bad.txt', exporter, fmt='srt')
    exporter.export.assert_not_called()
    assert artifacts['render_identity_audit']['passed'] is False


@pytest.mark.parametrize('mode', MODES)
def test_equal_copy_after_a_proved_render_step_remains_eligible(tmp_path, mode):
    original = document()
    clean = original.paragraphs[0].translated_text
    original.paragraphs[0].translated_text = '[[p1]]' + clean + '[[/p1]]'
    ledger = RenderChangeLedger(original)
    ledger.step('protocol_sanitation', lambda doc: setattr(doc.paragraphs[0], 'translated_text', clean))
    assert ledger.audit()['passed'] and ledger.audit()['change_count'] == 1
    instance, artifacts = pipeline(mode)
    export(instance, deepcopy(original), ledger, tmp_path / 'proved.docx')
    assert artifacts['render_identity_audit']['passed']


def test_binding_audit_is_read_only_and_legacy_audit_shape_is_unchanged():
    original = document()
    ledger = RenderChangeLedger(original)
    before = deepcopy(original)
    old_report = ledger.audit()
    assert 'export_document_matches_ledger' not in old_report
    assert ledger.audit(document=deepcopy(original)) == {**old_report, 'export_document_matches_ledger': True}
    assert original == before and ledger.audit() == old_report


def test_pure_runtime_probe_exercises_the_binding_producer(monkeypatch):
    from tarjomeh.runtime import _v2212_behavior_probes

    assert _v2212_behavior_probes() == {'export_document_ledger_binding': True}
    old_audit = RenderChangeLedger.audit
    monkeypatch.setattr(RenderChangeLedger, 'audit', lambda self, **kwargs: old_audit(self))
    assert _v2212_behavior_probes() == {'export_document_ledger_binding': False}


def test_new_runtime_probe_performs_no_file_or_network_io(monkeypatch):
    from pathlib import Path
    import socket
    from tarjomeh.runtime import _v2212_behavior_probes

    def forbidden(*args, **kwargs):
        raise AssertionError('Pure binding probe attempted I/O')

    monkeypatch.setattr(Path, 'open', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    assert _v2212_behavior_probes() == {'export_document_ledger_binding': True}
