from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tarjomeh.web.app import create_app
from tarjomeh.runtime import _v229_behavior_probes


@pytest.mark.parametrize('scope', ['complete', 'saved_prefix', 'bad', None, [], {}, False])
def test_export_scope_is_explicit_and_backwards_compatible(monkeypatch, scope):
    monkeypatch.setenv('UI_SECRET_TOKEN', 'test-token')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-only-not-used')
    client = create_app().test_client()
    db = MagicMock()
    db.get_job.return_value = dict(id='route-job', input_path='source.pdf', status='paused', config={})
    pipeline = MagicMock()
    pipeline.export_completed_job.return_value = Path('saved.docx')
    with patch('tarjomeh.jobs.database.JobDatabase', return_value=db), patch('tarjomeh.core.pipeline.TranslationPipeline', return_value=pipeline):
        response = client.post('/api/jobs/route-job/export', headers={'Authorization': 'Bearer test-token'},
                               json=dict(format='docx', scope=scope))
    if scope not in ('complete', 'saved_prefix'):
        assert response.status_code == 400
        pipeline.export_completed_job.assert_not_called()
    else:
        assert response.status_code == 200
        kwargs = pipeline.export_completed_job.call_args.kwargs
        assert kwargs.get('saved_prefix', False) == (scope == 'saved_prefix')
        assert ('PARTIAL_REVIEW' in str(pipeline.export_completed_job.call_args.args[1])) == (scope == 'saved_prefix')
        pipeline.close.assert_called_once()


def test_runtime_probe_is_pure_and_detects_disabled_guard(monkeypatch):
    import socket
    import tarjomeh.jobs.saved_prefix as prefix
    def forbidden(*a, **kw):
        raise AssertionError('Unexpected I/O')
    monkeypatch.setattr(Path, 'open', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    assert _v229_behavior_probes() == {'source_bound_saved_prefix_export': True}
    monkeypatch.setattr(prefix, 'saved_prefix_count', lambda *a: 2)
    assert _v229_behavior_probes() == {'source_bound_saved_prefix_export': False}
