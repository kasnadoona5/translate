"""Exercise numeric boundaries through actual admission and DOCX publication."""

import copy
import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_v20812_numeric_boundaries import SOURCE, TARGET

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import SourceStructureAdmissionError, TranslationPipeline
from tarjomeh.core.render_identity import RenderChangeLedger
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.docx_exporter import DocxExporter
from tarjomeh.memory.proper_nouns import ProperNouns
from tarjomeh.persian.typography import PersianTypographer


def run_chunk(source, target):
    config = TarjomehConfig()
    config.translation.enable_web_context = False
    config.translation.enable_back_translation = False
    config.translation.enable_critique = False
    config.glossary.enable_compliance_check = False
    config.glossary.enable_auto_extraction = False
    pipeline = object.__new__(TranslationPipeline)
    pipeline.config = config
    pipeline.db = MagicMock()
    pipeline.db.get_job.return_value = {"status": "running"}
    pipeline.db.get_chunk_events.return_value = []
    pipeline.db.get_job_artifact.return_value = {}
    pipeline.llm_client = MagicMock()
    pipeline.llm_client.complete.return_value = target
    memory = MagicMock()
    memory.proper_nouns = ProperNouns()
    context = memory.get_context_for_chunk.return_value
    for field in ("style_profile", "proper_nouns", "long_term", "short_term", "bilingual_summary"):
        setattr(context, field, "")
    context.format.return_value = ""
    glossary = MagicMock()
    glossary.find_terms.return_value = []
    glossary.format_for_prompt.return_value = ""
    chunk = Chunk(
        0,
        source,
        "Tables",
        "",
        metadata={
            "paragraph_indices": [10],
            "structural_roles": ["table"],
        },
    )
    kwargs = dict(
        idx=0,
        chunk=chunk,
        memory_manager=memory,
        web_searcher=MagicMock(),
        glossary_manager=glossary,
        compliance_checker=MagicMock(),
        critique_tool=MagicMock(),
        refiner_tool=MagicMock(),
        back_translator=MagicMock(),
        translations={},
        job_id="synthetic-numeric",
    )
    return pipeline, kwargs


def test_faithful_table_reaches_real_final_admission_without_repair_calls():
    pipeline, kwargs = run_chunk(SOURCE, TARGET)
    result = pipeline._translate_single_chunk(**kwargs)
    expected = PersianTypographer(pipeline.config.to_dict().get("persian")).process(TARGET)
    assert result == expected
    operations = [
        call.kwargs.get("_operation") for call in pipeline.llm_client.complete.call_args_list
    ]
    assert operations == ["translation"]
    events = [(call.args[2], call.args[3]) for call in pipeline.db.log_chunk_event.call_args_list]
    assert not any(kind == "integrity_final_failed" for kind, _ in events)
    repairs = [payload for kind, payload in events if kind == "targeted_language_repair"]
    assert repairs and not any(item["attempted"] for item in repairs)
    assert any(kind == "candidate_portfolio_selected" for kind, _ in events)


def test_real_wrong_count_still_cannot_commit_after_unsuccessful_bounded_repair():
    pipeline, kwargs = run_chunk("2.2 Three elements", "2.2 \u062f\u0648 \u0639\u0646\u0635\u0631")
    with pytest.raises(SourceStructureAdmissionError):
        pipeline._translate_single_chunk(**kwargs)
    assert any(
        call.args[2] == "integrity_final_failed"
        for call in pipeline.db.log_chunk_event.call_args_list
    )


@pytest.mark.parametrize("mode", ["target_only", "inline", "side_by_side"])
def test_faithful_table_has_unchanged_canonical_hash_and_real_docx_readback(tmp_path, mode):
    doc = TranslatedDocument(paragraphs=[TranslatedParagraph(0, SOURCE, TARGET)])
    before = copy.deepcopy(doc)
    before_hash = hashlib.sha256(TARGET.encode()).hexdigest()
    ledger = RenderChangeLedger(doc)
    artifacts = {}
    pipeline = SimpleNamespace(
        config=SimpleNamespace(output=SimpleNamespace(bilingual_mode=mode)),
        db=SimpleNamespace(
            save_job_artifact=lambda job, key, value: artifacts.update({key: value})
        ),
    )
    output = tmp_path / "checkpoint.docx"
    TranslationPipeline._export_with_render_identity(
        pipeline, "synthetic", doc, DocxExporter(), output, ledger, "docx"
    )
    assert output.is_file() and artifacts["render_identity_audit"]["passed"]
    assert doc == before
    assert hashlib.sha256(doc.paragraphs[0].translated_text.encode()).hexdigest() == before_hash


@pytest.mark.parametrize("mode", ["target_only", "inline", "side_by_side"])
def test_unexplained_render_change_still_prevents_publication(tmp_path, mode):
    doc = TranslatedDocument(paragraphs=[TranslatedParagraph(0, SOURCE, TARGET)])
    ledger = RenderChangeLedger(doc)
    doc.paragraphs[0].translated_text += " \u0627\u0636\u0627\u0641\u0647"
    pipeline = SimpleNamespace(
        config=SimpleNamespace(output=SimpleNamespace(bilingual_mode=mode)),
        db=MagicMock(),
    )
    output = tmp_path / "must-not-publish.docx"
    with pytest.raises(RuntimeError, match="Export blocked"):
        TranslationPipeline._export_with_render_identity(
            pipeline, "synthetic", doc, DocxExporter(), output, ledger, "docx"
        )
    assert not output.exists()


def test_probe_is_pure_and_cannot_pass_if_count_detector_is_disabled(monkeypatch):
    from pathlib import Path

    import tarjomeh.quality.structure_audit as audit
    from tarjomeh.runtime import _v20812_behavior_probes

    def forbidden(*args, **kwargs):
        raise AssertionError("Probe attempted file I/O")

    monkeypatch.setattr(Path, "open", forbidden)
    assert _v20812_behavior_probes()["numeric_announcement_boundaries"]
    monkeypatch.setattr(audit, "announced_count_evidence", lambda text: [])
    assert not _v20812_behavior_probes()["numeric_announcement_boundaries"]
