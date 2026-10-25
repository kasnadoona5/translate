"""Exercise the new evidence through real admission and DOCX publication."""

import hashlib
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_v221_structure import DIGIT_SCRIPTS, chapter_pair, saved_pair
from test_v20812_pipeline_export import run_chunk

from tarjomeh.core.pipeline import SourceStructureAdmissionError, TranslationPipeline
from tarjomeh.core.render_identity import RenderChangeLedger
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.docx_exporter import DocxExporter
from tarjomeh.persian.typography import PersianTypographer


@pytest.mark.parametrize("count", [9, 10])
@pytest.mark.parametrize("role", ["body", "table"])
@pytest.mark.parametrize("digits", DIGIT_SCRIPTS)
def test_faithful_chapter_admission_has_no_structure_repair(count, role, digits):
    source, target = chapter_pair(count, digits=digits)
    pipeline, kwargs = run_chunk(source, target)
    kwargs["chunk"].metadata["structural_roles"] = [role]
    result = pipeline._translate_single_chunk(**kwargs)
    expected = PersianTypographer(pipeline.config.to_dict().get("persian")).process(target)
    assert result == expected
    operations = [
        call.kwargs.get("_operation") for call in pipeline.llm_client.complete.call_args_list
    ]
    assert operations == ["translation"]


def test_wrong_chapter_quantity_still_blocks_after_bounded_repair():
    source, target = chapter_pair()
    target = target.replace(
        "\u062f\u0647\u200c\u0641\u0635\u0644\u06cc",
        "\u062f\u0648\u0627\u0632\u062f\u0647\u200c\u0641\u0635\u0644\u06cc",
    )
    pipeline, kwargs = run_chunk(source, target)
    kwargs["chunk"].metadata["structural_roles"] = ["body"]
    with pytest.raises(SourceStructureAdmissionError):
        pipeline._translate_single_chunk(**kwargs)


def test_mixed_table_body_chunk_does_not_bind_a_decimal_label_to_a_count():
    source, target = chapter_pair()
    source = "2.2 Aspects of dimensions.\n\n" + source
    target = "2.2 \u062c\u0646\u0628\u0647 \u0627\u0628\u0639\u0627\u062f.\n\n" + target
    pipeline, kwargs = run_chunk(source, target)
    kwargs["chunk"].metadata.update(paragraph_indices=[10, 11], structural_roles=["table", "body"])
    result = pipeline._translate_single_chunk(**kwargs)
    assert result
    operations = [
        call.kwargs.get("_operation") for call in pipeline.llm_client.complete.call_args_list
    ]
    assert operations == ["translation"]


@pytest.mark.parametrize("mode", ["target_only", "inline", "side_by_side"])
@pytest.mark.parametrize("saved", [False, True])
def test_docx_readback_and_canonical_hashes_stay_exact(tmp_path, mode, saved):
    source, target = saved_pair() if saved else chapter_pair()
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(0, source, target)])
    before = deepcopy(document)
    before_hash = hashlib.sha256(target.encode()).hexdigest()
    ledger = RenderChangeLedger(document)
    artifacts = {}
    pipeline = SimpleNamespace(
        config=SimpleNamespace(output=SimpleNamespace(bilingual_mode=mode)),
        db=SimpleNamespace(
            save_job_artifact=lambda job, key, value: artifacts.update({key: value})
        ),
    )
    output = tmp_path / "checkpoint.docx"
    TranslationPipeline._export_with_render_identity(
        pipeline,
        "synthetic",
        document,
        DocxExporter(),
        output,
        ledger,
        "docx",
    )
    assert output.is_file() and artifacts["render_identity_audit"]["passed"]
    assert document == before
    actual_hash = hashlib.sha256(document.paragraphs[0].translated_text.encode()).hexdigest()
    assert actual_hash == before_hash


@pytest.mark.parametrize("mode", ["target_only", "inline", "side_by_side"])
def test_unexplained_change_still_cannot_publish(tmp_path, mode):
    source, target = chapter_pair()
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(0, source, target)])
    ledger = RenderChangeLedger(document)
    document.paragraphs[0].translated_text += " \u0627\u0636\u0627\u0641\u0647"
    pipeline = SimpleNamespace(
        config=SimpleNamespace(output=SimpleNamespace(bilingual_mode=mode)),
        db=MagicMock(),
    )
    output = tmp_path / "must-not-publish.docx"
    with pytest.raises(RuntimeError, match="Export blocked"):
        TranslationPipeline._export_with_render_identity(
            pipeline,
            "synthetic",
            document,
            DocxExporter(),
            output,
            ledger,
            "docx",
        )
    assert not output.exists()
