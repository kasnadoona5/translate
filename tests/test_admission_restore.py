"""Tests for Option A - the restore contract, and crash isolation.

Two separate problems, both of which let damage reach the reader:

* Four gate call sites passed no ``previous``, so the gate could only *log* a
  rejection - it had no valid text to fall back to. The final gate is the one
  that matters, because it is the last thing between a candidate and the export.
* ``repair_corruption`` and ``audit_payload`` sat unguarded in
  ``_translate_single_chunk``. Because ``pause_on_sequential_error`` defaults to
  True and academic mode forces one worker, an exception from either one would
  pause the whole job on the FIRST affected chunk - and a report-only helper must
  never be able to do that.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import PipelineResult, TranslationPipeline


def _fast_config() -> TarjomehConfig:
    config = TarjomehConfig()
    config.llm.openrouter.api_keys = ["sk-test"]
    config.translation.mode = "fast"
    config.translation.enable_critique = False
    config.translation.enable_web_context = False
    config.translation.enable_back_translation = False
    config.glossary.enable_auto_extraction = False
    config.output.format = "txt"
    return config


def _run_pipeline(tmp_path: Path):
    """Run a one-chunk translation with the database and LLM mocked out."""
    source = tmp_path / "chapter.txt"
    source.write_text(
        "Chapter 1: The State\nA paragraph about institutional power.",
        encoding="utf-8",
    )
    with mock_patch("tarjomeh.core.pipeline.JobDatabase") as db_cls, \
            mock_patch("tarjomeh.core.pipeline.LLMClient") as llm_cls:
        db = db_cls.return_value
        db.get_job.return_value = None
        db.get_chunk_summary.return_value = {
            "total": 0, "completed": 0, "pending": 0, "errors": 0
        }
        db.get_chunks.return_value = []
        db.get_chunk_events.return_value = []
        db.get_job_artifact.return_value = None
        llm = llm_cls.return_value
        llm.count_tokens.return_value = 10
        llm.complete.return_value = "ترجمه‌ای درست از متن مبدأ."

        pipeline = TranslationPipeline(_fast_config())
        pipeline.llm_client = llm
        pipeline.db = db
        result = pipeline.run(
            input_path=source, output_path=tmp_path / "out.txt"
        )
        return result, db


def _logged_events(db: MagicMock) -> list[str]:
    return [
        call.args[2]
        for call in db.log_chunk_event.call_args_list
        if len(call.args) >= 3
    ]


# ---------------------------------------------------------------------------
# Crash isolation
# ---------------------------------------------------------------------------

def test_a_baseline_run_succeeds(tmp_path, monkeypatch) -> None:
    """Control: the harness itself must produce a working translation."""
    monkeypatch.chdir(tmp_path)
    result, _db = _run_pipeline(tmp_path)
    assert isinstance(result, PipelineResult)
    assert (tmp_path / "out.txt").exists()


def test_a_crashing_structure_audit_does_not_stop_the_translation(
    tmp_path, monkeypatch
) -> None:
    """Report-only must mean report-only, including its failure mode."""
    monkeypatch.chdir(tmp_path)

    def boom(*_args, **_kwargs):
        raise ValueError("simulated audit defect")

    with mock_patch("tarjomeh.core.pipeline.audit_payload", side_effect=boom):
        result, db = _run_pipeline(tmp_path)

    assert isinstance(result, PipelineResult), "the book must still be produced"
    assert (tmp_path / "out.txt").exists()
    assert "structure_audit_failed" in _logged_events(db), (
        "the failure must be recorded, not swallowed silently"
    )


def test_a_crashing_repair_helper_does_not_stop_the_translation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated repair defect")

    with mock_patch("tarjomeh.core.pipeline.repair_corruption", side_effect=boom):
        result, db = _run_pipeline(tmp_path)

    assert isinstance(result, PipelineResult)
    assert "unicode_corruption_repair_failed" in _logged_events(db)


@pytest.mark.parametrize(
    "target,event",
    [
        ("audit_payload", "structure_audit_failed"),
        ("repair_corruption", "unicode_corruption_repair_failed"),
    ],
)
def test_the_job_is_never_marked_paused_error_by_a_helper_defect(
    tmp_path, monkeypatch, target, event
) -> None:
    """pause_on_sequential_error defaults True, so an unguarded raise would have
    stopped the job on the first affected chunk."""
    monkeypatch.chdir(tmp_path)

    def boom(*_args, **_kwargs):
        raise ValueError("simulated defect")

    with mock_patch(f"tarjomeh.core.pipeline.{target}", side_effect=boom):
        _result, db = _run_pipeline(tmp_path)

    statuses = [
        call.args[1] for call in db.update_job_status.call_args_list
        if len(call.args) >= 2
    ]
    assert "paused_error" not in statuses, f"a {target} defect paused the job"
    assert event in _logged_events(db)


# ---------------------------------------------------------------------------
# The restore contract
# ---------------------------------------------------------------------------

def test_the_final_gate_now_receives_a_previous_version(tmp_path, monkeypatch) -> None:
    """Without `previous`, the corruption and duplicate checks cannot tell damage
    this step introduced from damage carried in from earlier."""
    monkeypatch.chdir(tmp_path)
    seen: list[dict] = []

    from tarjomeh.quality.integrity import PostEditIntegrityGate

    real_evaluate = PostEditIntegrityGate.evaluate

    def recording_evaluate(self, source, candidate, **kwargs):
        seen.append({"stage": kwargs.get("stage"), "previous": kwargs.get("previous")})
        return real_evaluate(self, source, candidate, **kwargs)

    with mock_patch.object(PostEditIntegrityGate, "evaluate", recording_evaluate):
        _run_pipeline(tmp_path)

    final = [entry for entry in seen if entry["stage"] == "final_translation"]
    assert final, "the final gate must run"
    assert final[0]["previous"] is not None, (
        "the final gate must be given a restore target"
    )
    assert str(final[0]["previous"]).strip(), "the restore target must be real text"
