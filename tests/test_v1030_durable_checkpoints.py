from __future__ import annotations

import ast
import re
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import TranslationPipeline
from tarjomeh.jobs.database import ChunkStatus, JobDatabase, JobStatus
from tarjomeh.runtime import runtime_capabilities


def _config(*, output_format: str = "txt") -> TarjomehConfig:
    config = TarjomehConfig()
    config.output.format = output_format
    config.output.bilingual_mode = "target_only"
    config.translation.stop_after_chapter = 1
    config.translation.enable_critique = False
    config.translation.enable_back_translation = False
    config.translation.enable_web_context = False
    config.translation.enable_book_research = False
    config.glossary.enable_auto_extraction = False
    config.memory.enable_4layer = False
    return config


def _source(root: Path) -> Path:
    source = root / "book.txt"
    source.write_text(
        "Chapter 1: Opening\nFirst chapter text.\n\n"
        "Chapter 2: Argument\nSecond chapter text.",
        encoding="utf-8",
    )
    return source


def _fake_translation(**kwargs: object) -> str:
    return "FA " + kwargs["chunk"].text  # type: ignore[union-attr]


def test_chunk_memory_and_checkpoint_intent_commit_atomically() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        root = Path(temp_dir)
        db = JobDatabase(root / "jobs.db")
        db.create_job("atomic", root / "book.txt", {})
        db.save_chunks(
            "atomic",
            [Chunk(0, "Source", "Chapter", "", {"chapter_position": 1})],
        )
        intent = {
            "chapter_position": 1,
            "chapter_title": "Chapter",
            "boundary_chunk_index": 0,
            "following_chapter_position": 2,
            "preview_positions": [1],
            "recovery": "normal_boundary",
        }

        db.commit_chunk_checkpoint(
            "atomic",
            0,
            ChunkStatus.COMPLETED,
            "Target",
            {"short_term": ["Target"]},
            paragraph_identity={"source_units": [], "target_units": []},
            chapter_checkpoint=intent,
        )

        state = db.get_job_artifact("atomic", "chapter_checkpoints") or {}
        assert state["version"] == 3
        assert state["pending"]["chapter_position"] == 1
        assert state["reached_positions"] == []
        assert db.get_memory_state("atomic") == {"short_term": ["Target"]}
        assert db.get_chunks("atomic")[0]["translation"] == "Target"


def test_failed_preview_is_recovered_before_any_new_translation() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        root = Path(temp_dir)
        source = _source(root)
        output = root / "translated.txt"
        pipeline = TranslationPipeline(_config())
        pipeline.db = JobDatabase(root / "jobs.db")
        pipeline._translate_single_chunk = MagicMock(
            side_effect=_fake_translation
        )

        with patch(
            "tarjomeh.core.pipeline.get_exporter",
            side_effect=RuntimeError("simulated preview interruption"),
        ), pytest.raises(RuntimeError, match="preview interruption"):
            pipeline.run(source, output, job_id="recover-preview")

        state = pipeline.db.get_job_artifact(
            "recover-preview", "chapter_checkpoints"
        ) or {}
        assert state["pending"]["chapter_position"] == 1
        assert state["pending"]["failure_count"] == 1
        assert state["reached_positions"] == []
        assert pipeline._translate_single_chunk.call_count == 1
        failed_job = pipeline.db.get_job("recover-preview") or {}
        assert ".checkpoint-" not in str(failed_job.get("output_path") or "")

        pipeline.config.translation.enable_book_research = True
        pipeline._prepare_book_research = MagicMock(
            side_effect=AssertionError(
                "research must wait for pending checkpoint publication"
            )
        )
        pipeline.run(source, output, job_id="recover-preview")

        paused = pipeline.db.get_job("recover-preview") or {}
        state = pipeline.db.get_job_artifact(
            "recover-preview", "chapter_checkpoints"
        ) or {}
        assert paused["raw_status"] == JobStatus.PAUSED
        assert Path(paused["output_path"]) == output
        assert output.is_file() and output.stat().st_size > 0
        assert state["pending"] is None
        assert state["reached_positions"] == [1]
        assert pipeline._translate_single_chunk.call_count == 1
        pipeline._prepare_book_research.assert_not_called()

        pipeline.config.translation.enable_book_research = False
        del pipeline._prepare_book_research
        pipeline.run(source, output, job_id="recover-preview")
        assert pipeline.db.get_job("recover-preview")["raw_status"] == JobStatus.COMPLETED
        assert pipeline._translate_single_chunk.call_count == 2


def test_legacy_completed_boundary_is_recovered_before_next_chunk() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        root = Path(temp_dir)
        source = _source(root)
        output = root / "translated.txt"
        pipeline = TranslationPipeline(_config())
        pipeline.db = JobDatabase(root / "jobs.db")
        pipeline._translate_single_chunk = MagicMock(
            side_effect=_fake_translation
        )

        with patch(
            "tarjomeh.core.pipeline.get_exporter",
            side_effect=RuntimeError("legacy interruption"),
        ), pytest.raises(RuntimeError):
            pipeline.run(source, output, job_id="legacy-recovery")

        with sqlite3.connect(pipeline.db.db_path) as conn:
            conn.execute(
                "DELETE FROM job_artifacts "
                "WHERE job_id=? AND artifact_key='chapter_checkpoints'",
                ("legacy-recovery",),
            )
            conn.execute(
                "UPDATE chunks SET status=?, translation=? "
                "WHERE job_id=? AND chunk_index=1",
                (
                    ChunkStatus.COMPLETED,
                    "FA Chapter 2: Argument\nSecond chapter text.",
                    "legacy-recovery",
                ),
            )
            conn.commit()

        pipeline.run(source, output, job_id="legacy-recovery")

        assert pipeline._translate_single_chunk.call_count == 1
        state = pipeline.db.get_job_artifact(
            "legacy-recovery", "chapter_checkpoints"
        ) or {}
        assert state["reached_positions"] == [1]
        assert state["latest_publication"]["recovery"] == "legacy_completed_boundary"
        assert state["latest_publication"]["late_recovery"] is True
        events = pipeline.db.get_chunk_events("legacy-recovery", 0)
        assert any(
            event["event_type"] == "legacy_checkpoint_recovered"
            for event in events
        )
        assert any(
            event["event_type"] == "late_chapter_checkpoint_recovery"
            for event in events
        )


def test_checkpoint_completion_is_idempotent() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        root = Path(temp_dir)
        db = JobDatabase(root / "jobs.db")
        db.create_job("idempotent", root / "book.txt", {})
        preview = root / "preview.docx"
        preview.write_bytes(b"preview")
        intent = {
            "chapter_position": 3,
            "chapter_title": "Part Three",
            "boundary_chunk_index": 8,
            "following_chapter_position": 4,
            "preview_positions": [1, 2, 3],
            "recovery": "normal_boundary",
        }
        db.save_pending_chapter_checkpoint("idempotent", intent)

        first = db.complete_chapter_checkpoint(
            "idempotent", intent, preview,
            output_sha256="abc", output_bytes=7,
        )
        second = db.complete_chapter_checkpoint(
            "idempotent", intent, preview,
            output_sha256="abc", output_bytes=7,
        )

        assert first["reached_positions"] == [3]
        assert second["reached_positions"] == [3]
        assert len(second["publications"]) == 1
        assert db.get_job("idempotent")["raw_status"] == JobStatus.PAUSED


def test_pending_retry_rederives_preview_scope_from_current_config() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        root = Path(temp_dir)
        db = JobDatabase(root / "jobs.db")
        db.create_job("scope", root / "book.txt", {})
        chunks = [
            Chunk(0, "One", "First", "", {"chapter_position": 1}),
            Chunk(1, "Two", "Second", "", {"chapter_position": 2}),
        ]
        db.save_chunks("scope", chunks)
        db.update_chunk("scope", 0, ChunkStatus.COMPLETED, "یک")
        db.save_job_artifact(
            "scope",
            "chapter_checkpoints",
            {
                "version": 3,
                "reached_positions": [],
                "pending": {
                    "chapter_position": 1,
                    "boundary_chunk_index": 0,
                    "preview_positions": [1, 2, 999],
                    "recovery": "pending_retry",
                },
            },
        )
        config = _config()
        pipeline = TranslationPipeline(config)
        pipeline.db = db

        recovered = pipeline._recover_pending_chapter_checkpoint(
            "scope", chunks, {0: "یک"}
        )

        assert recovered is not None
        assert recovered["preview_positions"] == [1]


def test_v1030_runtime_declares_durable_checkpoint_policy() -> None:
    manifest = runtime_capabilities()
    assert manifest["release"] == "v10.37.0"
    assert manifest["capabilities"]["durable_chapter_checkpoint_recovery"] is True
    assert manifest["policy_versions"]["checkpoint_export"] == 3


def test_v1030_release_scripts_are_parseable_and_checkpoint_aware() -> None:
    for relative in (
        "scripts/audit_tarjomeh_v1030_reports.sh",
        "scripts/audit_tarjomeh_v1030_companion.sh",
    ):
        source = Path(relative).read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S)
        assert blocks, relative
        for block in blocks:
            ast.parse(block, filename=relative)
        assert "v10.30" in source

    deployment = Path("scripts/deploy_tarjomeh_v1030.sh").read_text(
        encoding="utf-8"
    )
    assert 'TAG="v10.30.0"' in deployment
    assert "durable_chapter_checkpoint_recovery" in deployment
    assert "_recover_pending_chapter_checkpoint" in deployment
    assert "persist_job_output=False" in deployment
    assert "docker system prune" not in deployment
    assert "docker image prune" not in deployment
    assert deployment.count("VERIFY 9ROUTER UNCHANGED") == 1
    assert "NINE_MOUNTS" in deployment
