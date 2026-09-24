from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    TranslationPipeline,
    _canonical_chunk_paragraph_identity,
    _chunk_needs_review,
    _record_reconstructed_paragraph_identity_review,
    _salvage_local_refinement_edits,
    _target_units_from_identity,
)
from tarjomeh.jobs.database import ChunkStatus, JobDatabase
from tarjomeh.quality.critique import TranslationCritique
from tarjomeh.runtime import runtime_capabilities


class _AcceptingGate:
    def evaluate(self, _source: str, _candidate: str, **_kwargs: object):
        return SimpleNamespace(
            accepted=True,
            to_dict=lambda: {"accepted": True},
        )


class _EventDB:
    def __init__(self) -> None:
        self.events: list[tuple[str, int, str, dict[str, object]]] = []

    def log_chunk_event(
        self,
        job_id: str,
        chunk_index: int,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        self.events.append((job_id, chunk_index, event_type, payload))


def _multi_paragraph_chunk() -> Chunk:
    first = "First source paragraph."
    second = "Second source paragraph."
    text = f"{first}\n\n{second}"
    return Chunk(
        index=0,
        text=text,
        chapter_title="",
        section_title="",
        metadata={
            "paragraph_indices": [4, 5],
            "paragraph_protocol_version": 1,
            "source_paragraph_spans": [
                [0, len(first)],
                [len(first) + 2, len(text)],
            ],
        },
    )


def test_canonical_identity_recovers_boundary_without_lexical_change() -> None:
    chunk = _multi_paragraph_chunk()
    translation = "جملهٔ نخست است. جملهٔ دوم نیز اینجاست."

    canonical, identity = _canonical_chunk_paragraph_identity(
        chunk, translation
    )
    units = _target_units_from_identity(canonical, identity)

    assert units is not None and len(units) == 2
    assert " ".join(canonical.split()) == " ".join(translation.split())
    assert identity["reconstructed"] is True
    assert identity["paragraph_indices"] == [4, 5]


def test_checkpoint_persists_paragraph_identity_atomically(tmp_path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    chunk = _multi_paragraph_chunk()
    db.create_job("job", tmp_path / "book.pdf", {})
    db.save_chunks("job", [chunk])
    canonical, identity = _canonical_chunk_paragraph_identity(
        chunk, "ترجمهٔ نخست.\n\nترجمهٔ دوم."
    )

    db.commit_chunk_checkpoint(
        "job",
        0,
        ChunkStatus.COMPLETED,
        canonical,
        {"short_term": []},
        paragraph_identity=identity,
    )

    stored = db.get_job_artifact("job", "canonical_chunk_paragraphs_v1")
    assert stored["chunks"]["0"] == identity
    assert _target_units_from_identity(
        db.get_chunks("job")[0]["translation"], stored["chunks"]["0"]
    ) == ["ترجمهٔ نخست.", "ترجمهٔ دوم."]


def test_reconstructed_identity_is_review_only_before_memory_admission(
    tmp_path,
) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    chunk = _multi_paragraph_chunk()
    db.create_job("job", tmp_path / "book.pdf", {})
    db.save_chunks("job", [chunk])
    db.log_chunk_event("job", 0, "chunk_started", {})
    _canonical, identity = _canonical_chunk_paragraph_identity(
        chunk, "First target sentence. Second target sentence."
    )

    _record_reconstructed_paragraph_identity_review(db, "job", 0, identity)

    assert identity["reconstructed"] is True
    assert _chunk_needs_review(db, "job", 0) is True
    review = db.get_chunk_events("job", 0)[-1]
    assert review["event_type"] == "chunk_review_required"
    assert review["payload"]["reason"] == "paragraph_identity_reconstructed"


def test_critic_coverage_contract_accepts_complete_sentence_inventory() -> None:
    source = "First claim. Second claim."
    translation = "ادعای نخست. ادعای دوم."
    raw = json.dumps({
        "scores": {
            "accuracy": 9,
            "fluency": 9,
            "terminology": 9,
            "register": 9,
        },
        "overall": 9,
        "source_coverage": {
            "checked_source_segment_ids": ["p1:s1", "p1:s2"],
            "uncovered_source_segment_ids": [],
            "complete": True,
        },
        "issues": [],
    })

    result = TranslationCritique._parse_response(
        raw, source, translation, require_coverage=True
    )

    assert result.valid is True
    assert result.coverage_complete is True
    assert result.coverage_checked_segment_ids == ["p1:s1", "p1:s2"]


def test_critic_coverage_contract_rejects_unaccounted_source_sentence() -> None:
    source = "First claim. Second claim."
    raw = json.dumps({
        "scores": {
            "accuracy": 9,
            "fluency": 9,
            "terminology": 9,
            "register": 9,
        },
        "source_coverage": {
            "checked_source_segment_ids": ["p1:s1"],
            "uncovered_source_segment_ids": [],
            "complete": True,
        },
        "issues": [],
    })

    result = TranslationCritique._parse_response(
        raw, source, "ترجمه.", require_coverage=True
    )

    assert result.valid is False
    assert "coverage_checked_ids_incomplete" in result.validation_errors


def test_local_salvage_uses_grounded_paragraph_when_quote_repeats() -> None:
    previous = "این عبارت روشن است.\n\nاین عبارت روشن است."
    proposed = "این عبارت روشن است.\n\nاین استدلال روشن است."
    final, decisions, report = _salvage_local_refinement_edits(
        source="This phrase is clear.\n\nThis argument is clear.",
        previous=previous,
        proposed=proposed,
        issue_details=[{
            "issue_id": "second",
            "source_segment_id": "p2:s1",
            "source_quote": "This argument is clear.",
            "current_persian_quote": "این عبارت روشن است.",
        }],
        issue_decisions=[{
            "issue_id": "second",
            "decision": "accepted",
            "resulting_span": "این استدلال روشن است.",
        }],
        integrity_gate=_AcceptingGate(),  # type: ignore[arg-type]
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )

    assert final == proposed
    assert report["committed_count"] == 1
    assert decisions[0]["commit_status"] == "committed_local"


def test_final_canonical_admission_restores_exact_source_identifiers() -> None:
    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.config = TarjomehConfig()
    pipeline.db = _EventDB()
    chunk = Chunk(
        index=0,
        text="Write to politybooks.com at CB2 1UR; code RES-051-27-0303.",
        chapter_title="",
        section_title="",
        metadata={"paragraph_indices": [0]},
    )

    result = pipeline._canonicalize_final_translation(
        "job",
        0,
        chunk,
        "به politybooks. com در CB2 ۱UR بنویسید؛ کد RES-۰۵۱-۲۷-۰۳۰۳.",
    )

    assert "politybooks.com" in result
    assert "CB2 1UR" in result
    assert "RES-051-27-0303" in result
    payload = pipeline.db.events[-1][3]
    assert payload["final_source_artifact_reconciliation"]["repair_count"] >= 3


def test_final_canonical_admission_blocks_absent_source_identifier() -> None:
    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.config = TarjomehConfig()
    pipeline.db = _EventDB()
    chunk = Chunk(
        index=0,
        text="The code is RES-051-27-0303.",
        chapter_title="",
        section_title="",
        metadata={"paragraph_indices": [0]},
    )

    with pytest.raises(ValueError, match="source-identifier admission"):
        pipeline._canonicalize_final_translation(
            "job", 0, chunk, "این کد در متن حذف شده است."
        )


def test_v1029_release_contract_and_scripts_are_valid() -> None:
    root = Path(__file__).parents[1]
    paths = [
        root / "scripts" / "deploy_tarjomeh_v1029.sh",
        root / "scripts" / "audit_tarjomeh_v1029_reports.sh",
        root / "scripts" / "audit_tarjomeh_v1029_companion.sh",
    ]
    scripts = [path.read_text(encoding="utf-8") for path in paths]

    assert runtime_capabilities()["release"] == "v10.37.0"
    assert 'TAG="v10.29.1"' in scripts[0]
    assert 'git diff --quiet v10.27.0 "$TAG"' in scripts[0]
    assert 'ARG BASE_IMAGE' in scripts[0]
    assert 'FROM ${BASE_IMAGE}' in scripts[0]
    assert '--build-arg "BASE_IMAGE=$ROLLBACK"' in scripts[0]
    assert 'RUN pip install --no-cache-dir --no-deps .' in scripts[0]
    assert 'docker build --pull=false -t "$CANDIDATE" .' not in scripts[0]
    assert "RUNNING_V1029_CONFIRMED" in scripts[0]
    assert "canonical_chunk_paragraphs_v1" in scripts[0]
    assert "checked_source_segment_ids" in scripts[0]
    for unsafe in (
        "docker system prune",
        "docker image prune",
        "docker builder prune",
        "docker rm -f 9router",
    ):
        assert unsafe not in scripts[0]
    for source, path in zip(scripts, paths, strict=True):
        blocks = re.findall(
            r"<<'PY'[^\n]*\n(.*?)\nPY(?=\n|$)", source, re.DOTALL
        )
        assert blocks, path
        for block in blocks:
            ast.parse(block, filename=str(path))
