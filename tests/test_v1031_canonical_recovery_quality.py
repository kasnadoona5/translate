from __future__ import annotations

import ast
import asyncio
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    TranslationPipeline,
    _cached_recovery_candidate,
    _canonical_chunk_paragraph_identity,
    _chunk_style_policy,
    _final_canonical_admission_payload,
    _paragraph_structural_roles,
    _recovery_segment_cache_identity,
    _role_aware_recovery_groups,
    _unresolved_grounded_memory_issues,
    audit_canonical_document_identity,
)
from tarjomeh.jobs.database import ChunkStatus, JobDatabase
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.parsers.base import Chapter, Document, Paragraph, Section
from tarjomeh.runtime import runtime_behavior_probes, runtime_capabilities


def _chunk(text: str, roles: list[str] | None = None) -> Chunk:
    paragraphs = text.split("\n\n")
    return Chunk(
        index=0,
        text=text,
        chapter_title="Chapter",
        section_title="",
        metadata={
            "paragraph_indices": list(range(len(paragraphs))),
            "structural_roles": roles or ["body"] * len(paragraphs),
            "paragraph_protocol_version": 1,
        },
    )


def test_assembly_is_lexically_pure_for_canonical_identifiers(tmp_path) -> None:
    source = "Catalog identifiers."
    target = (
        "CB2 1UR؛ JC11.J47 2015 320.1-dc23 2015013426؛ "
        "politybooks.com؛ RES-051-27-0303"
    )
    chunk = _chunk(source)
    canonical, identity = _canonical_chunk_paragraph_identity(chunk, target)
    document = Document(
        title="",
        chapters=[Chapter("Chapter", sections=[Section(
            "", 1, [Paragraph(source)]
        )])],
    )
    pipeline = TranslationPipeline(TarjomehConfig())
    pipeline.db = JobDatabase(tmp_path / "jobs.db")
    pipeline.db.create_job("job", tmp_path / "source.pdf", {})
    pipeline.db.save_job_artifact(
        "job", "canonical_chunk_paragraphs_v1",
        {"version": 1, "chunks": {"0": identity}},
    )

    assembled = pipeline._assemble_translated_document(
        document, [chunk], {0: canonical}, job_id="job"
    )
    audit = audit_canonical_document_identity(
        assembled, [chunk], {0: canonical}
    )

    assert assembled.paragraphs[0].translated_text == target
    assert audit["lexically_identical"] is True


def test_final_admission_event_matches_atomic_chunk_text(tmp_path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    chunk = _chunk("Source.")
    target, identity = _canonical_chunk_paragraph_identity(chunk, "ترجمه.")
    db.create_job("job", tmp_path / "source.pdf", {})
    db.save_chunks("job", [chunk])
    admission = _final_canonical_admission_payload(target, identity)

    db.commit_chunk_checkpoint(
        "job", 0, ChunkStatus.COMPLETED, target, {},
        paragraph_identity=identity,
        canonical_admission=admission,
    )

    event = [
        item for item in db.get_chunk_events("job", 0)
        if item["event_type"] == "final_canonical_admission"
    ][-1]
    assert event["payload"]["canonical_target_hash"] == hashlib.sha256(
        target.encode("utf-8")
    ).hexdigest()
    assert db.get_chunks("job")[0]["translation"] == target


def test_manual_chunk_update_and_admission_event_are_atomic(tmp_path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    chunk = _chunk("Source.")
    db.create_job("job", tmp_path / "source.pdf", {})
    db.save_chunks("job", [chunk])
    target, identity = _canonical_chunk_paragraph_identity(chunk, "ترجمه.")

    db.update_chunk_with_event(
        "job", 0, ChunkStatus.COMPLETED, target,
        "final_canonical_admission",
        _final_canonical_admission_payload(target, identity),
    )

    assert db.get_chunks("job")[0]["translation"] == target
    assert db.get_chunk_events("job", 0)[-1]["payload"][
        "canonical_target_hash"
    ] == hashlib.sha256(target.encode("utf-8")).hexdigest()


def test_mixed_roles_group_only_contiguous_table_rows() -> None:
    chunk = _chunk(
        "Prose.\n\nRow one\n\nRow two\n\nMore prose.",
        ["body", "table", "table", "body"],
    )
    roles = _paragraph_structural_roles(chunk, 4)

    assert roles == ["body", "table", "table", "body"]
    assert _role_aware_recovery_groups(chunk.text.split("\n\n"), roles) == [
        (0, ["Prose."], "body"),
        (1, ["Row one", "Row two"], "table"),
        (3, ["More prose."], "body"),
    ]


def test_recovery_cache_is_exact_request_and_hash_bound() -> None:
    identity = _recovery_segment_cache_identity(
        segment_id="c10.p44",
        source="Source",
        prompt="Prompt",
        system_prompt="System",
        structural_role="table",
        request_profile='{"model":"combo2","provider":"openrouter"}',
    )
    candidate = "ترجمه"
    artifact = {"entries": {identity: {
        "candidate": candidate,
        "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
    }}}

    assert _cached_recovery_candidate(artifact, identity) == candidate
    artifact["entries"][identity]["candidate"] = "دستکاری"
    assert _cached_recovery_candidate(artifact, identity) == ""


def test_recovery_cache_merge_preserves_validated_siblings(tmp_path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("job", tmp_path / "source.pdf", {})

    db.merge_job_artifact_entry(
        "job", "translation_recovery_segments_v1", "entries", "first",
        {"candidate": "یک"},
    )
    db.merge_job_artifact_entry(
        "job", "translation_recovery_segments_v1", "entries", "second",
        {"candidate": "دو"},
    )

    artifact = db.get_job_artifact("job", "translation_recovery_segments_v1")
    assert artifact is not None
    assert set(artifact["entries"]) == {"first", "second"}


def test_grounded_minor_accuracy_at_point_six_quarantines_memory() -> None:
    critique = SimpleNamespace(issue_details=[{
        "issue_id": "p2:s1:accuracy:1",
        "source_segment_id": "p2:s1",
        "severity": "minor",
        "category": "accuracy",
        "confidence": 0.65,
        "source_quote": "institutional histories",
        "current_persian_quote": "تاریخ های نهادی",
        "suggested_correction": "تاریخ نهادها",
    }])

    assert [
        item["issue_id"] for item in _unresolved_grounded_memory_issues(critique)
    ] == ["p2:s1:accuracy:1"]


class _StyleDB:
    def get_chunk_events(self, _job_id: str, _chunk_index: int):
        detail = {
            "issue_id": "p2:s1:accuracy:1",
            "source_segment_id": "p2:s1",
            "severity": "minor",
            "category": "accuracy",
            "confidence": 0.65,
            "source_quote": "institutional histories",
            "current_persian_quote": "تاریخ های نهادی",
            "suggested_correction": "تاریخ نهادها",
        }
        return [
            {"event_type": "chunk_started", "payload": {}},
            {"event_type": "critique_completed", "payload": {
                "valid": True,
                "scores": {
                    "accuracy": 9,
                    "fluency": 9,
                    "terminology": 9,
                    "register": 9,
                    "average": 9,
                },
                "issue_details": [detail],
            }},
            {"event_type": "critique_needs_review", "payload": {
                "reason": "unresolved_grounded_quality_issue",
            }},
        ]


def test_style_memory_keeps_only_clean_paragraphs_from_review_chunk() -> None:
    policy = _chunk_style_policy(_StyleDB(), "job", 0)

    assert policy["approved"] is True
    assert policy["excluded_paragraphs"] == [1]
    assert policy["reason"] == "clean_final_critique"


class _SummaryLLM:
    async def chat(self, _prompt: str) -> str:
        return (
            "## English Summary\nThe theoretical epistem epistemological "
            "argument continues.\n## خلاصه فارسی\nبحث نظری ادامه می یابد."
        )

    def set_operation(self, _operation: str) -> None:
        return None


def test_summary_prefix_stutter_is_rejected_and_prior_retained() -> None:
    manager = MemoryManager(TarjomehConfig())
    manager.bilingual_summary.english_summary = "Prior accepted summary."
    manager.bilingual_summary.persian_summary = "خلاصه پذیرفته شده قبلی."

    report = asyncio.run(manager.update_bilingual_summary(
        _SummaryLLM(),
        "The theoretical and epistemological argument continues.",
        "بحث نظری و معرفت شناختی ادامه می یابد.",
    ))

    assert report["candidate_committed"] is False
    assert "english_summary_lexical_stutter" in report["candidate_quality"]["reasons"]
    assert manager.bilingual_summary.english_summary == "Prior accepted summary."


def test_v1031_release_scripts_are_parseable_and_9router_safe() -> None:
    for relative in (
        "scripts/audit_tarjomeh_v1031_reports.sh",
        "scripts/audit_tarjomeh_v1031_companion.sh",
    ):
        source = Path(relative).read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S)
        assert blocks, relative
        for block in blocks:
            ast.parse(block, filename=relative)
        assert "v10.31" in source
        assert "final_canonical_admission" in source

    deployment = Path("scripts/deploy_tarjomeh_v1031.sh").read_text(
        encoding="utf-8"
    )
    assert 'TAG="v10.31.0"' in deployment
    assert "docker system prune" not in deployment
    assert "docker image prune" not in deployment
    assert "VERIFY 9ROUTER UNCHANGED" in deployment
    assert "NINE_MOUNTS" in deployment


def test_v1031_runtime_contract_exercises_new_boundaries() -> None:
    manifest = runtime_capabilities()
    probes = runtime_behavior_probes()

    assert manifest["release"] == "v10.35.0"
    assert manifest["capabilities"]["canonical_export_is_lexically_pure"] is True
    assert manifest["capabilities"]["resumable_split_recovery_segments"] is True
    assert probes["canonical_admission_matches_final_text"] is True
    assert probes["mixed_roles_are_paragraph_scoped"] is True
