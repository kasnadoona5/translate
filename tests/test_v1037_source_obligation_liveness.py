"""A faithful academic argument must not be trapped by a narrow noun list."""

from __future__ import annotations

from pathlib import Path

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.pipeline import (
    _blocking_structure_findings,
    _cached_source_obligation_candidate,
    _candidate_text_hash,
    _final_candidate_selection_payload,
    _final_canonical_admission_payload,
    _persist_source_obligation_candidate,
    _source_obligation_resume_action,
    _source_obligation_resolution_payload,
)
from tarjomeh.jobs.database import ChunkStatus, JobDatabase
from tarjomeh.quality.structure_audit import audit_payload
from tarjomeh.runtime import runtime_behavior_probes, runtime_capabilities


SOURCE = (
    "The argument makes three claims. First, one; second, two; third, three."
)
FAITHFUL = (
    "این استدلال سه مدعای اصلی دارد. نخست، یک؛ دوم، دو؛ سوم، سه."
)
WRONG_COUNT = FAITHFUL.replace("سه مدعای", "دو مدعای")
UNRECOGNIZED = FAITHFUL.replace("مدعای", "فرضیهٔ")


def test_generic_mudda_announcement_matches_complete_source() -> None:
    assert audit_payload(SOURCE, FAITHFUL)["findings"] == []
    assert _source_obligation_resume_action(SOURCE, FAITHFUL, {
        "failure_count": 4,
    }) == "recheck"


def test_wrong_explicit_number_still_blocks() -> None:
    findings = audit_payload(SOURCE, WRONG_COUNT)["findings"]
    assert len(_blocking_structure_findings(findings)) == 1
    assert findings[0]["details"]["candidate_announced"] == 2


def test_unrecognized_announcement_reports_candidate_excerpt() -> None:
    finding = audit_payload(SOURCE, UNRECOGNIZED)["findings"][0]
    assert finding["details"]["candidate_episode_recognized"] is False
    assert finding["details"]["candidate_excerpt"] == UNRECOGNIZED
    assert "سه" in finding["details"]["candidate_cardinal_tokens_untyped"]


def test_repeated_cached_failure_has_one_fresh_generation_then_stops(tmp_path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("job", tmp_path / "book.pdf", {})
    db.save_chunks("job", [Chunk(0, SOURCE, "", "")])
    findings = audit_payload(SOURCE, WRONG_COUNT)["findings"]
    first = _persist_source_obligation_candidate(
        db, "job", 0, SOURCE, WRONG_COUNT, findings
    )
    assert first["same_candidate_failures"] == 1
    assert _source_obligation_resume_action(SOURCE, WRONG_COUNT, first) == "recheck"
    second = _persist_source_obligation_candidate(
        db, "job", 0, SOURCE, WRONG_COUNT, findings
    )
    assert _source_obligation_resume_action(SOURCE, WRONG_COUNT, second) == "regenerate"
    third = _persist_source_obligation_candidate(
        db, "job", 0, SOURCE, WRONG_COUNT, findings,
        fresh_generation_count=1,
    )
    assert third["same_candidate_failures"] == 3
    assert _source_obligation_resume_action(SOURCE, WRONG_COUNT, third) == "stop"


def test_changed_finding_does_not_inherit_old_repeat_count(tmp_path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("job", tmp_path / "book.pdf", {})
    db.save_chunks("job", [Chunk(0, SOURCE, "", "")])
    findings = audit_payload(SOURCE, WRONG_COUNT)["findings"]
    _persist_source_obligation_candidate(db, "job", 0, SOURCE, WRONG_COUNT, findings)
    old = _persist_source_obligation_candidate(
        db, "job", 0, SOURCE, WRONG_COUNT, findings
    )
    changed = [dict(findings[0], check_id="different_check")]
    changed_entry = _persist_source_obligation_candidate(
        db, "job", 0, SOURCE, WRONG_COUNT, changed
    )
    assert old["same_candidate_failures"] == 2
    assert changed_entry["same_candidate_failures"] == 1
    assert _source_obligation_resume_action(
        SOURCE, WRONG_COUNT, changed_entry
    ) == "recheck"


def test_admitted_replacement_resolves_pending_recovery_atomically(tmp_path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("job", tmp_path / "book.pdf", {})
    db.save_chunks("job", [Chunk(0, SOURCE, "", "")])
    _persist_source_obligation_candidate(
        db, "job", 0, SOURCE, WRONG_COUNT,
        audit_payload(SOURCE, WRONG_COUNT)["findings"],
    )
    final_hash = _candidate_text_hash(FAITHFUL)
    identity = {"canonical_target_hash": final_hash, "target_count": 1}
    quality = {"candidate_target_hash": final_hash}
    db.commit_chunk_checkpoint(
        "job", 0, ChunkStatus.COMPLETED, FAITHFUL, {"short_term": []},
        paragraph_identity=identity,
        candidate_selection=_final_candidate_selection_payload(FAITHFUL, identity),
        canonical_admission=_final_canonical_admission_payload(FAITHFUL, identity),
        final_quality_admission=quality,
        source_obligation_resolution=_source_obligation_resolution_payload(
            SOURCE, FAITHFUL
        ),
    )
    assert _cached_source_obligation_candidate(db, "job", 0, SOURCE) == ("", {})
    entry = db.get_job_artifact("job", "source_obligation_recovery_v1")["entries"]["0"]
    assert entry["status"] == "resolved"
    assert entry["replaced_candidate_sha256"] == _candidate_text_hash(WRONG_COUNT)
    assert entry["resolved_candidate_sha256"] == final_hash


def test_unattested_replacement_cannot_resolve_pending_recovery(tmp_path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("job", tmp_path / "book.pdf", {})
    db.save_chunks("job", [Chunk(0, SOURCE, "", "")])
    _persist_source_obligation_candidate(
        db, "job", 0, SOURCE, WRONG_COUNT,
        audit_payload(SOURCE, WRONG_COUNT)["findings"],
    )
    db.commit_chunk_checkpoint(
        "job", 0, ChunkStatus.COMPLETED, FAITHFUL, {"short_term": []},
        source_obligation_resolution=_source_obligation_resolution_payload(
            SOURCE, FAITHFUL
        ),
    )
    cached, entry = _cached_source_obligation_candidate(db, "job", 0, SOURCE)
    assert cached == WRONG_COUNT
    assert entry["status"] == "pending"


def test_v1037_runtime_and_operator_scripts_preserve_safety_contract() -> None:
    assert runtime_capabilities()["release"] == "v10.37.0"
    assert all(runtime_behavior_probes().values())
    deploy = Path("scripts/deploy_tarjomeh_v1037.sh").read_text(
        encoding="utf-8"
    )
    assert 'TAG="v10.37.0"' in deploy
    assert "RUNNING_V1037_CONFIRMED" in deploy
    assert "NINE_CONTAINER_ID" in deploy
    assert "docker system prune" not in deploy
    assert "docker volume prune" not in deploy
    assert "docker rm -f 9router" not in deploy
    for kind in ("reports", "companion"):
        script = Path(f"scripts/audit_tarjomeh_v1037_{kind}.sh").read_text(
            encoding="utf-8"
        )
        assert "source_obligation_fresh_events" in script
        assert "source_obligation_exhausted_events" in script
        assert "pending_finished_source_obligations" in script
