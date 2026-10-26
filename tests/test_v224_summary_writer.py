"""The administrative writer is explicit, atomic and independent of providers."""

import sqlite3
from copy import deepcopy

import pytest

from tarjomeh.jobs.database import JobDatabase
from tarjomeh.jobs.summary_revision import apply, prepare, rollback
from tarjomeh.memory.summary_revision import document_hash, history_key, text_hash

FA = (
    "\u0627\u062f\u0639\u0627 \u0628\u0647 \u06cc\u06a9 \u0634\u0631\u0637 "
    "\u0648\u0627\u0628\u0633\u062a\u0647 \u0627\u0633\u062a."
)


@pytest.fixture
def saved(tmp_path):
    path = tmp_path / "jobs.db"
    db = JobDatabase(path)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"synthetic source")
    db.create_job("synthetic", source, {})
    old = dict(
        english_summary="The earlier claim has the wrong subject.",
        persian_summary=FA,
        input_trust="advisory_inputs",
        trust_reasons=["review"],
    )
    ref = dict(
        chunk_index=0,
        source_sha256=text_hash("The enzyme carries the signal."),
        target_sha256=text_hash(FA),
    )
    state = dict(
        quality_policy_version=1,
        bilingual_summary=old,
        summary_coverage=[ref],
        summary_pending=[],
        summary_history=[dict(summary=old, coverage=[ref])],
        past_translations=[
            dict(
                chunk_index=0, source="The enzyme carries the signal.", translation=FA, active=True
            )
        ],
        style_samples=["unchanged"],
        other_layer={"retained": True},
    )
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE jobs SET status='paused',source_sha256=?", (text_hash("synthetic source"),)
        )
        conn.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
            ("synthetic", 0, "The enzyme carries the signal.", FA, "{}", "completed"),
        )
    db.save_memory_state("synthetic", state)
    proposal = dict(
        job_id="synthetic",
        document_sha256=text_hash("synthetic source"),
        previous_summary=old,
        candidate_summary=dict(old, english_summary="The enzyme carries the signal."),
        supersedes=[list(history_key(state["summary_history"][0]))],
        evidence=[
            dict(
                chunk_index=0,
                source_sha256=ref["source_sha256"],
                start=0,
                end=10,
                quote="The enzyme",
                claim="subject",
            )
        ],
    )
    return path, db, state, proposal, tmp_path / "backup.sqlite"


def execute(saved):
    path, db, state, proposal, backup = saved
    manifest = prepare(path, proposal)
    receipt = apply(
        path,
        manifest,
        approval=document_hash(manifest),
        reviewer="Human reviewer",
        backup=backup,
        operator_locks_held=True,
    )
    return manifest, receipt


def test_prepare_read_only_apply_idempotent_rollback_append_only(saved):
    path, db, before, proposal, backup = saved
    raw = path.read_bytes()
    manifest = prepare(path, proposal)
    assert path.read_bytes() == raw
    manifest, receipt = execute(saved)
    after = db.get_memory_state("synthetic")
    assert after["summary_history"][:-1] == before["summary_history"]
    assert after["bilingual_summary"] == proposal["candidate_summary"]
    for key in before.keys() - {"summary_history", "bilingual_summary"}:
        assert after[key] == before[key]
    assert sqlite3.connect(backup).execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert (
        apply(
            path,
            manifest,
            approval=document_hash(manifest),
            reviewer="Human reviewer",
            backup=backup,
            operator_locks_held=True,
        )["status"]
        == "already_applied"
    )
    result = rollback(
        path,
        "synthetic",
        document_hash(manifest),
        approval=receipt["after_sha256"],
        reviewer="Human reviewer",
        backup=backup.with_name("rollback.sqlite"),
        operator_locks_held=True,
    )
    restored = db.get_memory_state("synthetic")
    assert restored["bilingual_summary"] == before["bilingual_summary"]
    assert restored["summary_history"][:-1] == after["summary_history"]
    assert result["status"] == "rolled_back"
    events = db.get_chunk_events("synthetic")
    assert [e["event_type"] for e in events] == [
        "reviewed_summary_revision",
        "summary_revision_rollback",
    ]


@pytest.mark.parametrize(
    "change", ["status", "worker", "memory", "chunk", "source", "approval", "locks", "quote"]
)
def test_refuses_without_mutating_job(saved, change):
    path, db, before, proposal, backup = saved
    manifest = prepare(path, proposal)
    args = dict(
        approval=document_hash(manifest),
        reviewer="Human reviewer",
        backup=backup,
        operator_locks_held=True,
    )
    with sqlite3.connect(path) as conn:
        if change == "status":
            conn.execute("UPDATE jobs SET status='running'")
        if change == "worker":
            conn.execute(
                "INSERT INTO job_workers (job_id,worker_id,state,acquired_at,heartbeat_at) "
                "VALUES ('other','w','active','old','old')"
            )
        if change == "chunk":
            conn.execute("UPDATE chunks SET translation='changed'")
    if change == "memory":
        db.save_memory_state("synthetic", dict(before, other_layer="changed"))
    if change == "source":
        proposal_path = path.parent / "source.pdf"
        proposal_path.write_bytes(b"changed")
    if change == "approval":
        args["approval"] = "no"
    if change == "locks":
        args["operator_locks_held"] = False
    if change == "quote":
        manifest["evidence"][0]["quote"] = "wrong quote"
        args["approval"] = document_hash(manifest)
    unchanged = deepcopy(db.get_memory_state("synthetic"))
    with pytest.raises((ValueError, RuntimeError)):
        apply(path, manifest, **args)
    assert db.get_memory_state("synthetic") == unchanged
    assert db.get_chunk_events("synthetic") == []


def test_event_failure_rolls_back_memory_and_receipt(saved):
    path, db, before, proposal, backup = saved
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TRIGGER no_event BEFORE INSERT ON chunk_events "
            "BEGIN SELECT RAISE(ABORT,'test'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        execute(saved)
    assert db.get_memory_state("synthetic") == before
    assert (
        db.get_job_artifact(
            "synthetic", "reviewed_summary_revision:" + document_hash(prepare(path, proposal))
        )
        is None
    )


def test_rollback_refuses_intervening_memory(saved):
    path, db, before, proposal, backup = saved
    manifest, receipt = execute(saved)
    state = db.get_memory_state("synthetic")
    state["other_layer"] = "new work"
    db.save_memory_state("synthetic", state)
    with pytest.raises(ValueError, match="stale"):
        rollback(
            path,
            "synthetic",
            document_hash(manifest),
            approval=receipt["after_sha256"],
            reviewer="Human reviewer",
            backup=backup.with_name("rollback.sqlite"),
            operator_locks_held=True,
        )
    assert db.get_memory_state("synthetic") == state


@pytest.mark.parametrize("failure", ["backup", "late_worker", "late_state", "late_chunk"])
def test_backup_and_transaction_races_leave_saved_state_intact(saved, monkeypatch, failure):
    from tarjomeh.jobs import summary_revision as module

    path, db, before, proposal, backup = saved
    original = module._backup

    def injected(*args):
        if failure == "backup":
            raise OSError("unavailable backup disk")
        result = original(*args)
        with sqlite3.connect(path) as conn:
            if failure == "late_worker":
                conn.execute(
                    "INSERT INTO job_workers (job_id,worker_id,state,acquired_at,heartbeat_at) "
                    "VALUES ('synthetic','w','active','now','now')"
                )
            if failure == "late_chunk":
                conn.execute("UPDATE chunks SET translation='intervening work'")
        if failure == "late_state":
            db.save_memory_state("synthetic", dict(before, other_layer="new work"))
        return result

    monkeypatch.setattr(module, "_backup", injected)
    with pytest.raises((ValueError, OSError)):
        execute(saved)
    expected = dict(before, other_layer="new work") if failure == "late_state" else before
    assert db.get_memory_state("synthetic") == expected
    assert db.get_chunk_events("synthetic") == []


@pytest.mark.parametrize("field", ["english_summary", "persian_summary", "input_trust"])
def test_invalid_or_upgraded_candidate_cannot_be_prepared(saved, field):
    path, db, before, proposal, backup = saved
    proposal["candidate_summary"][field] = "reviewed_inputs" if field == "input_trust" else ""
    with pytest.raises(ValueError):
        prepare(path, proposal)
    assert db.get_memory_state("synthetic") == before
    assert not backup.exists()


def test_restart_retranslation_and_dependency_invalidation(saved):
    from tarjomeh.chunking.chunker import Chunk
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.memory.manager import MemoryManager

    path, db, before, proposal, backup = saved
    execute(saved)
    state = db.get_memory_state("synthetic")
    current = MemoryManager(TarjomehConfig())
    current.from_dict(deepcopy(state))
    assert current.bilingual_summary.serialize() == proposal["candidate_summary"]
    restored = MemoryManager(TarjomehConfig())
    restored.from_dict(current.to_dict())
    assert restored.bilingual_summary.serialize() == proposal["candidate_summary"]
    rows = [
        {
            "chunk_index": 0,
            "text": "The enzyme carries the signal.",
            "translation": FA,
            "status": "completed",
        }
    ]
    view, _ = restored.prior_context_view(0, rows)
    assert not view.bilingual_summary.english_summary
    view, _ = restored.prior_context_view(1, rows)
    assert view.bilingual_summary.serialize() == proposal["candidate_summary"]
    current.update_after_translation(
        Chunk(0, "The enzyme carries the signal.", "Chapter", ""),
        FA + " updated",
        long_term_reliable=False,
    )
    assert not current.bilingual_summary.english_summary
    assert current.summary_pending
    assert current.summary_history == state["summary_history"]


def test_unbalanced_candidate_rejected_without_export_policy_change(saved):
    path, db, before, proposal, backup = saved
    proposal["candidate_summary"]["english_summary"] += " ("
    with pytest.raises(ValueError, match="Unbalanced"):
        prepare(path, proposal)
    assert db.get_memory_state("synthetic") == before


def test_corrupt_history_summary_cannot_leave_unproven_candidate_active(saved):
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.memory.manager import MemoryManager

    path, db, before, proposal, backup = saved
    execute(saved)
    state = db.get_memory_state("synthetic")
    state["summary_history"][-1]["summary"]["english_summary"] = "Corrupt historical text."
    restored = MemoryManager(TarjomehConfig())
    restored.from_dict(state)
    assert restored.bilingual_summary.serialize() != proposal["candidate_summary"]


def test_rollback_event_identifies_rollback_operator(saved):
    path, db, before, proposal, backup = saved
    manifest, receipt = execute(saved)
    rollback(
        path,
        "synthetic",
        document_hash(manifest),
        approval=receipt["after_sha256"],
        reviewer="Different rollback reviewer",
        backup=backup.with_name("rollback.sqlite"),
        operator_locks_held=True,
    )
    assert (
        db.get_chunk_events("synthetic")[-1]["payload"]["reviewer"] == "Different rollback reviewer"
    )
