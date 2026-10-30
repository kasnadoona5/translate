"""Explicit text revisions must be atomic, stale-safe and evidence preserving."""

import json
import sqlite3
from copy import deepcopy

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.pipeline import _canonical_chunk_paragraph_identity
from tarjomeh.jobs.database import JobDatabase
from tarjomeh.jobs import text_revision as revision
from tarjomeh.memory.summary_revision import document_hash, history_policy, select_summary, text_hash
from tarjomeh.parsers.base import Chapter, Document, Paragraph, Section


@pytest.fixture
def saved(tmp_path, monkeypatch):
    path = tmp_path / "jobs.db"
    db = JobDatabase(path)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source fixture")
    db.create_job("sample", source, {})
    chunks, pairs, identities, refs = [], [], {}, []
    before = ["The first hypothesis is contested.", "The second hypothesis is contested."]
    after = ["The first hypothesis remains contested.", "The second hypothesis remains contested."]
    for index, text in enumerate(before):
        metadata = {"paragraph_indices": [index], "source_paragraph_spans": [[0, len(text)]],
                    "source_paragraph_hashes": [text_hash(text)], "structural_roles": ["body"]}
        chunk = Chunk(index, text, "Section", "", metadata=metadata)
        chunks.append(chunk)
        _, identities[str(index)] = _canonical_chunk_paragraph_identity(chunk, text)
        refs.append({"chunk_index": index, "source_sha256": text_hash(text), "target_sha256": text_hash(text)})
        pairs.append({"entry_id": index, "chunk_index": index, "source": text, "translation": text,
                      "active": True, "reliable": True, "canonical_target_hash": text_hash(text)})
    db.save_chunks("sample", chunks)
    for index, text in enumerate(before):
        db.update_chunk("sample", index, "completed", text)
    db.save_job_artifact("sample", "canonical_chunk_paragraphs_v1", {"version": 1, "chunks": identities})
    db.save_job_artifact("sample", "extra_final_refinement_v1", {"spent": True})
    summary = {"english_summary": "These are hypotheses.", "persian_summary": "Summary unchanged."}
    state = {"quality_policy_version": 1, "past_translations": pairs,
             "short_term_context": [{"chunk_index": i, "source": t, "translation": t, "trust": "trusted"}
                                    for i, t in enumerate(before)],
             "bilingual_summary": summary, "summary_coverage": refs, "summary_pending": [],
             "summary_history": [{"summary": summary, "coverage": refs}],
             "proper_nouns": {"nouns": {"Name": "Name"}}, "book_context": "unchanged research",
             "style_samples": [], "style_profile": "", "style_sample_records": [], "mapping_evidence": {}}
    db.save_memory_state("sample", state)
    db.update_job_status("sample", "paused")
    db.log_chunk_event("sample", 0, "llm_call_attempt", {"request_id": "preserved"})
    document = Document(title="Source", chapters=[Chapter(title="Section", sections=[Section(
        title="", level=1, paragraphs=[Paragraph(text) for text in before])])])
    monkeypatch.setattr(revision, "_source_document", lambda *args: (document, text_hash("source fixture")))
    proposal = {"job_id": "sample", "changes": [
        {"chunk_index": i, "before": before[i], "after": after[i], "reason": "Source-reviewed wording"}
        for i in range(2)]}
    return path, db, state, proposal, tmp_path / "backup.sqlite"


def perform(saved):
    path, _, _, proposal, backup = saved
    manifest = revision.prepare(path, proposal)
    receipt = revision.apply(path, manifest, approval=document_hash(manifest), reviewer="Human reviewer",
                             backup=backup, operator_locks_held=True)
    return manifest, receipt


def test_read_only_preparation_and_atomic_application_preserve_evidence(saved):
    path, db, before, proposal, backup = saved
    raw = path.read_bytes()
    manifest = revision.prepare(path, proposal)
    assert path.read_bytes() == raw
    assert revision.check(path, manifest)["status"] == "dry_run_only"
    assert manifest["summary_continuity"]["summary"] == before["bilingual_summary"]
    manifest, receipt = perform(saved)
    after = db.get_memory_state("sample")
    for key in ("bilingual_summary", "proper_nouns", "style_samples", "style_profile", "style_sample_records", "book_context", "mapping_evidence"):
        assert after[key] == before[key]
    assert len(after["past_translations"]) == 4
    assert all(not p["active"] for p in after["past_translations"][:2])
    assert all(p["active"] and not p["reliable"] for p in after["past_translations"][2:])
    assert all(p["trust"] == "advisory_review" for p in after["short_term_context"])
    assert after["summary_history"][:-1] == before["summary_history"]
    assert not history_policy(after["summary_history"])["invalid"]
    selected = select_summary(after["summary_history"], lambda ref: ref in after["summary_coverage"])
    assert selected["summary"] == before["bilingual_summary"]
    assert selected["coverage"] == after["summary_coverage"]
    assert db.get_job_artifact("sample", "extra_final_refinement_v1") == {"spent": True}
    assert len([e for e in db.get_chunk_events("sample", 0) if e["event_type"] == "llm_call_attempt"]) == 1
    assert all(row["status"] == "needs_review" for row in db.get_chunks("sample"))
    assert db.get_job("sample")["status"] == "paused"
    qualities = [e["payload"] for e in db.get_chunk_events("sample", 0) if e["event_type"] == "final_quality_admission"]
    assert qualities[-1]["durable_authority"] is False and qualities[-1]["final_scores"] == {}
    with sqlite3.connect(backup) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("SELECT translation FROM chunks ORDER BY chunk_index").fetchall() == [(c["before"],) for c in proposal["changes"]]
    assert revision.apply(path, manifest, approval=document_hash(manifest), reviewer="Human reviewer",
                          backup=backup, operator_locks_held=True)["status"] == "already_applied"
    assert revision.check(path, manifest)["status"] == "already_applied_verified"


@pytest.mark.parametrize("mutation", ["chunk", "memory", "term", "reservation", "worker", "status", "publication", "event"])
def test_any_intervening_state_invalidates_approval(saved, mutation):
    path, db, state, proposal, backup = saved
    manifest = revision.prepare(path, proposal)
    if mutation == "chunk":
        db.update_chunk("sample", 1, "needs_review", "Changed canonical text.")
    elif mutation == "memory":
        db.save_memory_state("sample", {**state, "book_context": "changed"})
    elif mutation in {"term", "reservation", "publication"}:
        db.save_job_artifact("sample", mutation, {"changed": True})
    elif mutation == "worker":
        db.claim_worker("sample", "other")
    elif mutation == "status":
        db.update_job_status("sample", "running")
    else:
        db.log_chunk_event("sample", 0, "new_audit_evidence", {})
    before = deepcopy(db.get_chunks("sample"))
    with pytest.raises(ValueError):
        revision.apply(path, manifest, approval=document_hash(manifest), reviewer="Human reviewer",
                       backup=backup, operator_locks_held=True)
    assert db.get_chunks("sample") == before
    assert not backup.exists()


@pytest.mark.parametrize("defect", ["approval", "locks", "empty", "duplicate", "lost_number", "style", "summary", "forged_manifest"])
def test_unsafe_edits_and_missing_approval_stop(saved, defect):
    path, db, state, proposal, backup = saved
    if defect == "empty":
        proposal["changes"][0]["after"] = ""
    elif defect == "duplicate":
        proposal["changes"].append(proposal["changes"][0])
    elif defect == "lost_number":
        proposal["changes"][0]["after"] = ""
    elif defect == "style":
        state["style_sample_records"] = [{"source_chunk_index": 0, "source_text": proposal["changes"][0]["before"],
                                          "text": proposal["changes"][0]["before"], "active": True}]
        db.save_memory_state("sample", state)
    elif defect == "summary":
        state["summary_coverage"][0]["target_sha256"] = "0" * 64
        db.save_memory_state("sample", state)
    if defect in {"empty", "duplicate", "lost_number", "style", "summary"}:
        with pytest.raises(ValueError):
            revision.prepare(path, proposal)
        return
    manifest = revision.prepare(path, proposal)
    if defect == "forged_manifest":
        manifest["changes"][0]["integrity"]["accepted"] = False
    with pytest.raises(ValueError):
        revision.apply(path, manifest, approval="wrong" if defect == "approval" else document_hash(manifest),
                       reviewer="Human reviewer", backup=backup, operator_locks_held=defect != "locks")
    assert not backup.exists()


def test_failure_after_first_write_rolls_back_all_chunks_and_memory(saved, monkeypatch):
    path, db, before, proposal, backup = saved
    manifest = revision.prepare(path, proposal)
    from tarjomeh.core import pipeline
    original = pipeline._final_canonical_admission_payload
    calls = []
    def fail_second(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated crash")
        return original(*args)
    monkeypatch.setattr(pipeline, "_final_canonical_admission_payload", fail_second)
    with pytest.raises(RuntimeError, match="simulated crash"):
        revision.apply(path, manifest, approval=document_hash(manifest), reviewer="Human reviewer",
                       backup=backup, operator_locks_held=True)
    assert db.get_memory_state("sample") == before
    assert [c["translation"] for c in db.get_chunks("sample")] == [c["before"] for c in proposal["changes"]]
    assert backup.exists()


def test_corrupt_continuity_is_not_summary_authority(saved):
    _, db, _, _, _ = saved
    perform(saved)
    history = db.get_memory_state("sample")["summary_history"]
    history[-1]["coverage"][0]["target_sha256"] = "0" * 64
    assert history_policy(history)["invalid"] == [1]


def test_source_drift_after_application_is_not_hidden_by_receipt(saved):
    path, db, _, _, _ = saved
    manifest, _ = perform(saved)
    from pathlib import Path
    Path(db.get_job("sample")["input_path"]).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source bytes"):
        revision.check(path, manifest)


def test_memory_and_chunk_drift_between_backup_and_write_cannot_commit(saved, monkeypatch):
    path, db, state, proposal, backup = saved
    manifest = revision.prepare(path, proposal)
    original = revision._backup
    def raced(*args):
        snapshot = original(*args)
        db.update_chunk("sample", 1, "needs_review", "Concurrent edit.")
        return snapshot
    monkeypatch.setattr(revision, "_backup", raced)
    with pytest.raises(ValueError):
        revision.apply(path, manifest, approval=document_hash(manifest), reviewer="Human reviewer",
                       backup=backup, operator_locks_held=True)
    assert db.get_chunk("sample", 0)["translation"] == proposal["changes"][0]["before"]
    assert db.get_memory_state("sample") == state
    assert backup.exists()


def test_audits_keep_old_policy_history_without_reusing_old_authority(saved):
    from tarjomeh.memory.text_revision import partition_revision_policy_events

    _, db, _, proposal, _ = saved
    manifest, receipt = perform(saved)
    old = {"chunk_index": 0, "canonical_target_hash": text_hash(proposal["changes"][0]["before"]), "long_term_reliable": True}
    current = {"chunk_index": 0, "canonical_target_hash": text_hash(proposal["changes"][0]["after"]), "long_term_reliable": False}
    unknown = {"chunk_index": 0, "long_term_reliable": True}
    events = [old, current, unknown]
    artifacts = {revision.PREFIX + document_hash(manifest): receipt}
    result = partition_revision_policy_events(events, db.get_chunks("sample"), artifacts, "sample")
    assert result["current"] == [current, unknown]
    assert result["superseded_hash_bound_history"] == [old]
    assert events == [old, current, unknown]
    bad = deepcopy(artifacts)
    bad[next(iter(bad))]["revision_id"] = "0" * 64
    assert partition_revision_policy_events(events, db.get_chunks("sample"), bad, "sample")["current"] == events
