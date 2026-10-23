"""Summary failure recovery must not publish a partially prepared memory state."""

import asyncio
from copy import deepcopy

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.memory.bilingual_summary import BilingualSummary
from tarjomeh.memory.manager import MemoryManager

FA = (
    "\u0627\u062f\u0639\u0627 \u0628\u0647 \u06cc\u06a9 \u0634\u0631\u0637 "
    "\u0648\u0627\u0628\u0633\u062a\u0647 \u0627\u0633\u062a."
)
HEADER = "\n## \u062e\u0644\u0627\u0635\u0647 \u0641\u0627\u0631\u0633\u06cc\n"
OLD = "## English Summary\nThe earlier claim is conditional." + HEADER + FA
NEW = "## English Summary\nThe later claim is conditional." + HEADER + FA


class Client:
    def __init__(self, response=NEW, failure=None):
        self.response = response
        self.failure = failure
        self.prompts = []
        self.operations = []

    def set_operation(self, name):
        self.operations.append(name)

    async def chat(self, prompt):
        self.prompts.append(prompt)
        if self.failure:
            raise self.failure
        return self.response


def _add(manager, index, target=FA):
    manager.update_after_translation(
        Chunk(index=index, text=f"The claim {index} is conditional.",
              chapter_title="Chapter", section_title="",
              metadata={"structural_roles": ["body"]}),
        target,
        style_evidence={"final_scores": dict.fromkeys(
            ("accuracy", "fluency", "terminology", "register"), 9.5,
        )},
    )


def _manager(*, empty=False, policy=1):
    manager = MemoryManager(TarjomehConfig())
    manager.quality_policy_version = policy
    if not empty:
        _add(manager, 0)
        source, target, refs = manager.pending_summary_inputs(0)
        result = asyncio.run(manager.update_bilingual_summary(
            Client(OLD), source, target, coverage_refs=refs,
        ))
        assert result["candidate_committed"]
    _add(manager, 1)
    return manager


def _state(manager):
    return deepcopy({
        "summary": manager.bilingual_summary.serialize(),
        "coverage": manager.summary_coverage,
        "pending": manager.summary_pending,
        "history": manager.summary_history,
    })


def _update(manager, client):
    source, target, refs = manager.pending_summary_inputs(1)
    return asyncio.run(manager.update_bilingual_summary(
        client, source, target, coverage_refs=refs,
    ))


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("policy", [0, 1])
def test_reconciliation_error_retains_exact_previous_state_and_restart(monkeypatch, empty, policy):
    manager = _manager(empty=empty, policy=policy)
    before = _state(manager)
    baseline_restart = MemoryManager(TarjomehConfig())
    baseline_restart.from_dict(deepcopy(manager.to_dict()))
    client = Client()

    def fail(**kwargs):
        raise RuntimeError("injected reconciliation failure")

    monkeypatch.setattr(manager, "reconcile_bilingual_summary", fail)
    result = _update(manager, client)
    assert _state(manager) == before
    assert result["candidate_committed"] is False
    assert result["status"] == "failed_retained_previous_summary"
    assert result["failure_stage"] == "summary_commit_preparation"
    assert result["context_retained"] is (not empty)
    assert len(client.prompts) == 1
    resumed = MemoryManager(TarjomehConfig())
    resumed.from_dict(deepcopy(manager.to_dict()))
    # Legacy empty-summary trust is normalized on every restore, with or without failure.
    assert _state(resumed) == _state(baseline_restart)


def test_history_serialization_error_cannot_clear_pending_or_extend_coverage(monkeypatch):
    manager = _manager()
    before = _state(manager)
    serialize = BilingualSummary.serialize

    def fail(candidate):
        if candidate.english_summary == "The later claim is conditional.":
            raise RuntimeError("injected history serialization failure")
        return serialize(candidate)

    with monkeypatch.context() as context:
        context.setattr(BilingualSummary, "serialize", fail)
        result = _update(manager, Client())
    assert _state(manager) == before
    assert result["candidate_committed"] is False
    assert result["failure_stage"] == "summary_commit_preparation"


@pytest.mark.parametrize("stage", ["provider_call", "candidate_parse", "candidate_quality"])
def test_preparation_failures_report_retention_without_retry(monkeypatch, stage):
    manager = _manager()
    before = _state(manager)
    client = Client(
        failure=RuntimeError("injected provider failure") if stage == "provider_call" else None,
    )
    if stage != "provider_call":
        method = "update" if stage == "candidate_parse" else "candidate_quality"

        def fail(*args, **kwargs):
            raise RuntimeError("injected preparation failure")

        monkeypatch.setattr(BilingualSummary, method, fail)
    result = _update(manager, client)
    assert _state(manager) == before
    assert result["candidate_committed"] is False
    assert result["failure_stage"] == stage
    assert result["authority"] == "argument_orientation_only"
    assert len(client.prompts) == 1


def test_cancellation_before_commit_leaves_everything_unchanged():
    manager = _manager()
    before = _state(manager)
    client = Client(failure=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        _update(manager, client)
    assert _state(manager) == before
    assert len(client.prompts) == 1


def test_success_commits_matching_state_without_mutating_prior_records():
    manager = _manager()
    previous = _state(manager)
    old_summary, old_coverage = manager.bilingual_summary, manager.summary_coverage
    old_pending, old_history = manager.summary_pending, manager.summary_history
    client = Client()
    result = _update(manager, client)
    assert result["candidate_committed"] is True
    assert manager.bilingual_summary.english_summary == "The later claim is conditional."
    assert manager.bilingual_summary.persian_summary == FA
    assert [ref["chunk_index"] for ref in manager.summary_coverage] == [0, 1]
    assert manager.summary_pending == []
    assert manager.summary_history[-1]["summary"] == manager.bilingual_summary.serialize()
    assert manager.summary_history[-1]["coverage"] == manager.summary_coverage
    assert old_summary.serialize() == previous["summary"]
    assert old_coverage == previous["coverage"] and old_pending == previous["pending"]
    assert old_history == previous["history"]
    assert len(client.prompts) == 1 and client.operations == ["bilingual_summary_update"]
    resumed = MemoryManager(TarjomehConfig())
    resumed.from_dict(deepcopy(manager.to_dict()))
    assert _state(resumed) == _state(manager)
    _add(resumed, 0, FA + " " + FA)
    assert not resumed.bilingual_summary.get_context()
    assert resumed.summary_coverage == []
    assert [ref["chunk_index"] for ref in resumed.summary_pending] == [0, 1]


def test_canonical_mismatch_still_raises_without_provider_or_memory_change():
    manager = _manager()
    before = _state(manager)
    source, target, refs = manager.pending_summary_inputs(1)
    client = Client()
    with pytest.raises(ValueError, match="canonical pending"):
        asyncio.run(manager.update_bilingual_summary(
            client, source + " changed", target, coverage_refs=refs,
        ))
    assert _state(manager) == before and client.prompts == []


def test_rejected_candidate_keeps_full_diagnostics_and_pending_coverage():
    manager = _manager()
    before = _state(manager)
    result = _update(manager, Client("bad summary"))
    assert result["candidate_committed"] is False and _state(manager) == before
    assert manager.last_rejected_summary["candidate_committed"] is False
    assert "english_summary_missing_or_too_short" in result["candidate_quality"]["reasons"]


def test_probe_is_pure_and_detects_disabled_commit(monkeypatch):
    import socket
    from pathlib import Path

    from tarjomeh.runtime import _v21_behavior_probes

    def forbidden(*args, **kwargs):
        raise AssertionError("Probe attempted external I/O")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    assert _v21_behavior_probes()["atomic_summary_commit"]
    monkeypatch.setattr(MemoryManager, "_commit_bilingual_summary", lambda *a, **k: {})
    assert not _v21_behavior_probes()["atomic_summary_commit"]


def test_candidate_reconciliation_never_mutates_active_summary(monkeypatch):
    manager = _manager()
    before = _state(manager)
    old_word = "\u062a\u0634\u06a9\u06cc\u0644\u0627\u062a"
    new_word = "\u0633\u0627\u0632\u0645\u0627\u0646"
    monkeypatch.setattr(manager.proper_nouns, "all_nouns", lambda: {"organization": new_word})
    monkeypatch.setattr(manager.proper_nouns, "provenance_for", lambda source: {
        "origin": "accepted_correction", "superseded": [{"target": old_word}],
    })
    candidate = BilingualSummary()
    candidate.english_summary = "The organization describes a conditional claim."
    candidate.persian_summary = old_word + " " + FA
    result = manager.reconcile_bilingual_summary(summary=candidate)
    assert result["replacement_count"] == 1
    assert candidate.persian_summary == new_word + " " + FA
    assert _state(manager) == before


def test_legacy_success_keeps_coverage_collections_unchanged():
    manager = _manager(policy=0)
    before = _state(manager)
    result = _update(manager, Client())
    assert result["candidate_committed"] is True
    assert manager.bilingual_summary.english_summary == "The later claim is conditional."
    assert manager.summary_pending == before["pending"]
    assert manager.summary_coverage == before["coverage"]
    assert manager.summary_history == before["history"]
