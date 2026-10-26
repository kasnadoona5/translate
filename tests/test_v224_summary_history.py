"""Reviewed summary history must not revive exact superseded context."""

from copy import deepcopy

import pytest

from tarjomeh.memory.summary_revision import (
    document_hash,
    history_key,
    history_policy,
    select_summary,
)


def summary(text):
    return {
        "english_summary": text,
        "persian_summary": (
            "\u0627\u062f\u0639\u0627 \u0628\u0647 \u06cc\u06a9 \u0634\u0631\u0637 "
            "\u0648\u0627\u0628\u0633\u062a\u0647 \u0627\u0633\u062a."
        ),
        "input_trust": "advisory_inputs",
        "trust_reasons": ["old_review"],
    }


def ref(index):
    return {
        "chunk_index": index,
        "source_sha256": str(index) * 64,
        "target_sha256": str(index + 1) * 64,
    }


def fixture():
    safe = {"summary": summary("Earlier safe argument."), "coverage": [ref(0)]}
    bad = {
        "summary": summary("The wrong subject carries the signal."),
        "coverage": [ref(0), ref(1)],
    }
    corrected = summary("The enzyme carries the signal.")
    manifest = {
        "schema": 1,
        "job_id": "synthetic",
        "document_sha256": "a" * 64,
        "expected_memory_sha256": "b" * 64,
        "previous_summary": bad["summary"],
        "candidate_summary": corrected,
        "coverage": bad["coverage"],
        "supersedes": [list(history_key(bad))],
        "evidence": [
            {
                "chunk_index": 1,
                "source_sha256": "1" * 64,
                "start": 0,
                "end": 3,
                "quote": "The",
                "claim": "subject",
            }
        ],
    }
    digest = document_hash(manifest)
    entry = {
        "summary": corrected,
        "coverage": bad["coverage"],
        "reviewed_revision": {
            "schema": 1,
            "revision_id": digest,
            "approval_sha256": digest,
            "reviewer": "offline reviewer",
            "manifest": manifest,
        },
    }
    return [safe, bad, entry]


def test_superseded_summary_is_excluded_even_when_corrected_coverage_is_not_eligible():
    history = fixture()
    before = deepcopy(history)
    assert select_summary(history, lambda r: r["chunk_index"] < 2) == history[2]
    assert select_summary(history, lambda r: r["chunk_index"] < 1) == history[0]
    policy = history_policy(history)
    assert history_key(history[1]) in policy["blocked"]
    assert history == before


@pytest.mark.parametrize("field", ["approval_sha256", "revision_id", "reviewer", "manifest"])
def test_corrupt_revision_cannot_authorize_its_candidate(field):
    history = fixture()
    history[2]["reviewed_revision"][field] = "" if field == "reviewer" else "invalid"
    selected = select_summary(history, lambda r: True)
    assert selected != history[2]
    assert history_policy(history)["invalid"]


def test_changed_coverage_cannot_reactivate_the_exact_old_bad_version():
    history = fixture()
    # The correction depended on both canonical chunks; only the first survives.
    selected = select_summary(history, lambda r: r == ref(0))
    assert selected == history[0]


def test_no_revision_history_has_identical_latest_eligible_behavior():
    history = fixture()[:2]
    assert select_summary(history, lambda r: True) == history[-1]
    assert not history_policy(history)["blocked"]


def test_legacy_queue_can_retain_its_existing_empty_coverage_fallback():
    history = fixture()[:2]
    legacy = {"summary": summary("Earlier unscoped legacy summary."), "coverage": []}
    history.append(legacy)
    assert select_summary(history, lambda r: True, require_coverage=False) == legacy
    assert select_summary(history, lambda r: True) == history[1]


def test_no_revision_manager_queue_preserves_legacy_fallback():
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.memory.manager import MemoryManager

    current = MemoryManager(TarjomehConfig())
    legacy = {"summary": summary("Earlier unscoped legacy summary."), "coverage": []}
    current.summary_history = fixture()[:2] + [legacy]
    current.bilingual_summary.deserialize(current.summary_history[1]["summary"])
    current.summary_coverage = current.summary_history[1]["coverage"]
    current._queue_summary_coverage()
    assert current.bilingual_summary.serialize() == legacy["summary"]


def test_explicit_rollback_is_append_only_and_does_not_restore_revoked_candidate():
    history = fixture()
    record = history[-1]["reviewed_revision"]
    rollback = {
        "summary": deepcopy(history[1]["summary"]),
        "coverage": deepcopy(history[1]["coverage"]),
        "summary_revision_rollback": {"revision_id": record["revision_id"]},
    }
    original = deepcopy(history)
    history.append(rollback)
    assert select_summary(history, lambda r: True) == rollback
    assert history[:3] == original
    assert not history_policy(history)["blocked"]
