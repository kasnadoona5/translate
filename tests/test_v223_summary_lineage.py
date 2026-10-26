"""Cumulative summary provenance must survive reviewed updates and restarts."""

import asyncio
from copy import deepcopy

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import _chapter_summary_memory_admission
from tarjomeh.memory.bilingual_summary import BilingualSummary
from tarjomeh.memory.manager import MemoryManager

FA = (
    "\u0627\u062f\u0639\u0627 \u0628\u0647 \u06cc\u06a9 \u0634\u0631\u0637 "
    "\u0648\u0627\u0628\u0633\u062a\u0647 \u0627\u0633\u062a."
)
RESPONSE = (
    "## English Summary\nThe earlier and later claims are conditional.\n"
    "## \u062e\u0644\u0627\u0635\u0647 \u0641\u0627\u0631\u0633\u06cc\n" + FA
)


class Client:
    def __init__(self, response=RESPONSE, failure=None):
        self.response, self.failure, self.prompts = response, failure, []

    async def chat(self, prompt):
        self.prompts.append(prompt)
        if self.failure:
            raise self.failure
        return self.response


def manager():
    result = MemoryManager(TarjomehConfig())
    result.quality_policy_version = 1
    return result


def add(current, index, reliable=True, target=FA):
    return current.update_after_translation(
        Chunk(index, f"The claim {index} is conditional.", "Chapter", "",
              metadata={"structural_roles": ["body"]}),
        target, long_term_reliable=reliable,
        reliability_reasons=[] if reliable else ["old_passage_needs_review"],
    )


def update(current, index, trust="reviewed_inputs", reasons=None, client=None):
    client = client or Client()
    source, target, refs = current.pending_summary_inputs(index)
    report = asyncio.run(current.update_bilingual_summary(
        client, source, target, coverage_refs=refs,
        input_trust=trust, trust_reasons=reasons,
    ))
    assert len(client.prompts) == 1
    return report


@pytest.mark.parametrize("restore", [False, True])
def test_real_batch_admission_cannot_upgrade_cumulative_advisory_summary(restore):
    current = manager()
    policies = {0: add(current, 0, reliable=False)}

    class DB:
        def get_chunk_events(self, job, index):
            return [{"event_type": "memory_update_policy", "payload": policies[index]}]

    chunks = [Chunk(i, f"The claim {i} is conditional.", "Chapter", "") for i in range(2)]
    first = _chapter_summary_memory_admission(DB(), "offline", chunks, 0, contributing_indices=[0])
    assert first["input_trust"] == "advisory_inputs"
    assert update(current, 0, first["input_trust"], first["trust_reasons"])["candidate_committed"]
    if restore:
        state = deepcopy(current.to_dict())
        current = manager()
        current.from_dict(state)
    policies[1] = add(current, 1)
    second = _chapter_summary_memory_admission(DB(), "offline", chunks, 1, contributing_indices=[1])
    assert second["input_trust"] == "reviewed_inputs"
    report = update(current, 1, second["input_trust"], second["trust_reasons"])
    assert report["candidate_committed"]
    assert report["input_trust"] == "advisory_inputs"
    assert "old_passage_needs_review" in report["trust_reasons"]
    assert "prior_summary_advisory_inputs" in report["trust_reasons"]
    assert [ref["chunk_index"] for ref in current.summary_coverage] == [0, 1]
    assert current.bilingual_summary.persian_summary == FA
    assert current.bilingual_summary.english_summary == (
        "The earlier and later claims are conditional."
    )
    assert report["authority"] == "argument_orientation_only"


@pytest.mark.parametrize("previous", [
    None, "reviewed_inputs", "advisory_inputs", "context_only", "unknown",
])
@pytest.mark.parametrize("new", ["reviewed_inputs", "advisory_inputs", "unknown"])
def test_effective_trust_and_reason_bounds(previous, new):
    current = manager()
    if previous is not None:
        current.bilingual_summary.update(RESPONSE)
        current.bilingual_summary.input_trust = previous
        current.bilingual_summary.trust_reasons = ["old", "shared"]
    else:
        current.bilingual_summary.set_input_trust("advisory_inputs", ["inactive_empty_reason"])
    old = current.bilingual_summary.serialize()
    add(current, 0)
    report = update(current, 0, new, ["shared", "new"])
    expected = (
        "reviewed_inputs" if new == "reviewed_inputs" and previous in (None, "reviewed_inputs")
        else "advisory_inputs"
    )
    assert report["input_trust"] == expected
    reasons = report["trust_reasons"]
    assert reasons.count("shared") == 1 and "new" in reasons
    assert ("old" in reasons) is (previous is not None)
    assert "inactive_empty_reason" not in reasons
    assert ("prior_summary_advisory_inputs" in reasons) is (
        previous not in (None, "reviewed_inputs")
    )
    assert old["input_trust"] == (previous or "advisory_inputs")


def test_replaced_canonical_dependency_does_not_create_permanent_taint():
    current = manager()
    add(current, 0)
    assert update(current, 0)["input_trust"] == "reviewed_inputs"
    add(current, 1, reliable=False)
    assert update(current, 1, "advisory_inputs", ["obsolete"])["input_trust"] == "advisory_inputs"
    add(current, 1, target=FA + " " + FA)
    assert current.bilingual_summary.input_trust == "reviewed_inputs"
    assert "obsolete" not in current.bilingual_summary.trust_reasons
    assert update(current, 1)["input_trust"] == "reviewed_inputs"


@pytest.mark.parametrize("failure", [RuntimeError("offline failure"), asyncio.CancelledError()])
def test_failure_preserves_exact_state(failure):
    current = manager()
    add(current, 0)
    update(current, 0, "advisory_inputs", ["old"])
    add(current, 1)
    before = deepcopy(current.to_dict())
    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            update(current, 1, client=Client(failure=failure))
    else:
        assert not update(current, 1, client=Client(failure=failure))["candidate_committed"]
    assert current.to_dict() == before


def test_rejected_candidate_preserves_prior_summary_and_pending():
    current = manager()
    add(current, 0)
    update(current, 0, "advisory_inputs", ["old"])
    add(current, 1)
    before = deepcopy(current.to_dict())
    report = update(current, 1, client=Client("invalid"))
    assert not report["candidate_committed"]
    after = current.to_dict()
    after["last_rejected_summary"] = before["last_rejected_summary"]
    assert after == before


def test_reason_cap_and_prior_object_are_preserved():
    previous = BilingualSummary()
    previous.english_summary = "An earlier claim."
    previous.set_input_trust("advisory_inputs", [f"reason_{i}" for i in range(15)])
    before = previous.serialize()
    candidate = BilingualSummary()
    candidate.set_cumulative_input_trust("reviewed_inputs", ["reason_0", "new"], previous)
    assert len(candidate.trust_reasons) == 12
    assert len(set(candidate.trust_reasons)) == 12
    assert candidate.trust_reasons[0] == "prior_summary_advisory_inputs"
    assert previous.serialize() == before


def test_pure_probe_fails_if_trust_inheritance_is_disabled(monkeypatch):
    from tarjomeh.runtime import _v223_behavior_probes

    assert _v223_behavior_probes() == {"cumulative_summary_input_trust": True}

    def broken(self, trust, reasons, previous):
        self.set_input_trust(trust, reasons)
        return {"inherited_advisory": False}

    monkeypatch.setattr(BilingualSummary, "set_cumulative_input_trust", broken)
    assert _v223_behavior_probes() == {"cumulative_summary_input_trust": False}
