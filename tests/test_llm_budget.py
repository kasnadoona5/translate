"""Tests for LLM output-budget prediction and length recovery (Phase 8).

Two groups:

* **Invariants** - behaviour that must be identical before and after the
  Phase 8 changes. These are the safety net: attempt one must keep its prompt,
  model and reasoning policy, translation must never lose reasoning, and the
  50K floor / 85K ceiling constants must stay put (a cap is not a quota).
* **Budget learning and ceiling** - the defects themselves. The escalation
  observed in production (32,280 -> 150,494 requested against an 85,000 clamp)
  came from learning demand from *failed* calls and from a ceiling that could
  report more room than the context window physically has.
"""

from __future__ import annotations

import threading

import pytest

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import (
    _PRESERVE_REASONING_ON_ALL_LENGTH_RETRIES,
    LLMClient,
    _use_no_reasoning_recovery,
)

COMBO = "combo1"
SERVED = "deepseek-v4-flash"


def _client(**recovery_overrides) -> LLMClient:
    """An LLMClient shell with no network or tokenizer setup.

    Only config plus the budget-history state is needed; every function under
    test is pure apart from that.
    """
    config = TarjomehConfig()
    config.llm.model = COMBO
    config.llm.openrouter.api_keys = ["sk-test"]
    for name, value in recovery_overrides.items():
        setattr(config.llm.recovery, name, value)

    client = LLMClient.__new__(LLMClient)
    client.config = config
    client._budget_history = {}
    client._budget_lock = threading.Lock()
    return client


def _payload(prompt: str = "translate this", max_tokens: int = 12000) -> dict:
    return {
        "model": COMBO,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }


def _usage(completion: int, reasoning: int = 0) -> dict:
    return {
        "completion_tokens": completion,
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    }


# ===========================================================================
# Invariants — these must pass unchanged before and after Phase 8
# ===========================================================================

def test_budget_constants_are_unchanged() -> None:
    """max_tokens is a CAP, not a quota.

    Requesting 50,000 and using 3,000 bills 3,000, so these constants cost
    nothing on a call that behaves. Lowering them would risk truncating a
    legitimately long critique on a denser book - i.e. tuning to one book.
    """
    recovery = TarjomehConfig().llm.recovery
    assert recovery.predictive_min_tokens == 50000
    assert recovery.adaptive_max_tokens == 85000
    assert recovery.context_window_tokens == 131072
    assert recovery.context_safety_tokens == 2048
    assert recovery.history_window == 20


def test_translation_never_loses_reasoning_on_a_length_retry() -> None:
    assert "translation" in _PRESERVE_REASONING_ON_ALL_LENGTH_RETRIES
    assert "translation_split_recovery" in _PRESERVE_REASONING_ON_ALL_LENGTH_RETRIES


@pytest.mark.parametrize("attempt", [0, 1])
def test_no_reasoning_recovery_is_not_used_on_early_critique_attempts(attempt) -> None:
    """Attempts before the last keep full reasoning; quality is not traded away."""
    assert not _use_no_reasoning_recovery(
        operation="critique",
        attempt=attempt,
        max_attempts=4,
        has_fallback=True,
        final_expanded=False,
    )


def test_no_reasoning_recovery_is_used_only_on_the_last_attempt() -> None:
    assert _use_no_reasoning_recovery(
        operation="critique",
        attempt=3,
        max_attempts=4,
        has_fallback=True,
        final_expanded=False,
    )


def test_preflight_changes_only_max_tokens() -> None:
    """Attempt one keeps its prompt, model and reasoning policy byte for byte."""
    client = _client()
    original = _payload()
    original["reasoning"] = {"enabled": True, "exclude": True}

    payload, calculation = client._preflight_payload(original, "translation", 0)

    assert payload["messages"] == original["messages"]
    assert payload["model"] == original["model"]
    assert payload["reasoning"] == original["reasoning"]
    assert set(payload) == set(original), "no key added or removed"
    assert calculation["stage"] == "preflight"


def test_preflight_is_a_no_op_when_recovery_is_disabled() -> None:
    client = _client(enabled=False)
    original = _payload()
    payload, calculation = client._preflight_payload(original, "translation", 0)
    assert payload is original
    assert calculation == {}


def test_usage_evidence_does_not_mix_token_scales() -> None:
    """A router reporting reasoning above completion_tokens must be clamped."""
    client = _client()
    evidence = client._usage_evidence(_usage(1000, reasoning=999999), content="")
    assert evidence["completion_tokens"] == 1000
    assert evidence["reasoning_tokens"] == 1000
    assert evidence["reported_reasoning_tokens"] == 999999


def test_usage_evidence_infers_reasoning_when_unreported() -> None:
    client = _client()
    evidence = client._usage_evidence(_usage(5000), content="short answer")
    assert evidence["completion_tokens"] == 5000
    assert evidence["reasoning_tokens"] == 5000 - evidence["visible_tokens"]


# ===========================================================================
# 8.1 — never request more output than the context window can serve
# ===========================================================================

def test_ceiling_respects_the_context_window_on_a_large_prompt() -> None:
    """max() reported more room than physically exists, so the request was
    unservable: prompt 100,000 + answer 50,000 against a 131,072 window."""
    client = _client()
    recovery = client.config.llm.recovery
    # ~100k tokens of prompt.
    payload = _payload("word " * 100_000, max_tokens=50_000)

    ceiling, prompt_tokens = client._adaptive_ceiling(payload)
    budget = recovery.context_window_tokens - recovery.context_safety_tokens

    assert prompt_tokens > 50_000, "test needs a genuinely large prompt"
    assert prompt_tokens + ceiling <= budget, (
        f"unservable: prompt {prompt_tokens} + ceiling {ceiling} > {budget}"
    )


def test_ceiling_is_never_floored_by_what_was_already_requested() -> None:
    """The old final `max(payload_max, ceiling)` meant the ceiling could not bind."""
    client = _client()
    payload = _payload("word " * 100_000, max_tokens=85_000)
    ceiling, _ = client._adaptive_ceiling(payload)
    assert ceiling < 85_000


def test_preflight_never_exceeds_the_window_on_a_large_prompt() -> None:
    client = _client()
    recovery = client.config.llm.recovery
    payload, calculation = client._preflight_payload(
        _payload("word " * 100_000, max_tokens=12_000), "critique", 0
    )
    budget = recovery.context_window_tokens - recovery.context_safety_tokens
    assert calculation["estimated_prompt_tokens"] + payload["max_tokens"] <= budget


def test_ceiling_still_allows_the_full_allowance_on_a_small_prompt() -> None:
    """The fix must not shrink normal calls: a short prompt keeps the 85K cap."""
    client = _client()
    ceiling, _ = client._adaptive_ceiling(_payload("short prompt", max_tokens=12_000))
    assert ceiling == 85_000


def test_preflight_still_applies_the_50k_floor_on_a_small_prompt() -> None:
    client = _client()
    payload, _ = client._preflight_payload(_payload("short prompt"), "critique", 0)
    assert payload["max_tokens"] >= 50_000


# ===========================================================================
# 8.2 — learn only from successful, visible completions
# ===========================================================================

def test_a_truncated_call_teaches_nothing() -> None:
    """This is the ratchet: ceil(failed_budget * 1.5) was recorded as demand,
    so failing at 85,000 recorded 127,500 and the floor never came back down."""
    client = _client()
    client._record_budget_observation(
        payload=_payload(max_tokens=61_944),
        operation="critique",
        usage=_usage(61_944, reasoning=61_000),
        content="",
        finish_reason="length",
        response_model=SERVED,
    )
    assert client._budget_history == {}, "a truncated call must not raise the floor"


def test_an_empty_response_teaches_nothing() -> None:
    client = _client()
    client._record_budget_observation(
        payload=_payload(),
        operation="critique",
        usage=_usage(85_000, reasoning=85_000),
        content="   ",
        finish_reason="stop",
        response_model=SERVED,
    )
    assert client._budget_history == {}


def test_a_successful_call_records_completion_tokens() -> None:
    """completion_tokens includes reasoning, which is correct: the budget has
    to cover reasoning + answer."""
    client = _client()
    client._record_budget_observation(
        payload=_payload(),
        operation="critique",
        usage=_usage(31_380, reasoning=28_000),
        content="a real critique",
        finish_reason="stop",
        response_model=SERVED,
    )
    demand, samples = client._history_demand(
        client._budget_key("critique", _payload(), SERVED)
    )
    assert samples == 1
    assert demand == 31_380


# ===========================================================================
# 8.3 — key history by the model that actually served the call
# ===========================================================================

def test_history_is_keyed_by_the_serving_model_not_the_combo() -> None:
    """One combo fronts 14 models; keying on the combo name let a looping
    model's statistics poison the budget every other model inherits."""
    client = _client()
    for model, consumed in (("deepseek-v4-flash", 80_000), ("minimax-m3", 9_000)):
        client._record_budget_observation(
            payload=_payload(),
            operation="critique",
            usage=_usage(consumed),
            content="ok",
            finish_reason="stop",
            response_model=model,
        )

    deepseek, _ = client._history_demand(
        client._budget_key("critique", _payload(), "deepseek-v4-flash")
    )
    minimax, _ = client._history_demand(
        client._budget_key("critique", _payload(), "minimax-m3")
    )
    assert deepseek == 80_000
    assert minimax == 9_000, "minimax must not inherit deepseek's demand"


def test_budget_key_falls_back_to_the_configured_model_at_preflight() -> None:
    """No response model is known before the call; history is simply empty."""
    client = _client()
    key = client._budget_key("critique", _payload(), "")
    assert COMBO in key
    assert client._history_demand(key) == (0, 0)


def test_operations_do_not_share_a_budget() -> None:
    client = _client()
    client._record_budget_observation(
        payload=_payload(), operation="critique", usage=_usage(70_000),
        content="ok", finish_reason="stop", response_model=SERVED,
    )
    translation_key = client._budget_key("translation", _payload(), SERVED)
    assert client._history_demand(translation_key) == (0, 0)


# ===========================================================================
# 8.4 — bounded rolling history with decay
# ===========================================================================

def test_one_exceptional_call_does_not_define_the_floor_forever() -> None:
    """high_water was an all-time max() that never decayed."""
    client = _client()
    key = client._budget_key("critique", _payload(), SERVED)

    def record(consumed: int) -> None:
        client._record_budget_observation(
            payload=_payload(), operation="critique", usage=_usage(consumed),
            content="ok", finish_reason="stop", response_model=SERVED,
        )

    record(80_000)
    for _ in range(19):
        record(9_000)

    demand, samples = client._history_demand(key)
    assert samples == 20
    assert demand < 80_000, "the outlier still defines the floor"
    assert demand >= 9_000


def test_history_decays_once_the_window_rolls_past_the_outlier() -> None:
    client = _client()
    key = client._budget_key("critique", _payload(), SERVED)

    def record(consumed: int) -> None:
        client._record_budget_observation(
            payload=_payload(), operation="critique", usage=_usage(consumed),
            content="ok", finish_reason="stop", response_model=SERVED,
        )

    record(80_000)
    for _ in range(int(client.config.llm.recovery.history_window)):
        record(9_000)

    demand, samples = client._history_demand(key)
    assert samples == client.config.llm.recovery.history_window
    assert demand == 9_000, "the outlier should have rolled out of the window"


def test_history_window_is_never_exceeded() -> None:
    client = _client()
    key = client._budget_key("critique", _payload(), SERVED)
    for i in range(100):
        client._record_budget_observation(
            payload=_payload(), operation="critique", usage=_usage(1_000 + i),
            content="ok", finish_reason="stop", response_model=SERVED,
        )
    _, samples = client._history_demand(key)
    assert samples == client.config.llm.recovery.history_window


# ===========================================================================
# 8.6 — no futile identical retry at the ceiling
# ===========================================================================

def test_at_ceiling_length_failure_switches_to_the_fallback_model() -> None:
    """Chunk 5 retried at exactly 85,000 three times with a byte-identical
    prompt, at 11-14 minutes each. That cannot produce a different outcome."""
    client = _client(model="fallback-model")
    original = _payload("short prompt", max_tokens=85_000)

    payload, calculation = client._recovery_payload(
        original,
        attempt=2,
        max_attempts=4,
        failure_reason="length",
        operation="critique",
        failure_usage=_usage(85_000, reasoning=85_000),
        failure_content="",
    )
    assert payload["model"] == "fallback-model"
    assert calculation["strategy"] == "ceiling_exhausted_fallback_model"


def test_at_ceiling_length_failure_stops_when_no_fallback_is_configured() -> None:
    """For critique the honest outcome is to stop and let the existing
    qa_unavailable path mark the chunk for review - ~40 minutes sooner."""
    from tarjomeh.core.llm_client import TruncatedCompletionError

    client = _client(model="")
    original = _payload("short prompt", max_tokens=85_000)

    with pytest.raises(TruncatedCompletionError):
        client._recovery_payload(
            original,
            attempt=2,
            max_attempts=4,
            failure_reason="length",
            operation="critique",
            failure_usage=_usage(85_000, reasoning=85_000),
            failure_content="",
        )


def test_a_below_ceiling_length_failure_still_escalates_normally() -> None:
    """Within-request growth is useful and must be preserved."""
    client = _client()
    original = _payload("short prompt", max_tokens=12_000)
    payload, calculation = client._recovery_payload(
        original,
        attempt=1,
        max_attempts=4,
        failure_reason="length",
        operation="critique",
        failure_usage=_usage(12_000, reasoning=11_000),
        failure_content="",
    )
    assert payload["max_tokens"] > 12_000
    assert calculation.get("strategy") != "ceiling_exhausted_stop"


def test_a_non_length_failure_at_the_ceiling_is_not_treated_as_exhausted() -> None:
    """A transport error at the cap is retryable; only truncation is terminal."""
    client = _client(model="")
    original = _payload("short prompt", max_tokens=85_000)
    payload, calculation = client._recovery_payload(
        original,
        attempt=1,
        max_attempts=4,
        failure_reason="transport",
        operation="critique",
        failure_usage={},
        failure_content="",
    )
    assert calculation.get("strategy") != "ceiling_exhausted_stop"
    assert payload["max_tokens"] >= 85_000 or payload["max_tokens"] > 0
