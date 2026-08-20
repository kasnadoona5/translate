"""Regression tests for the independent audit of the remediation branch."""

from __future__ import annotations

import asyncio
import threading

import pytest

from tarjomeh.context.search_providers import (
    BaseSearchProvider,
    SearchProviderChain,
    SearchResult,
)
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import (
    ContextWindowExceededError,
    LLMClient,
    TruncatedCompletionError,
)
from tarjomeh.core.pipeline import TranslationPipeline
from tarjomeh.quality.integrity import PostEditIntegrityGate
from tarjomeh.quality.structure_audit import announced_counts, ordinal_sequence_length


def _budget_client(*, fallback: str = "") -> LLMClient:
    config = TarjomehConfig()
    config.llm.model = "combo1"
    config.llm.recovery.model = fallback
    client = LLMClient.__new__(LLMClient)
    client.config = config
    client._budget_history = {}
    client._budget_lock = threading.Lock()
    client._route_served_model = {}
    return client


def _payload(max_tokens: int = 50_000, *, model: str = "combo1") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Review this passage."}],
        "max_tokens": max_tokens,
    }


def _length_usage(tokens: int) -> dict:
    return {
        "completion_tokens": tokens,
        "completion_tokens_details": {"reasoning_tokens": tokens},
    }


def test_pipeline_reuses_one_async_loop_and_propagates_context() -> None:
    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline._async_loop = None
    pipeline._async_thread = None
    pipeline._async_loop_lock = threading.Lock()
    pipeline._async_loop_ready = threading.Event()
    pipeline._closed = False
    client = LLMClient(TarjomehConfig())
    pipeline.llm_client = client
    pipeline.critic_client = client

    client.set_operation("audit_probe")

    async def identities() -> tuple[int, int, str]:
        return (
            id(asyncio.get_running_loop()),
            id(client._aclient),
            client._operation.get(),
        )

    try:
        observed = [pipeline._run_async(identities()) for _ in range(5)]
        assert len(set(observed)) == 1
        assert observed[0][2] == "audit_probe"
        assert len(client._async_clients) == 1
        assert pipeline._async_thread is not None
        assert pipeline._async_thread.is_alive()
    finally:
        pipeline.close()

    assert pipeline._async_thread is not None
    assert not pipeline._async_thread.is_alive()
    assert not client._async_clients


def test_length_recovery_uses_the_previous_attempt_payload() -> None:
    client = _budget_client()
    first = _payload(50_000)
    second, _ = client._recovery_payload(
        first,
        attempt=1,
        max_attempts=4,
        failure_reason="length",
        operation="critique",
        failure_usage=_length_usage(50_000),
    )
    assert second["max_tokens"] > first["max_tokens"]

    third, _ = client._recovery_payload(
        second,
        attempt=2,
        max_attempts=4,
        failure_reason="length",
        operation="critique",
        failure_usage=_length_usage(second["max_tokens"]),
    )
    assert third["max_tokens"] > second["max_tokens"]

    with pytest.raises(TruncatedCompletionError):
        client._recovery_payload(
            third,
            attempt=3,
            max_attempts=4,
            failure_reason="length",
            operation="critique",
            failure_usage=_length_usage(third["max_tokens"]),
        )


def test_ceiling_allows_fallback_then_one_distinct_final_strategy() -> None:
    client = _budget_client(fallback="independent-judge")
    capped = _payload(85_000)
    fallback, evidence = client._recovery_payload(
        capped,
        attempt=2,
        max_attempts=4,
        failure_reason="length",
        operation="critique",
        failure_usage=_length_usage(85_000),
    )
    assert fallback["model"] == "independent-judge"
    assert evidence["strategy"] == "ceiling_exhausted_fallback_model"

    final, final_evidence = client._recovery_payload(
        fallback,
        attempt=3,
        max_attempts=4,
        failure_reason="length",
        operation="critique",
        failure_usage=_length_usage(85_000),
    )
    assert final["model"] == "independent-judge"
    assert final["reasoning"]["effort"] == "none"
    assert final_evidence["strategy"] == "ceiling_exhausted_changed_contract"


def test_preflight_refuses_an_impossible_context_before_network_io() -> None:
    client = _budget_client()
    client.count_tokens = lambda _text: 200_000
    with pytest.raises(ContextWindowExceededError):
        client._preflight_payload(
            _payload(),
            operation="critique",
            expected_output_tokens=2_000,
        )


def test_combo_route_reuses_the_actual_served_model_history() -> None:
    client = _budget_client()
    request = _payload()
    client._record_budget_observation(
        payload=request,
        operation="critique",
        usage=_length_usage(32_000),
        content="valid response",
        finish_reason="stop",
        response_model="deepseek-v4-flash",
    )
    _prepared, evidence = client._preflight_payload(
        request,
        operation="critique",
        expected_output_tokens=2_000,
    )
    assert evidence["history_samples"] == 1
    assert evidence["reasoning_evidence"] == "served_model_operation_p90"


def test_live_credentials_rehydrate_a_redacted_saved_job() -> None:
    live = TarjomehConfig()
    live.llm.openrouter.api_keys = ["translator-secret"]
    live.llm.critic.enabled = True
    live.llm.critic.api_keys = ["critic-secret"]
    saved = live.to_dict(redact_secrets=True)

    restored = TarjomehConfig.from_dict(saved, credential_source=live)
    assert restored.llm.openrouter.api_keys == ["translator-secret"]
    assert restored.llm.critic.api_keys == ["critic-secret"]


class _CountingProvider(BaseSearchProvider):
    name = "counting"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def search(self, query: str) -> list[SearchResult]:
        self.calls += 1
        await asyncio.sleep(0.01)
        return [SearchResult(title=query, url="https://example.test", snippet="ok")]


def test_concurrent_identical_searches_share_one_paid_query() -> None:
    provider = _CountingProvider()
    chain = SearchProviderChain([provider], max_retries=0, query_budget=3)

    async def run() -> list[list[SearchResult]]:
        return await asyncio.gather(
            chain.search("same query"),
            chain.search("  SAME   query "),
            chain.search("same query"),
        )

    results = asyncio.run(run())
    assert provider.calls == 1
    assert chain.queries_used == 1
    assert all(result and result[0].snippet == "ok" for result in results)


def test_source_grounded_repetition_is_not_rejected_as_edit_corruption() -> None:
    source_clause = "the state shapes institutions through relations of political power"
    target_clause = (
        "el estado organiza instituciones mediante relaciones duraderas "
        "de poder politico"
    )
    source = f"{source_clause}. {source_clause}."
    previous = f"{target_clause}."
    candidate = f"{target_clause}. {target_clause}."

    result = PostEditIntegrityGate().evaluate(
        source,
        candidate,
        previous=previous,
        stage="refinement",
    )
    classifications = {finding.check_id for finding in result.findings}
    assert "duplicate_span_introduced" not in classifications


def test_structure_markers_ignore_punctuation_and_unrelated_numbers() -> None:
    persian = (
        "\u0646\u062e\u0633\u062a\u060c \u0627\u0644\u0641\u061b "
        "\u062f\u0648\u0645\u060c \u0628\u061b "
        "\u0633\u0648\u0645\u060c \u067e."
    )
    assert ordinal_sequence_length(persian) == 3
    assert announced_counts("The two authors disagree. First a; second b; third c.") == []
