"""Regressions for provider-neutral transport and memory reconciliation."""

from __future__ import annotations

import json

import httpx

from tarjomeh.context.search_providers import SearchResult, rank_search_results
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import LLMClient
from tarjomeh.core.transport import (
    IncrementalCompletionDecoder,
    decode_response_text,
)
from tarjomeh.memory.bilingual_summary import BilingualSummary
from tarjomeh.memory.proper_nouns import ProperNouns
from tarjomeh.quality.integrity import PostEditIntegrityGate, classify_numbers


class _ChunkedBody(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    def __iter__(self):
        yield from self.chunks


def _sse_response(chunks: list[bytes]) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream", "x-request-id": "safe-id"},
        stream=_ChunkedBody(chunks),
    )


def test_incremental_decoder_supports_multiline_sse_data() -> None:
    decoder = IncrementalCompletionDecoder("openai_compatible")
    for line in (
        "data: {\"choices\":",
        "data: [{\"delta\":{\"content\":\"visible\"},\"finish_reason\":\"stop\"}]}",
        "",
        "data: [DONE]",
        "",
    ):
        decoder.feed_line(line)
    parsed, evidence = decoder.finish()
    assert parsed["choices"][0]["message"]["content"] == "visible"
    assert evidence["terminal_received"] is True
    assert evidence["data_event_count"] == 2
    assert evidence["event_types"] == {"done": 1, "openai_chunk": 1}
    assert len(evidence["response_sha256"]) == 64


def test_buffered_decoder_preserves_bounded_trailing_json_compatibility() -> None:
    parsed, evidence = decode_response_text(
        '{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}\n'
        "gateway diagnostic"
    )
    assert parsed["choices"][0]["message"]["content"] == "ok"
    assert evidence["trailing_json_repair"] is True


def test_incomplete_stream_replays_exact_request_before_recovery_changes() -> None:
    config = TarjomehConfig()
    config.llm.openrouter.api_keys = ["key-a", "key-b"]
    config.llm.openrouter.api_base = "http://172.17.0.1:20128/v1"
    config.llm.recovery.max_attempts = 2
    config.retry.base_delay = 0
    config.retry.max_delay = 0
    config.retry.jitter = False
    requests: list[dict] = []
    authorizations: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        authorizations.append(request.headers["authorization"])
        if len(requests) == 1:
            return _sse_response([
                b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            ])
        return _sse_response([
            b'data: {"choices":[{"delta":{"content":"complete"},',
            b'"finish_reason":"stop"}]}\n\n',
            b'data: [DONE]\n\n',
        ])

    client = LLMClient(config)
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    events: list[dict] = []
    client.set_attempt_observer(events.append)
    try:
        result = client.complete(
            messages=[{"role": "user", "content": "Judge this translation."}],
            _operation="critique",
        )
    finally:
        client.close()

    assert result == "complete"
    assert requests[0] == requests[1]
    assert authorizations == ["Bearer key-a", "Bearer key-a"]
    assert events[0]["request_payload_sha256"] == events[1]["request_payload_sha256"]
    assert events[0]["failure_reason"] == "incomplete_stream"
    assert events[0]["transport"]["response_identifiers"] == {
        "x-request-id": "safe-id"
    }
    assert events[1]["success"] is True


def test_curated_memory_replaces_auto_but_auto_cannot_replace_curated() -> None:
    nouns = ProperNouns()
    nouns.add_noun("central concept", "پیشنهاد خودکار", provenance="auto_extraction")
    preserved = nouns.add_noun(
        "Central Concept", "پیشنهاد دوم", provenance="incremental_extraction"
    )
    replaced = nouns.add_noun(
        "central concept", "برگردان مصوب", provenance="curated_glossary"
    )
    rejected = nouns.add_noun(
        "central concept", "پیشنهاد دیرهنگام", provenance="auto_extraction"
    )

    assert preserved["action"] == "preserved_higher_authority"
    assert replaced["action"] == "replaced_lower_authority"
    assert rejected["action"] == "preserved_higher_authority"
    assert nouns.serialize()["nouns"]["central concept"] == "برگردان مصوب"
    assert nouns.provenance_for("CENTRAL CONCEPT")["origin"] == "curated_glossary"

    restored = ProperNouns()
    restored.deserialize(nouns.serialize())
    assert restored.serialize()["nouns"] == nouns.serialize()["nouns"]
    assert restored.provenance_for("central concept")["superseded"]


def test_summary_deduplicates_protocol_provenance_without_rewriting_content() -> None:
    summary = BilingualSummary()
    summary.update(
        "## English Summary\n"
        "Provenance: translated content only.\n"
        "The argument develops.\n"
        "The argument develops."
    )
    assert summary.english_summary == "The argument develops."


def test_numeric_integrity_reports_source_roles_without_changing_verdict() -> None:
    roles = classify_numbers("Chapter 4 cites Scholar (2016: 33-71) and 40 percent.")
    assert roles["structural"]["4"] == 1
    assert roles["citation"]["2016"] == 1
    assert roles["prose"]["40%"] == 1

    result = PostEditIntegrityGate().evaluate(
        "The study appeared in 2016 and included 40 percent of cases.",
        "این پژوهش ۴۰ درصد موارد را دربر گرفت.",
    )
    finding = next(item for item in result.blocking if item.check_id == "numbers_missing")
    assert finding.details["missing_by_role"] == {"citation": ["2016"]}


def test_book_identity_requires_author_support_for_partial_title_match() -> None:
    results = [
        SearchResult(
            title="Institutions Across Regions",
            url="https://example.test/wrong-book",
            snippet="A different book by another researcher.",
        ),
        SearchResult(
            title="Institutions Across Time and Space",
            url="https://publisher.test/right-book",
            snippet="The study by Lena Orlov examines institutional change.",
        ),
    ]
    accepted, audit = rank_search_results(
        "Institutions Across Time and Space Lena Orlov review",
        results,
        identity="Institutions Across Time and Space",
        title="Institutions Across Time and Space",
        author="Lena Orlov",
    )
    assert accepted == [results[1]]
    assert "author_identity_mismatch" in audit[0]["reasons"]
