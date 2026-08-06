from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

import httpx

from tarjomeh.context.book_researcher import BookResearcher
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import LLMClient
from tarjomeh.core.pipeline import sanitize_document_protocol_artifacts
from tarjomeh.core.structured_output import parse_structured_output
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph


def _response(content: str) -> Mock:
    response = Mock()
    response.status_code = 200
    response.headers = {}
    response.raise_for_status = Mock()
    response.text = json.dumps({
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": content},
        }],
        "usage": {},
    })
    return response


class TestProviderCompatibility(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TarjomehConfig()
        self.config.llm.openrouter.api_keys = ["key"]
        self.config.llm.openrouter.api_base = "http://172.17.0.1:20128/v1"
        self.config.retry.base_delay = 0
        self.config.retry.max_delay = 0
        self.config.retry.jitter = False
        self.client = LLMClient(self.config)

    def tearDown(self) -> None:
        self.client.close()

    def test_transport_defaults_and_9router_contract(self) -> None:
        _, headers, payload = self.client._prepare_request([
            {"role": "user", "content": "Translate."}
        ])
        self.assertTrue(payload["stream"])
        self.assertEqual(headers["X-9Router-Token-Saver"], "off")
        self.assertEqual(self.config.llm.transport.read_timeout_seconds, 900.0)

    def test_visible_answer_drops_leading_reasoning_wrapper(self) -> None:
        self.client._client.post = lambda *args, **kwargs: _response(
            "<think>private reasoning</think>ترجمه نهایی"
        )
        events = []
        self.client.set_attempt_observer(events.append)
        result = self.client.complete(
            messages=[{"role": "user", "content": "Translate."}],
            _operation="translation",
        )
        self.assertEqual(result, "ترجمه نهایی")
        self.assertEqual(
            events[0]["response_normalization"]["removed_reasoning_wrappers"],
            1,
        )

    def test_openai_sse_keeps_reasoning_out_of_visible_content(self) -> None:
        stream = "\n".join([
            'data: {"model":"judge","choices":[{"delta":{"reasoning_content":"private"}}]}',
            'data: {"choices":[{"delta":{"content":"{\\"score\\":9}"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ])
        parsed = LLMClient._parse_response_json(stream)
        message = parsed["choices"][0]["message"]
        self.assertEqual(message["content"], '{"score":9}')
        self.assertEqual(message["reasoning_content"], "private")

    def test_anthropic_sse_is_normalized(self) -> None:
        stream = "\n".join([
            'event: message_start',
            'data: {"type":"message_start","message":{"id":"m1","model":"model-a","usage":{"input_tokens":10}}}',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"private"}}',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"visible"}}',
            'event: message_delta',
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}',
            'event: message_stop',
            'data: {"type":"message_stop"}',
        ])
        parsed = LLMClient._parse_response_json(stream)
        message = parsed["choices"][0]["message"]
        self.assertEqual(message["content"], "visible")
        self.assertEqual(message["reasoning_content"], "private")
        self.assertEqual(parsed["usage"]["total_tokens"], 14)

    def test_anthropic_max_tokens_is_canonical_length(self) -> None:
        parsed = LLMClient._parse_response_json(json.dumps({
            "model": "model-a",
            "content": [{"type": "text", "text": "partial"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 2, "output_tokens": 5},
        }))
        self.assertEqual(parsed["choices"][0]["finish_reason"], "length")

    def test_read_timeout_has_bounded_unknown_outcome_retry(self) -> None:
        self.config.llm.recovery.max_attempts = 4
        calls = 0

        def post(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("stream became inactive")

        self.client._client.post = post
        with self.assertRaises(httpx.ReadTimeout):
            self.client.complete(
                messages=[{"role": "user", "content": "Judge."}],
                _operation="critique",
            )
        self.assertEqual(calls, 2)

    def test_structured_parser_recovers_json_after_protocol_text(self) -> None:
        parsed = parse_structured_output(
            "<think>work</think>Result:\n```json\n{\"terms\": []}\n```",
            expected=dict,
        )
        self.assertEqual(parsed, {"terms": []})

    def test_export_guard_removes_only_safe_leading_wrapper(self) -> None:
        document = TranslatedDocument(
            title="test",
            paragraphs=[TranslatedParagraph(
                index=0,
                source_text="Source",
                translated_text="<think>private</think>ترجمه",
            )],
        )
        report = sanitize_document_protocol_artifacts(document)
        self.assertEqual(document.paragraphs[0].translated_text, "ترجمه")
        self.assertEqual(report["safe_edit_count"], 1)
        self.assertEqual(report["remaining_artifact_count"], 0)


class TestResearchEvidencePreservation(unittest.IsolatedAsyncioTestCase):
    async def test_all_synthesis_failures_return_partial_data_not_exception(self) -> None:
        class FailingLLM:
            def set_operation(self, operation):
                self.operation = operation

            async def chat(self, prompt):
                raise ValueError("malformed structured response")

        researcher = BookResearcher.__new__(BookResearcher)
        researcher.config = TarjomehConfig()
        researcher.llm_client = FailingLLM()
        data, used_batches, error = await researcher._synthesise(
            "Book",
            "Author",
            "Excerpt",
            [{"title": "Source", "snippet": "Evidence", "url": "https://example.test"}],
            allow_follow_ups=False,
        )
        self.assertTrue(used_batches)
        self.assertEqual(data["terms"], [])
        self.assertIn("malformed structured response", error)
