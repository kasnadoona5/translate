"""Provider-neutral HTTP response decoding for LLM completions.

The translation pipeline speaks one internal OpenAI-shaped completion contract.
This module owns only the wire boundary: ordinary JSON plus OpenAI- and
Anthropic-compatible server-sent events (SSE).  It deliberately does not alter
request prompts, model settings, retry policy, or translated content.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import httpx


_PROFILE_NAMES = {
    "9router",
    "openrouter",
    "opencode",
    "openai_compatible",
    "anthropic_compatible",
}
_REQUEST_ID_HEADERS = (
    "x-request-id",
    "request-id",
    "x-openrouter-generation-id",
    "cf-ray",
)


class TransportDecodeError(Exception):
    """A response body could not be accepted as a complete completion."""

    def __init__(
        self,
        kind: str,
        message: str,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.evidence = dict(evidence or {})


def detect_transport_profile(
    *,
    configured: str,
    provider: str,
    url: str,
) -> str:
    """Resolve a protocol profile without guessing from the selected model."""
    explicit = (configured or "auto").strip().casefold()
    if explicit != "auto":
        return explicit if explicit in _PROFILE_NAMES else "openai_compatible"

    parsed = urllib.parse.urlparse(url)
    endpoint = f"{parsed.netloc}{parsed.path}".casefold()
    if "9router" in endpoint or parsed.port == 20128:
        return "9router"
    if "openrouter.ai" in endpoint:
        return "openrouter"
    if "opencode" in endpoint:
        return "opencode"
    if provider.casefold() == "anthropic":
        return "anthropic_compatible"
    return "openai_compatible"


def _canonical_finish_reason(value: Any) -> Any:
    if value in {"max_tokens", "max_output_tokens"}:
        return "length"
    if value in {"end_turn", "stop_sequence"}:
        return "stop"
    return value


def _append_text(value: Any, destination: list[str]) -> None:
    """Append visible/reasoning text from common compatible delta shapes."""
    if isinstance(value, str):
        destination.append(value)
        return
    if not isinstance(value, list):
        return
    for block in value:
        if not isinstance(block, dict):
            continue
        text = block.get("text", block.get("content", ""))
        if isinstance(text, str):
            destination.append(text)


@dataclass
class _CompletionAccumulator:
    profile: str
    result: dict[str, Any] = field(default_factory=dict)
    content_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    terminal_received: bool = False
    data_event_count: int = 0
    parsed_event_count: int = 0
    provider_error: str = ""
    dialects: set[str] = field(default_factory=set)
    event_types: dict[str, int] = field(default_factory=dict)

    def consume(self, event_name: str, data: str) -> None:
        self.data_event_count += 1
        if data.strip() == "[DONE]":
            self.terminal_received = True
            self.dialects.add("openai")
            self.event_types["done"] = self.event_types.get("done", 0) + 1
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            snippet = data[:300].replace("\n", "\\n")
            raise TransportDecodeError(
                "malformed_sse",
                f"LLM endpoint returned malformed SSE data: {snippet!r}",
            ) from exc
        if not isinstance(chunk, dict):
            raise TransportDecodeError(
                "malformed_sse",
                "LLM endpoint returned a non-object SSE data event.",
            )

        self.parsed_event_count += 1
        if chunk.get("error"):
            self.provider_error = str(chunk["error"])
            raise TransportDecodeError(
                "provider_error",
                f"LLM SSE stream reported an error: {chunk['error']}",
            )

        chunk_type = str(chunk.get("type", event_name or ""))
        event_type = chunk_type or "openai_chunk"
        self.event_types[event_type] = self.event_types.get(event_type, 0) + 1
        if chunk_type.startswith(("message_", "content_block_")):
            self.dialects.add("anthropic")
        if chunk_type == "message_start":
            message = chunk.get("message", {})
            if isinstance(message, dict):
                for key in ("id", "model"):
                    if message.get(key) and key not in self.result:
                        self.result[key] = message[key]
                start_usage = message.get("usage")
                if isinstance(start_usage, dict):
                    self.usage.update(start_usage)
            return
        if chunk_type == "content_block_start":
            block = chunk.get("content_block", {})
            if isinstance(block, dict):
                destination = (
                    self.reasoning_parts
                    if block.get("type") in {"thinking", "reasoning"}
                    else self.content_parts
                )
                _append_text(block.get("text", block.get("thinking", "")), destination)
            return
        if chunk_type == "content_block_delta":
            delta = chunk.get("delta", {})
            if isinstance(delta, dict):
                if delta.get("type") in {"thinking_delta", "reasoning_delta"}:
                    _append_text(
                        delta.get("thinking", delta.get("reasoning", "")),
                        self.reasoning_parts,
                    )
                else:
                    _append_text(delta.get("text", ""), self.content_parts)
            return
        if chunk_type == "message_delta":
            delta = chunk.get("delta", {})
            if isinstance(delta, dict) and delta.get("stop_reason") is not None:
                self.finish_reason = _canonical_finish_reason(delta["stop_reason"])
            delta_usage = chunk.get("usage")
            if isinstance(delta_usage, dict):
                self.usage.update(delta_usage)
            return
        if chunk_type == "message_stop":
            self.terminal_received = True
            return

        self.dialects.add("openai")
        for key in ("id", "object", "created", "model", "system_fingerprint"):
            if key in chunk and key not in self.result:
                self.result[key] = chunk[key]
        raw_usage = chunk.get("usage")
        if isinstance(raw_usage, dict):
            self.usage.update(raw_usage)

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        choice = choices[0]
        if not isinstance(choice, dict):
            return
        delta = choice.get("delta")
        message = choice.get("message")
        payload = delta if isinstance(delta, dict) else message
        if isinstance(payload, dict):
            _append_text(payload.get("content"), self.content_parts)
            for key in ("reasoning_content", "reasoning", "analysis"):
                _append_text(payload.get(key), self.reasoning_parts)
        if choice.get("finish_reason") is not None:
            self.finish_reason = _canonical_finish_reason(choice["finish_reason"])

    def finish(self) -> tuple[dict[str, Any], dict[str, Any]]:
        evidence = {
            "transport_profile": self.profile,
            "protocol": (
                "mixed_sse"
                if len(self.dialects) > 1
                else f"{next(iter(self.dialects), 'unknown')}_sse"
            ),
            "event_count": self.parsed_event_count,
            "data_event_count": self.data_event_count,
            "terminal_received": bool(
                self.terminal_received or self.finish_reason is not None
            ),
            "content_bytes": len("".join(self.content_parts).encode("utf-8")),
            "reasoning_bytes": len("".join(self.reasoning_parts).encode("utf-8")),
            "event_types": dict(sorted(self.event_types.items())),
        }
        if not self.data_event_count:
            raise TransportDecodeError(
                "empty_stream",
                "LLM endpoint returned an empty SSE chat-completion stream.",
                evidence,
            )
        if not self.terminal_received and self.finish_reason is None:
            raise TransportDecodeError(
                "incomplete_stream",
                "LLM SSE stream ended without [DONE] or a finish reason.",
                evidence,
            )

        if "input_tokens" in self.usage and "prompt_tokens" not in self.usage:
            self.usage["prompt_tokens"] = self.usage.get("input_tokens", 0)
        if "output_tokens" in self.usage and "completion_tokens" not in self.usage:
            self.usage["completion_tokens"] = self.usage.get("output_tokens", 0)
        if self.usage:
            self.usage.setdefault(
                "total_tokens",
                int(self.usage.get("prompt_tokens", 0) or 0)
                + int(self.usage.get("completion_tokens", 0) or 0),
            )
        self.result["choices"] = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "".join(self.content_parts),
                "reasoning_content": "".join(self.reasoning_parts),
            },
            "finish_reason": self.finish_reason,
        }]
        if self.usage:
            self.result["usage"] = self.usage
        return self.result, evidence


class IncrementalCompletionDecoder:
    """Incrementally assemble SSE events or a compatible ordinary JSON body."""

    def __init__(self, profile: str) -> None:
        self.profile = profile
        self.mode = ""
        self.event_name = ""
        self.data_lines: list[str] = []
        self.json_lines: list[str] = []
        self.response_bytes = 0
        self.line_count = 0
        self._response_hasher = hashlib.sha256()
        self._accumulator = _CompletionAccumulator(profile)

    def feed_line(self, raw_line: str) -> None:
        self.line_count += 1
        encoded_line = raw_line.encode("utf-8") + b"\n"
        self.response_bytes += len(encoded_line)
        self._response_hasher.update(encoded_line)
        line = raw_line.rstrip("\r")
        if not self.mode and line:
            self.mode = (
                "sse"
                if line.startswith(("data:", "event:", ":", "id:", "retry:"))
                else "json"
            )
        if self.mode == "json":
            self.json_lines.append(line)
            return
        if not self.mode:
            return
        if not line:
            self._dispatch_event()
            return
        if line.startswith(":"):
            return
        field, separator, value = line.partition(":")
        if not separator:
            # The SSE specification permits unknown fields; a bare line is not
            # a valid field and usually signals gateway framing corruption.
            raise TransportDecodeError(
                "malformed_sse",
                "LLM endpoint returned an invalid SSE field.",
                self.evidence(),
            )
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            # Some compatible gateways omit the spec's blank separator between
            # events. A new named event is still an unambiguous boundary.
            if self.data_lines:
                self._dispatch_event()
            self.event_name = value
        elif field == "data":
            # OpenAI-compatible streams are also commonly emitted as adjacent
            # one-line JSON data fields without blank separators. Preserve real
            # multiline SSE data when the accumulated value is not complete.
            if self.data_lines and self._complete_data_value(self.data_lines):
                self._dispatch_event()
            self.data_lines.append(value)
        elif field in {"id", "retry"}:
            return
        # Unknown named fields are ignored as required by the SSE specification.

    @staticmethod
    def _complete_data_value(lines: list[str]) -> bool:
        value = "\n".join(lines).strip()
        if value == "[DONE]":
            return True
        try:
            json.loads(value)
        except json.JSONDecodeError:
            return False
        return True

    def _dispatch_event(self) -> None:
        if self.data_lines:
            self._accumulator.consume(self.event_name, "\n".join(self.data_lines))
        self.event_name = ""
        self.data_lines = []

    def evidence(self) -> dict[str, Any]:
        return {
            "transport_profile": self.profile,
            "protocol": f"{self.mode or 'empty'}_response",
            "response_bytes": self.response_bytes,
            "line_count": self.line_count,
            "event_count": self._accumulator.parsed_event_count,
            "data_event_count": self._accumulator.data_event_count,
            "terminal_received": bool(
                self._accumulator.terminal_received
                or self._accumulator.finish_reason is not None
            ),
            "content_bytes": len(
                "".join(self._accumulator.content_parts).encode("utf-8")
            ),
            "reasoning_bytes": len(
                "".join(self._accumulator.reasoning_parts).encode("utf-8")
            ),
            "event_types": dict(sorted(self._accumulator.event_types.items())),
            "response_sha256": self._response_hasher.hexdigest(),
        }

    def finish(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.mode == "sse":
            self._dispatch_event()
            try:
                data, evidence = self._accumulator.finish()
            except TransportDecodeError as exc:
                exc.evidence = {**self.evidence(), **exc.evidence}
                raise
            return data, {**self.evidence(), **evidence}

        text = "\n".join(self.json_lines).strip()
        evidence = self.evidence()
        evidence["protocol"] = "json"
        if not text:
            raise TransportDecodeError(
                "empty_response",
                "LLM endpoint returned an empty response body.",
                evidence,
            )
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            # Preserve the previous client's compatibility with gateways that
            # append non-JSON diagnostics after one complete response object.
            # A genuinely truncated object still fails this bounded repair.
            last_brace = text.rfind("}")
            try:
                data = json.loads(text[:last_brace + 1]) if last_brace >= 0 else None
            except json.JSONDecodeError:
                data = None
            if not isinstance(data, dict):
                snippet = text[:300].replace("\n", "\\n")
                raise TransportDecodeError(
                    "malformed_json",
                    f"LLM endpoint returned malformed JSON response: {snippet!r}",
                    evidence,
                ) from exc
            evidence["trailing_json_repair"] = True
            evidence["trailing_bytes_ignored"] = len(
                text[last_brace + 1:].encode("utf-8")
            )
        if not isinstance(data, dict):
            raise TransportDecodeError(
                "malformed_json",
                "LLM endpoint returned a non-object response.",
                evidence,
            )
        return data, evidence


def decode_response_text(
    response_text: str,
    *,
    profile: str = "openai_compatible",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode a buffered response through the same standards-aware parser."""
    decoder = IncrementalCompletionDecoder(profile)
    for line in response_text.splitlines():
        decoder.feed_line(line)
    return decoder.finish()


def _response_metadata(response: httpx.Response) -> dict[str, Any]:
    identifiers = {
        name: value
        for name in _REQUEST_ID_HEADERS
        if (value := response.headers.get(name))
    }
    return {
        "response_content_type": response.headers.get("content-type", "")[:160],
        "response_identifiers": identifiers,
    }


def stream_completion_sync(
    client: httpx.Client,
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    profile: str,
) -> tuple[httpx.Response, dict[str, Any], dict[str, Any]]:
    """Read a completion incrementally without buffering an SSE body first."""
    started = time.monotonic()
    first_line_at: float | None = None
    decoder = IncrementalCompletionDecoder(profile)
    with client.stream("POST", url, headers=headers, json=payload) as response:
        response.raise_for_status()
        try:
            for line in response.iter_lines():
                if first_line_at is None and line:
                    first_line_at = time.monotonic()
                decoder.feed_line(line)
            data, evidence = decoder.finish()
        except TransportDecodeError as exc:
            exc.evidence = {
                **decoder.evidence(),
                **_response_metadata(response),
                **exc.evidence,
            }
            raise
        evidence.update(_response_metadata(response))
    evidence["first_event_seconds"] = (
        round(first_line_at - started, 3) if first_line_at is not None else None
    )
    return response, data, evidence


async def stream_completion_async(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    profile: str,
) -> tuple[httpx.Response, dict[str, Any], dict[str, Any]]:
    """Asynchronous counterpart to :func:`stream_completion_sync`."""
    started = time.monotonic()
    first_line_at: float | None = None
    decoder = IncrementalCompletionDecoder(profile)
    async with client.stream("POST", url, headers=headers, json=payload) as response:
        response.raise_for_status()
        try:
            async for line in response.aiter_lines():
                if first_line_at is None and line:
                    first_line_at = time.monotonic()
                decoder.feed_line(line)
            data, evidence = decoder.finish()
        except TransportDecodeError as exc:
            exc.evidence = {
                **decoder.evidence(),
                **_response_metadata(response),
                **exc.evidence,
            }
            raise
        evidence.update(_response_metadata(response))
    evidence["first_event_seconds"] = (
        round(first_line_at - started, 3) if first_line_at is not None else None
    )
    return response, data, evidence
