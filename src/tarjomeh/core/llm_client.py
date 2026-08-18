"""LLM Client for the Tarjomeh translation system.

Handles API key rotation, exponential backoff with jitter on 429/5xx errors,
token counting via tiktoken, usage tracking, and OpenAI-compatible requests
to OpenRouter and Ollama.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import threading
import time
from typing import Any

import httpx
import tiktoken

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.structured_output import normalize_model_text
from tarjomeh.core.transport import (
    TransportDecodeError,
    decode_response_text,
    detect_transport_profile,
    stream_completion_async,
    stream_completion_sync,
)

logger = logging.getLogger(__name__)


_SCHEMA_REPAIR_OPERATIONS = {
    "critique_json_repair",
    "refinement_json_repair",
    "translation_paragraph_repair",
}

# Structured helpers may benefit from normal model reasoning on their first
# attempt. If their bounded output contract fails, recovery disables reasoning
# so the model can spend the remaining allowance on valid JSON.
_STRUCTURED_HELPER_OPERATIONS = {
    "auto_term_extraction",
    "book_research",
    "book_research_initial",
    "book_research_followup",
    "book_research_batch",
    "book_research_synthesis",
    "proper_noun_initial",
    "proper_noun_incremental",
    "web_context_term_detection",
}
_JSON_OBJECT_HELPER_OPERATIONS = {
    "auto_term_extraction",
    "book_research",
    "book_research_initial",
    "book_research_followup",
    "book_research_batch",
    "book_research_synthesis",
}

# For these quality-bearing operations, a length-only retry must preserve the
# exact request contract. Attempt 2 may increase only max_tokens.
_UNCHANGED_SECOND_ATTEMPT_OPERATIONS = {
    "translation",
    "translation_split_recovery",
    "critique",
    "refinement",
}
_PRESERVE_REASONING_ON_ALL_LENGTH_RETRIES = {
    "translation",
    "translation_split_recovery",
}

_FULL_QUALITY_LENGTH_RECOVERY_OPERATIONS = {
    "translation",
    "translation_split_recovery",
    "critique",
    "refinement",
}
_QUALITY_RECOVERY_OPERATIONS = {
    "critique",
    "critique_json_repair",
    "refinement",
    "refinement_json_repair",
}
_TRANSPORT_FAILURE_REASONS = {
    "empty_stream",
    "empty_response",
    "incomplete_stream",
    "malformed_sse",
    "malformed_json",
    "provider_error",
    "read_timeout",
    "connect_timeout",
    "transport_error",
}


def _use_no_reasoning_recovery(
    *,
    operation: str,
    attempt: int,
    max_attempts: int,
    has_fallback: bool,
    final_expanded: bool,
) -> bool:
    """Select no-reasoning recovery only after a normal request fails."""
    if operation in _STRUCTURED_HELPER_OPERATIONS:
        return True
    if final_expanded or operation == "glossary_auto_correction":
        return True
    if operation not in _QUALITY_RECOVERY_OPERATIONS:
        return False
    if attempt == max_attempts - 1:
        return True
    # Avoid repeating the same low-reasoning model when no fallback exists.
    return attempt >= 2 and not has_fallback


class EmptyCompletionError(Exception):
    """Raised when the LLM returns an empty completion."""
    pass


class TruncatedCompletionError(Exception):
    """Raised when the completion was cut off at max_tokens (finish_reason='length')."""
    pass


class MalformedLLMResponseError(Exception):
    """Raised when an OpenAI-compatible endpoint returns malformed response JSON."""
    pass


class IncompleteCompletionError(Exception):
    """Raised when a provider explicitly reports an incomplete generation."""
    pass


def _map_transport_decode_error(exc: TransportDecodeError) -> Exception:
    """Map wire-level failures onto the client's established public errors."""
    if exc.kind in {"empty_stream", "empty_response"}:
        mapped: Exception = EmptyCompletionError(str(exc))
    elif exc.kind in {"incomplete_stream", "provider_error"}:
        mapped = IncompleteCompletionError(str(exc))
    else:
        mapped = MalformedLLMResponseError(str(exc))
    setattr(mapped, "transport_failure_kind", exc.kind)
    setattr(mapped, "transport_evidence", dict(exc.evidence))
    return mapped


class LLMClient:
    """OpenAI-compatible client for communicating with OpenRouter or Ollama.

    Provides sync and async interfaces, key rotation, retries, and token counting.
    """

    def __init__(self, config: TarjomehConfig) -> None:
        self.config = config
        self._timeout = self._http_timeout()
        self._client = httpx.Client(timeout=self._timeout)
        self._thread_local = threading.local()

        # Track usage
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_total_tokens = 0
        self.total_cost = 0.0

        # API key index for rotation
        self._api_key_index = 0
        self._api_key_lock = threading.Lock()
        self._attempt_observer: Any = None
        # Keyed (operation, transport profile, serving model); the value is a
        # bounded rolling list of observed completion sizes.
        self._budget_history: dict[tuple[str, str, str], dict[str, list[int]]] = {}
        self._budget_lock = threading.Lock()

        # _thread_local only ever exposes the CALLING thread's clients, so
        # aclose() could never reach a worker thread's AsyncClient and every
        # worker leaked its connection pool. Keep a process-wide registry.
        self._all_async_clients: list[httpx.AsyncClient] = []
        self._async_registry_lock = threading.Lock()

    @property
    def _aclient(self) -> httpx.AsyncClient:
        """Get or create an AsyncClient bound to this thread's active loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop_policy().get_event_loop()

        clients: dict[int, httpx.AsyncClient] = (
            getattr(self._thread_local, "clients", None) or {}
        )
        self._thread_local.clients = clients

        # Drop entries whose loop is gone. _run_async creates a fresh loop per
        # call, so without this the cache grows without bound.
        for stale_id, stale in list(clients.items()):
            if stale_id != id(loop) and stale.is_closed:
                clients.pop(stale_id, None)

        client = clients.get(id(loop))
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=self._timeout)
            clients[id(loop)] = client
            with self._async_registry_lock:
                self._all_async_clients.append(client)
        return client

    def _http_timeout(self) -> httpx.Timeout:
        transport = self.config.llm.transport
        return httpx.Timeout(
            connect=float(transport.connect_timeout_seconds),
            read=float(transport.read_timeout_seconds),
            write=float(transport.write_timeout_seconds),
            pool=float(transport.pool_timeout_seconds),
        )

    def close(self) -> None:
        """Close the sync HTTP client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close every async HTTP client this instance created, on any thread."""
        with self._async_registry_lock:
            clients = list(self._all_async_clients)
            self._all_async_clients.clear()
        for client in clients:
            if not client.is_closed:
                try:
                    await client.aclose()
                except Exception:
                    logger.debug("AsyncClient close failed", exc_info=True)
        if hasattr(self._thread_local, "clients"):
            self._thread_local.clients.clear()

    def _get_next_api_key(self) -> str:
        """Retrieve the next non-empty API key from the rotation list."""
        keys = self.config.llm.openrouter.api_keys
        if not keys:
            return ""
        # Filter empty keys
        valid_keys = [k for k in keys if k.strip()]
        if not valid_keys:
            return ""
        
        with self._api_key_lock:
            key = valid_keys[self._api_key_index]
            self._api_key_index = (self._api_key_index + 1) % len(valid_keys)
        return key

    def count_tokens(self, text: str) -> int:
        """Accurately count tokens in a string using tiktoken."""
        model_name = self.config.llm.model
        is_fallback = False
        try:
            # Clean OpenRouter names like "anthropic/claude-3-5-sonnet" -> "claude-3-5-sonnet"
            cleaned_model = model_name.split("/")[-1] if "/" in model_name else model_name
            try:
                encoding = tiktoken.encoding_for_model(cleaned_model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
                is_fallback = True
        except Exception as e:
            logger.debug("Failed to load tiktoken encoding for model %s: %s", model_name, e)
            encoding = tiktoken.get_encoding("cl100k_base")
            is_fallback = True

        raw_count = len(encoding.encode(text))
        if is_fallback:
            # Add a 5% safety margin for fallback token count
            return int(raw_count * 1.05) + 1
        return raw_count

    def _prompt_metrics(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return stable prompt evidence without persisting prompt contents."""
        messages = [
            message for message in payload.get("messages", [])
            if isinstance(message, dict)
        ]
        canonical = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prompt_text = "\n".join(str(message.get("content", "")) for message in messages)
        return {
            "prompt_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "prompt_characters": len(prompt_text),
            "message_count": len(messages),
            "estimated_prompt_tokens": self.count_tokens(prompt_text),
        }

    def _usage_evidence(
        self,
        usage: dict[str, Any],
        content: str = "",
    ) -> dict[str, int]:
        """Normalize provider usage counters onto the completion-token scale."""
        details = usage.get("completion_tokens_details") or {}
        if not isinstance(details, dict):
            details = {}
        completion = int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        )
        reported_reasoning = int(
            details.get("reasoning_tokens")
            or usage.get("reasoning_tokens")
            or 0
        )
        visible = self.count_tokens(content) if content else 0
        # Some OpenAI-compatible routers report reasoning in a converted token
        # scale larger than completion_tokens. Do not mix those units.
        normalized_reasoning = (
            min(reported_reasoning, completion)
            if reported_reasoning > 0 and completion > 0
            else max(0, reported_reasoning)
        )
        if normalized_reasoning <= 0 and completion > 0:
            normalized_reasoning = max(0, completion - visible)
        return {
            "completion_tokens": completion,
            "reported_reasoning_tokens": reported_reasoning,
            "reasoning_tokens": normalized_reasoning,
            "visible_tokens": visible,
        }

    def _budget_key(
        self,
        operation: str,
        payload: dict[str, Any],
        response_model: str = "",
    ) -> tuple[str, str, str]:
        """Key demand history by what actually served the request.

        payload["model"] is the configured model name, which in a 9router
        deployment is a COMBO name fronting many models. Keying on it merged
        every model behind that combo into one statistic, so one looping
        model's demand poisoned the budget every other model inherited.
        At preflight no response model is known yet, so this falls back to the
        configured name and a first call simply has no history.
        """
        profile = str(getattr(self.config.llm.transport, "profile", "auto"))
        served = (response_model or str(payload.get("model", ""))).strip()
        return (operation, profile, served)

    def _history_demand(self, key: tuple[str, str, str]) -> tuple[int, int]:
        """Return (p90 demand, sample count) over the retained window.

        Replaces an all-time high_water max() that never decayed: one
        exceptional call set the floor permanently. p90 tolerates a single
        long-but-legitimate response without letting it define the floor, and
        it decays as the window rolls.
        """
        with self._budget_lock:
            samples = list(self._budget_history.get(key, {}).get("samples", []))
        if not samples:
            return 0, 0
        ordered = sorted(samples)
        index = min(len(ordered) - 1, int(round(0.90 * (len(ordered) - 1))))
        return int(ordered[index]), len(ordered)

    def _record_budget_observation(
        self,
        *,
        payload: dict[str, Any],
        operation: str,
        usage: dict[str, Any],
        content: str,
        finish_reason: str | None,
        response_model: str = "",
    ) -> None:
        """Learn output demand only from completions that actually finished.

        A truncated or empty response is evidence the model looped, not
        evidence it needed a larger budget. Recording it at 1.5x the FAILED
        budget ratcheted the floor upward permanently and irreversibly:
        failing at 61,944 recorded 92,916, and failing at 85,000 recorded
        127,500.
        """
        if finish_reason != "stop":
            return
        if not (content or "").strip():
            return
        evidence = self._usage_evidence(usage, content)
        # completion_tokens includes reasoning, which is correct here: the
        # budget has to cover reasoning + answer. Sizing from visible output
        # alone would under-budget a reasoning model.
        consumed = max(evidence["completion_tokens"], evidence["visible_tokens"])
        if consumed <= 0:
            return
        key = self._budget_key(operation, payload, response_model)
        window = int(self.config.llm.recovery.history_window)
        with self._budget_lock:
            stats = self._budget_history.setdefault(key, {"samples": []})
            samples = stats["samples"]
            samples.append(consumed)
            del samples[: max(0, len(samples) - window)]

    def _answer_estimate(
        self,
        payload: dict[str, Any],
        expected_output_tokens: int,
    ) -> tuple[int, str]:
        if expected_output_tokens > 0:
            return int(expected_output_tokens), "source_token_estimate"
        prompt_tokens = self._prompt_metrics(payload)["estimated_prompt_tokens"]
        return max(512, int(prompt_tokens) // 3), "prompt_token_fallback"

    def _adaptive_ceiling(self, payload: dict[str, Any]) -> tuple[int, int]:
        recovery = self.config.llm.recovery
        prompt_tokens = self._prompt_metrics(payload)["estimated_prompt_tokens"]
        # How much answer room is GENUINELY left in the window. An upper bound,
        # never a floor: the previous max() reported more room than physically
        # exists (a 100k prompt has 29,024 tokens left, not 50,000), producing
        # requests the provider cannot serve, which were then retried. The old
        # trailing max(payload_max, ceiling) repeated the mistake by flooring
        # the result at whatever had already been requested, so the ceiling
        # could never bind.
        context_room = max(
            0,
            int(recovery.context_window_tokens)
            - prompt_tokens
            - int(recovery.context_safety_tokens),
        )
        ceiling = min(int(recovery.adaptive_max_tokens), context_room)
        return ceiling, prompt_tokens

    def _preflight_payload(
        self,
        original: dict[str, Any],
        operation: str,
        expected_output_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Size Attempt 1 without changing its prompt, model, or reasoning policy."""
        recovery = self.config.llm.recovery
        if not recovery.enabled or not recovery.predictive_first_attempt:
            return original, {}

        payload = dict(original)
        answer_estimate, answer_evidence = self._answer_estimate(
            payload, expected_output_tokens
        )
        historical_total, history_samples = self._history_demand(
            self._budget_key(operation, payload)
        )
        reasoning_estimate = max(
            int(recovery.bootstrap_reasoning_tokens),
            historical_total - answer_estimate,
        )
        reasoning_evidence = (
            "served_model_operation_p90" if historical_total else "bootstrap"
        )

        answer_headroom = math.ceil(answer_estimate * 1.35) + 512
        reasoning_headroom = math.ceil(reasoning_estimate * 1.20)
        calculated = math.ceil((answer_headroom + reasoning_headroom) * 1.25)
        original_max = int(payload.get("max_tokens", self.config.llm.max_tokens))
        ceiling, prompt_tokens = self._adaptive_ceiling(payload)
        requested = min(
            max(
                original_max,
                int(recovery.predictive_min_tokens),
                historical_total,
                calculated,
            ),
            ceiling,
        )
        payload["max_tokens"] = requested
        return payload, {
            "stage": "preflight",
            "answer_estimate_tokens": answer_estimate,
            "answer_evidence": answer_evidence,
            "reasoning_estimate_tokens": reasoning_estimate,
            "reasoning_evidence": reasoning_evidence,
            "history_samples": history_samples,
            "historical_demand_p90_tokens": historical_total,
            "answer_headroom_tokens": answer_headroom,
            "reasoning_headroom_tokens": reasoning_headroom,
            "uncertainty_multiplier": 1.25,
            "calculated_max_tokens": calculated,
            "estimated_prompt_tokens": prompt_tokens,
            "configured_ceiling": ceiling,
            "applied_max_tokens": requested,
            "ceiling_applied": requested < calculated,
        }

    def _prepare_request(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Prepare URL, headers, and payload for the LLM request."""
        provider = self.config.llm.provider.lower()
        
        # Prepend system prompt if provided
        final_messages = list(messages)
        if system_prompt:
            final_messages.insert(0, {"role": "system", "content": system_prompt})

        if provider == "openrouter":
            # Resolution order: config api_base → env var → OpenRouter default.
            api_base = (
                getattr(self.config.llm.openrouter, "api_base", "")
                or os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
            ).rstrip("/")
            url = f"{api_base}/chat/completions"
            key = self._get_next_api_key()
            headers = {
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": self.config.llm.openrouter.site_url,
                "X-Title": self.config.llm.openrouter.app_name,
                "Content-Type": "application/json",
            }
            if (
                self.config.llm.transport.bypass_9router_token_saver
                and self._is_9router_endpoint(api_base)
            ):
                headers["X-9Router-Token-Saver"] = "off"
            payload = {
                "model": self.config.llm.model,
                "messages": final_messages,
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens,
                "stream": bool(self.config.llm.transport.streaming),
            }
            if getattr(self.config.llm.openrouter, "exclude_reasoning", True):
                payload["reasoning"] = {"exclude": True}
        elif provider == "ollama":
            host = self.config.llm.ollama.host.rstrip("/")
            url = f"{host}/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.config.llm.ollama.model,
                "messages": final_messages,
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens,
                "stream": bool(self.config.llm.transport.streaming),
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Update payload with custom kwargs
        payload.update(kwargs)
        return url, headers, payload

    @staticmethod
    def _is_9router_endpoint(api_base: str) -> bool:
        lowered = api_base.casefold()
        return "9router" in lowered or ":20128" in lowered

    def _transport_profile(self, url: str) -> str:
        return detect_transport_profile(
            configured=str(getattr(self.config.llm.transport, "profile", "auto")),
            provider=self.config.llm.provider,
            url=url,
        )

    @staticmethod
    def _buffered_response_evidence(
        response: httpx.Response,
        response_text: str,
        profile: str,
        parsed_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        response_headers = (
            response.headers
            if isinstance(response.headers, (dict, httpx.Headers))
            else {}
        )
        identifiers = {
            name: value
            for name in (
                "x-request-id",
                "request-id",
                "x-openrouter-generation-id",
                "cf-ray",
            )
            if (value := response_headers.get(name))
        }
        return {
            **parsed_evidence,
            "transport_profile": profile,
            "response_bytes": len(response_text.encode("utf-8")),
            "response_content_type": str(
                response_headers.get("content-type", "")
            )[:160],
            "response_identifiers": identifiers,
            "first_event_seconds": None,
            "buffered_response": True,
        }

    def _request_sync(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> tuple[httpx.Response, dict[str, Any], dict[str, Any]]:
        """Execute one request using incremental transport in production."""
        profile = self._transport_profile(url)
        # Existing tests and embedders may replace ``post`` on this client
        # instance. Honour that explicit seam; a normal httpx.Client has no
        # instance-level post attribute and therefore uses true streaming.
        explicit_post_override = "post" in vars(self._client)
        try:
            if bool(payload.get("stream")) and not explicit_post_override:
                response, data, evidence = stream_completion_sync(
                    self._client,
                    url=url,
                    headers=headers,
                    payload=payload,
                    profile=profile,
                )
            else:
                response = self._client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                response_text = response.text
                data, parsed_evidence = decode_response_text(
                    response_text,
                    profile=profile,
                )
                evidence = self._buffered_response_evidence(
                    response, response_text, profile, parsed_evidence
                )
        except TransportDecodeError as exc:
            raise _map_transport_decode_error(exc) from exc
        return response, self._normalize_completion_envelope(data), evidence

    async def _request_async(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> tuple[httpx.Response, dict[str, Any], dict[str, Any]]:
        """Asynchronous request counterpart with the same decoding contract."""
        profile = self._transport_profile(url)
        try:
            if bool(payload.get("stream")):
                response, data, evidence = await stream_completion_async(
                    self._aclient,
                    url=url,
                    headers=headers,
                    payload=payload,
                    profile=profile,
                )
            else:
                response = await self._aclient.post(
                    url, headers=headers, json=payload
                )
                response.raise_for_status()
                response_text = response.text
                data, parsed_evidence = decode_response_text(
                    response_text,
                    profile=profile,
                )
                evidence = self._buffered_response_evidence(
                    response, response_text, profile, parsed_evidence
                )
        except TransportDecodeError as exc:
            raise _map_transport_decode_error(exc) from exc
        return response, self._normalize_completion_envelope(data), evidence

    def _update_usage(self, response: httpx.Response, response_json: dict[str, Any]) -> None:
        """Update token usage counters and cost metrics."""
        usage = response_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_total_tokens += total_tokens

        # Attempt to parse cost from response JSON or OpenRouter headers
        cost = response_json.get("cost")
        if cost is not None and isinstance(cost, (str, int, float)):
            try:
                self.total_cost += float(cost)
            except (ValueError, TypeError):
                pass
        else:
            if hasattr(response, "headers") and hasattr(response.headers, "get"):
                cost_str = response.headers.get("x-openrouter-cost")
                # Ensure we don't try to parse Mock objects
                if cost_str and isinstance(cost_str, (str, int, float)):
                    try:
                        self.total_cost += float(cost_str)
                    except ValueError:
                        pass

    @staticmethod
    def _parse_response_json(response_text: str) -> dict[str, Any]:
        """Parse JSON/SSE and return one canonical chat-completion envelope."""
        try:
            parsed, _evidence = decode_response_text(response_text)
        except TransportDecodeError as exc:
            raise _map_transport_decode_error(exc) from exc
        return LLMClient._normalize_completion_envelope(parsed)

    @staticmethod
    def _normalize_completion_envelope(data: dict[str, Any]) -> dict[str, Any]:
        """Normalize OpenAI- and Anthropic-shaped non-streaming responses."""
        if isinstance(data.get("choices"), list):
            return data
        blocks = data.get("content")
        if not isinstance(blocks, list):
            return data

        visible: list[str] = []
        reasoning: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", ""))
            value = block.get("text", block.get("thinking", ""))
            if not isinstance(value, str):
                continue
            if block_type in {"thinking", "reasoning"}:
                reasoning.append(value)
            elif block_type in {"text", "output_text"}:
                visible.append(value)

        raw_usage = data.get("usage", {})
        usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}
        if "input_tokens" in usage and "prompt_tokens" not in usage:
            usage["prompt_tokens"] = usage.get("input_tokens", 0)
        if "output_tokens" in usage and "completion_tokens" not in usage:
            usage["completion_tokens"] = usage.get("output_tokens", 0)
        usage.setdefault(
            "total_tokens",
            int(usage.get("prompt_tokens", 0) or 0)
            + int(usage.get("completion_tokens", 0) or 0),
        )
        return {
            "id": data.get("id"),
            "model": data.get("model"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(visible),
                    "reasoning_content": "".join(reasoning),
                },
                "finish_reason": LLMClient._canonical_finish_reason(
                    data.get("stop_reason")
                ),
            }],
            "usage": usage,
        }

    @staticmethod
    def _canonical_finish_reason(value: Any) -> Any:
        """Map common provider stop reasons to the OpenAI-compatible values."""
        if value in {"max_tokens", "max_output_tokens"}:
            return "length"
        if value in {"end_turn", "stop_sequence"}:
            return "stop"
        return value

    @staticmethod
    def _assemble_sse_chat_completion(response_text: str) -> dict[str, Any]:
        """Combine OpenAI- or Anthropic-compatible SSE into one response."""
        return LLMClient._parse_response_json(response_text)

    def set_trace_context(self, job_id: str | None, chunk_index: int | None) -> None:
        """Attach job context to attempt events in the current worker thread."""
        self._thread_local.trace_context = {
            "job_id": job_id,
            "chunk_index": chunk_index,
        }

    def set_operation(self, operation: str) -> None:
        """Label subsequent calls in this worker without changing chat protocols."""
        self._thread_local.operation = operation

    def set_attempt_observer(self, observer: Any) -> None:
        """Register a callback that receives sanitized per-attempt diagnostics."""
        self._attempt_observer = observer

    def _emit_attempt(self, payload: dict[str, Any]) -> None:
        context = getattr(self._thread_local, "trace_context", {})
        event = {**context, **payload}
        observer = self._attempt_observer
        if observer is not None:
            try:
                observer(event)
            except Exception:
                logger.exception("Failed to persist LLM attempt diagnostics.")

    def limit_next_call_attempts(self, attempts: int) -> None:
        """Constrain one follow-up call so nested repair cannot multiply retries."""
        self._thread_local.next_attempt_limit = max(0, int(attempts))

    def last_call_attempt_count(self) -> int:
        return int(getattr(self._thread_local, "last_call_attempts", 0))

    def remaining_attempt_budget(self, total: int = 3) -> int:
        used = int(getattr(self._thread_local, "last_call_attempts", 0))
        return max(0, int(total) - used)

    def _max_call_attempts(self) -> int:
        override = getattr(self._thread_local, "next_attempt_limit", None)
        if hasattr(self._thread_local, "next_attempt_limit"):
            del self._thread_local.next_attempt_limit
        recovery = self.config.llm.recovery
        if not recovery.enabled:
            configured = 1
        else:
            configured = min(
                max(1, int(self.config.retry.max_retries) + 1),
                int(recovery.max_attempts),
                4,
            )
        if override is not None:
            return min(configured, max(1, int(override)))
        return configured

    def _recovery_payload(
        self,
        original: dict[str, Any],
        attempt: int,
        max_attempts: int,
        failure_reason: str,
        operation: str = "completion",
        failure_usage: dict[str, Any] | None = None,
        failure_content: str = "",
        expected_output_tokens: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build recovery without mutating the unchanged first-attempt payload."""
        payload = dict(original)
        recovery = self.config.llm.recovery
        calculation: dict[str, Any] = {}

        # The first transport recovery is an exact logical replay. Increasing
        # output budget or changing reasoning cannot repair missing/corrupted
        # wire data and would make diagnosis and billing less trustworthy.
        if attempt == 1 and failure_reason in _TRANSPORT_FAILURE_REASONS:
            return payload, {
                "stage": "transport_replay",
                "same_request_contract": True,
                "failure_reason": failure_reason,
            }

        # Full-quality Attempt 2 intentionally keeps the original model and
        # reasoning policy. A configured fallback remains a third bounded rung.
        fallback_attempt = 2
        if attempt >= fallback_attempt and recovery.model.strip():
            payload["model"] = recovery.model.strip()

        # Helper Attempt 1 keeps normal model reasoning. Any bounded recovery
        # uses the portable OpenRouter/9router no-reasoning request so strict
        # JSON is emitted instead of consuming the allowance as hidden thought.
        helper_no_reasoning = (
            self.config.llm.provider.lower() == "openrouter"
            and operation in _STRUCTURED_HELPER_OPERATIONS
        )
        if helper_no_reasoning:
            payload["reasoning"] = {"effort": "none", "exclude": True}

        if failure_reason != "length":
            return payload, calculation

        preserve_second_request = (
            attempt == 1
            and operation in _UNCHANGED_SECOND_ATTEMPT_OPERATIONS
        )
        final_expanded = False
        no_reasoning_recovery = False
        if (
            self.config.llm.provider.lower() == "openrouter"
            and operation not in _PRESERVE_REASONING_ON_ALL_LENGTH_RETRIES
            and not preserve_second_request
        ):
            final_expanded = bool(
                recovery.expanded_final_attempt
                and max_attempts >= 4
                and attempt == max_attempts - 1
            )
            no_reasoning_recovery = _use_no_reasoning_recovery(
                operation=operation,
                attempt=attempt,
                max_attempts=max_attempts,
                has_fallback=bool(recovery.model.strip()),
                final_expanded=final_expanded,
            )
            reasoning_effort = (
                recovery.final_reasoning_effort
                if final_expanded or no_reasoning_recovery
                else recovery.reasoning_effort
            )
            payload["reasoning"] = {
                "effort": reasoning_effort,
                "exclude": True,
            }

        evidence = self._usage_evidence(failure_usage or {}, failure_content)
        original_max = int(original.get("max_tokens", self.config.llm.max_tokens))
        reasoning_tokens = evidence["reasoning_tokens"]
        if reasoning_tokens <= 0:
            reasoning_tokens = max(
                0,
                original_max - evidence["visible_tokens"],
            )
        answer_estimate, answer_evidence = self._answer_estimate(
            original, expected_output_tokens
        )
        answer_headroom = math.ceil(answer_estimate * 1.35) + 512
        reasoning_headroom = math.ceil(reasoning_tokens * 1.20)
        calculated = math.ceil((answer_headroom + reasoning_headroom) * 1.25)
        if (
            self.config.llm.provider.lower() == "openrouter"
            and operation not in _PRESERVE_REASONING_ON_ALL_LENGTH_RETRIES
            and (final_expanded or no_reasoning_recovery)
        ):
            calculated = max(calculated, int(recovery.max_tokens))
        ceiling, prompt_tokens = self._adaptive_ceiling(payload)
        # 8.5: within-request growth is useful -- if THIS call truncated, its
        # next attempt should get more room -- but it must be clamped by the
        # corrected ceiling so it cannot ask for more than the window holds.
        minimum_growth = max(1024, math.ceil(original_max * 0.50))
        requested = min(max(original_max + minimum_growth, calculated), ceiling)

        # 8.6: the previous attempt already ran AT the ceiling and still
        # truncated, and no extra room is available. An identical prompt at the
        # same cap cannot produce a different outcome -- in production this
        # burned three attempts of 11-14 minutes each on one chunk.
        exhausted_strategy = ""
        if (
            failure_reason == "length"
            and original_max >= ceiling
            and requested <= original_max
        ):
            if recovery.model.strip():
                payload["model"] = recovery.model.strip()
                exhausted_strategy = "ceiling_exhausted_fallback_model"
            else:
                # For critique the honest outcome is to stop here and let the
                # existing qa_unavailable path mark the chunk for review.
                raise TruncatedCompletionError(
                    "Output ceiling reached and no escalation is configured."
                )

        payload["max_tokens"] = requested
        calculation = {
            "stage": "recovery",
            "answer_estimate_tokens": answer_estimate,
            "answer_evidence": answer_evidence,
            "visible_output_tokens": evidence["visible_tokens"],
            "reported_completion_tokens": evidence["completion_tokens"],
            "reported_reasoning_tokens": evidence["reported_reasoning_tokens"],
            "reasoning_estimate_tokens": reasoning_tokens,
            "answer_headroom_tokens": answer_headroom,
            "reasoning_headroom_tokens": reasoning_headroom,
            "uncertainty_multiplier": 1.25,
            "calculated_max_tokens": calculated,
            "minimum_growth_tokens": minimum_growth,
            "estimated_prompt_tokens": prompt_tokens,
            "configured_ceiling": ceiling,
            "applied_max_tokens": requested,
            "ceiling_applied": requested < max(original_max + minimum_growth, calculated),
        }
        if exhausted_strategy:
            calculation["strategy"] = exhausted_strategy
        return payload, calculation

    def _initial_payload_for_operation(
        self,
        original: dict[str, Any],
        operation: str,
    ) -> dict[str, Any]:
        """Apply operation policy without changing the default request flow."""
        if self.config.llm.provider.lower() != "openrouter":
            return original
        if operation in _STRUCTURED_HELPER_OPERATIONS:
            payload = dict(original)
            if operation in _JSON_OBJECT_HELPER_OPERATIONS:
                payload["response_format"] = {"type": "json_object"}
            return payload
        if operation in {"translation", "translation_split_recovery"}:
            mode = self.config.llm.translation_reasoning
            if mode == "enabled":
                payload = dict(original)
                payload["reasoning"] = {"enabled": True, "exclude": True}
                return payload
            if mode == "disabled":
                payload = dict(original)
                payload["reasoning"] = {"effort": "none", "exclude": True}
                return payload
        if operation not in _SCHEMA_REPAIR_OPERATIONS:
            return original
        payload = dict(original)
        payload["reasoning"] = {
            "effort": self.config.llm.recovery.final_reasoning_effort,
            "exclude": True,
        }
        payload["max_tokens"] = max(
            int(original.get("max_tokens", self.config.llm.max_tokens)),
            int(self.config.llm.recovery.max_tokens),
        )
        return payload

    @staticmethod
    def _retryable_exception(exc: Exception, status_code: int | None) -> bool:
        return bool(
            isinstance(
                exc,
                (
                    EmptyCompletionError,
                    TruncatedCompletionError,
                    IncompleteCompletionError,
                    MalformedLLMResponseError,
                ),
            )
            or status_code is None
            or status_code in (429, 500, 502, 503, 504)
        )

    def _attempt_event(
        self,
        *,
        operation: str,
        attempt: int,
        max_attempts: int,
        payload: dict[str, Any],
        started: float,
        success: bool,
        finish_reason: str | None,
        status_code: int | None,
        usage: dict[str, Any],
        content: str = "",
        response_model: str = "",
        failure_reason: str = "",
        error: str = "",
        transport_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt_metrics = self._prompt_metrics(payload)
        usage_evidence = self._usage_evidence(usage, content)
        contract = {
            key: payload.get(key)
            for key in (
                "model", "messages", "temperature", "reasoning", "response_format"
            )
            if key in payload
        }
        contract_sha256 = hashlib.sha256(
            json.dumps(
                contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        request_sha256 = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        event = {
            "operation": operation,
            "attempt": attempt + 1,
            "max_attempts": max_attempts,
            "normal_attempt": attempt == 0,
            "recovery": attempt > 0,
            "recovery_stage": (
                "normal"
                if attempt == 0
                else (
                    "final_expanded"
                    if self.config.llm.recovery.expanded_final_attempt
                    and max_attempts >= 4
                    and attempt == max_attempts - 1
                    else (
                        "fallback_model"
                        if attempt >= (
                            2
                        ) and self.config.llm.recovery.model.strip()
                        else "same_model"
                    )
                )
            ),
            "success": success,
            "finish_reason": finish_reason,
            "status_code": status_code,
            "model": payload.get("model"),
            "response_model": response_model,
            "max_tokens": payload.get("max_tokens"),
            "reasoning": payload.get("reasoning"),
            "temperature": payload.get("temperature"),
            "response_format": payload.get("response_format"),
            "stream_requested": bool(payload.get("stream", False)),
            "transport_read_timeout_seconds": (
                self.config.llm.transport.read_timeout_seconds
            ),
            "request_contract_sha256": contract_sha256,
            "request_payload_sha256": request_sha256,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "duration_seconds": round(time.monotonic() - started, 3),
            "reported_reasoning_tokens": usage_evidence["reported_reasoning_tokens"],
            "normalized_reasoning_tokens": usage_evidence["reasoning_tokens"],
            "visible_completion_tokens": usage_evidence["visible_tokens"],
            "visible_output_present": bool((content or "").strip()),
            "usage_reported": any(
                usage.get(key) is not None
                for key in (
                    "prompt_tokens", "completion_tokens", "total_tokens",
                    "input_tokens", "output_tokens",
                )
            ),
            **prompt_metrics,
        }
        event["token_accounting_status"] = (
            "reported"
            if event["usage_reported"]
            else "provider_usage_missing"
            if event["visible_output_present"]
            else "no_visible_output"
        )
        if failure_reason:
            event["failure_reason"] = failure_reason
        if error:
            event["error"] = error[:1000]
        evidence = dict(transport_evidence or {})
        if evidence:
            event["transport"] = {
                key: evidence.get(key)
                for key in (
                    "transport_profile",
                    "protocol",
                    "response_bytes",
                    "line_count",
                    "event_count",
                    "data_event_count",
                    "terminal_received",
                    "content_bytes",
                    "reasoning_bytes",
                    "event_types",
                    "response_sha256",
                    "response_content_type",
                    "response_identifiers",
                    "first_event_seconds",
                    "buffered_response",
                    "trailing_json_repair",
                    "trailing_bytes_ignored",
                )
                if key in evidence
            }
        return event

    def complete(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronously complete one bounded logical request."""
        operation = str(kwargs.pop("_operation", "completion"))
        recovery_source_text = str(kwargs.pop("_recovery_source_text", "") or "")
        expected_output_tokens = (
            math.ceil(self.count_tokens(recovery_source_text) * 1.35)
            if recovery_source_text else 0
        )
        url, headers, original_payload = self._prepare_request(
            messages, system_prompt, **kwargs
        )
        original_payload = self._initial_payload_for_operation(
            original_payload, operation
        )
        original_payload, preflight_calculation = self._preflight_payload(
            original_payload, operation, expected_output_tokens
        )
        max_attempts = self._max_call_attempts()
        if (
            max_attempts >= 2
            and operation in _FULL_QUALITY_LENGTH_RECOVERY_OPERATIONS
            and self.config.llm.recovery.model.strip()
        ):
            max_attempts = max(max_attempts, 3)
        failure_reason = ""
        failure_usage: dict[str, Any] = {}
        failure_content = ""

        for attempt in range(max_attempts):
            self._thread_local.last_call_attempts = attempt + 1
            recovery_calculation: dict[str, Any] = {}
            if attempt == 0:
                payload = original_payload
            else:
                payload, recovery_calculation = self._recovery_payload(
                    original_payload,
                    attempt,
                    max_attempts,
                    failure_reason,
                    operation,
                    failure_usage,
                    failure_content,
                    expected_output_tokens,
                )
            started = time.monotonic()
            finish_reason = None
            status_code = None
            content: Any = None
            usage: dict[str, Any] = {}
            response_model = ""
            normalization: dict[str, Any] = {}
            response_reasoning_chars = 0
            transport_evidence: dict[str, Any] = {
                "transport_profile": self._transport_profile(url),
            }
            try:
                response, res_json, transport_evidence = self._request_sync(
                    url, headers, payload
                )
                status_code = response.status_code
                response_model = str(res_json.get("model", "") or "")
                self._update_usage(response, res_json)
                raw_usage = res_json.get("usage", {})
                usage = raw_usage if isinstance(raw_usage, dict) else {}
                choices = res_json.get("choices", [])
                if not choices:
                    raise MalformedLLMResponseError(
                        f"Empty choices in response: {res_json}"
                    )

                finish_reason = choices[0].get("finish_reason")
                message = choices[0].get("message", {})
                content = message.get("content")
                reasoning_content = message.get("reasoning_content", "")
                response_reasoning_chars = (
                    len(reasoning_content)
                    if isinstance(reasoning_content, str) else 0
                )
                content, normalization = normalize_model_text(content)
                if finish_reason == "length":
                    raise TruncatedCompletionError(
                        "Provider stopped the completion at the output limit."
                    )
                if finish_reason in {"error", "cancelled"}:
                    raise IncompleteCompletionError(
                        f"Provider returned finish_reason={finish_reason!r}."
                    )
                if not content or not content.strip():
                    raise EmptyCompletionError(
                        "LLM returned an empty or null translation completion."
                    )

                self._record_budget_observation(
                    payload=payload,
                    operation=operation,
                    usage=usage,
                    content=content,
                    finish_reason=finish_reason,
                    response_model=response_model,
                )

                event = self._attempt_event(
                    operation=operation,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    payload=payload,
                    started=started,
                    success=True,
                    finish_reason=finish_reason,
                    status_code=status_code,
                    usage=usage,
                    content=content,
                    response_model=response_model,
                    transport_evidence=transport_evidence,
                )
                if attempt == 0 and preflight_calculation:
                    event["preflight_calculation"] = preflight_calculation
                if recovery_calculation:
                    event["recovery_calculation"] = recovery_calculation
                event["response_normalization"] = normalization
                event["response_reasoning_chars"] = response_reasoning_chars
                self._emit_attempt(event)
                return content
            except (
                httpx.HTTPStatusError,
                httpx.RequestError,
                EmptyCompletionError,
                TruncatedCompletionError,
                IncompleteCompletionError,
                MalformedLLMResponseError,
            ) as exc:
                status_code = (
                    getattr(exc.response, "status_code", None)
                    if hasattr(exc, "response") else status_code
                )
                transport_evidence.update(
                    dict(getattr(exc, "transport_evidence", {}) or {})
                )
                if isinstance(exc, TruncatedCompletionError):
                    failure_reason = "length"
                elif isinstance(exc, IncompleteCompletionError):
                    failure_reason = str(
                        getattr(exc, "transport_failure_kind", "")
                        or finish_reason
                        or "incomplete_completion"
                    )
                elif isinstance(exc, EmptyCompletionError):
                    failure_reason = str(
                        getattr(exc, "transport_failure_kind", "")
                        or "empty_completion"
                    )
                elif isinstance(exc, MalformedLLMResponseError):
                    failure_reason = str(
                        getattr(exc, "transport_failure_kind", "")
                        or "malformed_response"
                    )
                else:
                    if isinstance(exc, httpx.ReadTimeout):
                        failure_reason = "read_timeout"
                    elif isinstance(exc, httpx.ConnectTimeout):
                        failure_reason = "connect_timeout"
                    else:
                        failure_reason = (
                            f"http_{status_code}" if status_code else "transport_error"
                        )

                failure_usage = dict(usage)
                failure_content = content if isinstance(content, str) else ""
                self._record_budget_observation(
                    payload=payload,
                    operation=operation,
                    usage=usage,
                    content=failure_content,
                    finish_reason=finish_reason,
                    response_model=response_model,
                )

                event = self._attempt_event(
                    operation=operation,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    payload=payload,
                    started=started,
                    success=False,
                    finish_reason=finish_reason,
                    status_code=status_code,
                    usage=usage,
                    content=failure_content,
                    response_model=response_model,
                    failure_reason=failure_reason,
                    error=str(exc),
                    transport_evidence=transport_evidence,
                )
                if attempt == 0 and preflight_calculation:
                    event["preflight_calculation"] = preflight_calculation
                if recovery_calculation:
                    event["recovery_calculation"] = recovery_calculation
                self._emit_attempt(event)
                should_retry = (
                    attempt + 1 < max_attempts
                    and self._retryable_exception(exc, status_code)
                )
                if isinstance(exc, httpx.ReadTimeout):
                    should_retry = should_retry and attempt < int(
                        self.config.llm.transport.unknown_outcome_retries
                    )
                if not should_retry:
                    logger.error(
                        "LLM call failed permanently after %d attempt(s): %s",
                        attempt + 1,
                        exc,
                    )
                    raise

                delay = min(
                    self.config.retry.max_delay,
                    self.config.retry.base_delay * (2 ** attempt),
                )
                if self.config.retry.jitter:
                    delay = delay / 2 + random.uniform(0, delay / 2)
                logger.warning(
                    "LLM call failed (attempt %d/%d, reason=%s, status=%s): %s. "
                    "Retrying in %.2fs...",
                    attempt + 1,
                    max_attempts,
                    failure_reason,
                    status_code,
                    exc,
                    delay,
                )
                time.sleep(delay)
                exact_transport_replay = (
                    attempt == 0 and failure_reason in _TRANSPORT_FAILURE_REASONS
                )
                if (
                    self.config.llm.provider.lower() == "openrouter"
                    and not exact_transport_replay
                ):
                    headers["Authorization"] = f"Bearer {self._get_next_api_key()}"

        raise RuntimeError("LLM request exhausted its bounded attempt budget.")

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously complete one bounded logical request."""
        operation = str(kwargs.pop("_operation", "completion"))
        recovery_source_text = str(kwargs.pop("_recovery_source_text", "") or "")
        expected_output_tokens = (
            math.ceil(self.count_tokens(recovery_source_text) * 1.35)
            if recovery_source_text else 0
        )
        url, headers, original_payload = self._prepare_request(
            messages, system_prompt, **kwargs
        )
        original_payload = self._initial_payload_for_operation(
            original_payload, operation
        )
        original_payload, preflight_calculation = self._preflight_payload(
            original_payload, operation, expected_output_tokens
        )
        max_attempts = self._max_call_attempts()
        if (
            max_attempts >= 2
            and operation in _FULL_QUALITY_LENGTH_RECOVERY_OPERATIONS
            and self.config.llm.recovery.model.strip()
        ):
            max_attempts = max(max_attempts, 3)
        failure_reason = ""
        failure_usage: dict[str, Any] = {}
        failure_content = ""

        for attempt in range(max_attempts):
            self._thread_local.last_call_attempts = attempt + 1
            recovery_calculation: dict[str, Any] = {}
            if attempt == 0:
                payload = original_payload
            else:
                payload, recovery_calculation = self._recovery_payload(
                    original_payload,
                    attempt,
                    max_attempts,
                    failure_reason,
                    operation,
                    failure_usage,
                    failure_content,
                    expected_output_tokens,
                )
            started = time.monotonic()
            finish_reason = None
            status_code = None
            content: Any = None
            usage: dict[str, Any] = {}
            response_model = ""
            normalization: dict[str, Any] = {}
            response_reasoning_chars = 0
            transport_evidence: dict[str, Any] = {
                "transport_profile": self._transport_profile(url),
            }
            try:
                response, res_json, transport_evidence = await self._request_async(
                    url, headers, payload
                )
                status_code = response.status_code
                response_model = str(res_json.get("model", "") or "")
                self._update_usage(response, res_json)
                raw_usage = res_json.get("usage", {})
                usage = raw_usage if isinstance(raw_usage, dict) else {}
                choices = res_json.get("choices", [])
                if not choices:
                    raise MalformedLLMResponseError(
                        f"Empty choices in response: {res_json}"
                    )

                finish_reason = choices[0].get("finish_reason")
                message = choices[0].get("message", {})
                content = message.get("content")
                reasoning_content = message.get("reasoning_content", "")
                response_reasoning_chars = (
                    len(reasoning_content)
                    if isinstance(reasoning_content, str) else 0
                )
                content, normalization = normalize_model_text(content)
                if finish_reason == "length":
                    raise TruncatedCompletionError(
                        "Provider stopped the completion at the output limit."
                    )
                if finish_reason in {"error", "cancelled"}:
                    raise IncompleteCompletionError(
                        f"Provider returned finish_reason={finish_reason!r}."
                    )
                if not content or not content.strip():
                    raise EmptyCompletionError(
                        "LLM returned an empty or null translation completion."
                    )

                self._record_budget_observation(
                    payload=payload,
                    operation=operation,
                    usage=usage,
                    content=content,
                    finish_reason=finish_reason,
                    response_model=response_model,
                )

                event = self._attempt_event(
                    operation=operation,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    payload=payload,
                    started=started,
                    success=True,
                    finish_reason=finish_reason,
                    status_code=status_code,
                    usage=usage,
                    content=content,
                    response_model=response_model,
                    transport_evidence=transport_evidence,
                )
                if attempt == 0 and preflight_calculation:
                    event["preflight_calculation"] = preflight_calculation
                if recovery_calculation:
                    event["recovery_calculation"] = recovery_calculation
                event["response_normalization"] = normalization
                event["response_reasoning_chars"] = response_reasoning_chars
                self._emit_attempt(event)
                return content
            except (
                httpx.HTTPStatusError,
                httpx.RequestError,
                EmptyCompletionError,
                TruncatedCompletionError,
                IncompleteCompletionError,
                MalformedLLMResponseError,
            ) as exc:
                status_code = (
                    getattr(exc.response, "status_code", None)
                    if hasattr(exc, "response") else status_code
                )
                transport_evidence.update(
                    dict(getattr(exc, "transport_evidence", {}) or {})
                )
                if isinstance(exc, TruncatedCompletionError):
                    failure_reason = "length"
                elif isinstance(exc, IncompleteCompletionError):
                    failure_reason = str(
                        getattr(exc, "transport_failure_kind", "")
                        or finish_reason
                        or "incomplete_completion"
                    )
                elif isinstance(exc, EmptyCompletionError):
                    failure_reason = str(
                        getattr(exc, "transport_failure_kind", "")
                        or "empty_completion"
                    )
                elif isinstance(exc, MalformedLLMResponseError):
                    failure_reason = str(
                        getattr(exc, "transport_failure_kind", "")
                        or "malformed_response"
                    )
                else:
                    if isinstance(exc, httpx.ReadTimeout):
                        failure_reason = "read_timeout"
                    elif isinstance(exc, httpx.ConnectTimeout):
                        failure_reason = "connect_timeout"
                    else:
                        failure_reason = (
                            f"http_{status_code}" if status_code else "transport_error"
                        )

                failure_usage = dict(usage)
                failure_content = content if isinstance(content, str) else ""
                self._record_budget_observation(
                    payload=payload,
                    operation=operation,
                    usage=usage,
                    content=failure_content,
                    finish_reason=finish_reason,
                    response_model=response_model,
                )

                event = self._attempt_event(
                    operation=operation,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    payload=payload,
                    started=started,
                    success=False,
                    finish_reason=finish_reason,
                    status_code=status_code,
                    usage=usage,
                    content=failure_content,
                    response_model=response_model,
                    failure_reason=failure_reason,
                    error=str(exc),
                    transport_evidence=transport_evidence,
                )
                if attempt == 0 and preflight_calculation:
                    event["preflight_calculation"] = preflight_calculation
                if recovery_calculation:
                    event["recovery_calculation"] = recovery_calculation
                self._emit_attempt(event)
                should_retry = (
                    attempt + 1 < max_attempts
                    and self._retryable_exception(exc, status_code)
                )
                if isinstance(exc, httpx.ReadTimeout):
                    should_retry = should_retry and attempt < int(
                        self.config.llm.transport.unknown_outcome_retries
                    )
                if not should_retry:
                    logger.error(
                        "LLM call failed permanently after %d attempt(s): %s",
                        attempt + 1,
                        exc,
                    )
                    raise

                delay = min(
                    self.config.retry.max_delay,
                    self.config.retry.base_delay * (2 ** attempt),
                )
                if self.config.retry.jitter:
                    delay = delay / 2 + random.uniform(0, delay / 2)
                logger.warning(
                    "LLM call failed (attempt %d/%d, reason=%s, status=%s): %s. "
                    "Retrying in %.2fs...",
                    attempt + 1,
                    max_attempts,
                    failure_reason,
                    status_code,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                exact_transport_replay = (
                    attempt == 0 and failure_reason in _TRANSPORT_FAILURE_REASONS
                )
                if (
                    self.config.llm.provider.lower() == "openrouter"
                    and not exact_transport_replay
                ):
                    headers["Authorization"] = f"Bearer {self._get_next_api_key()}"

        raise RuntimeError("LLM request exhausted its bounded attempt budget.")
    async def chat(self, prompt: str) -> str:
        """Compatibility method for async quality and context tools."""
        operation = getattr(self._thread_local, "operation", "chat")
        self._thread_local.operation = "chat"
        return await self.acomplete(
            messages=[{"role": "user", "content": prompt}],
            _operation=operation,
        )
