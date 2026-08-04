"""LLM Client for the Tarjomeh translation system.

Handles API key rotation, exponential backoff with jitter on 429/5xx errors,
token counting via tiktoken, usage tracking, and OpenAI-compatible requests
to OpenRouter and Ollama.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import threading
import time
from typing import Any

import httpx
import tiktoken

from tarjomeh.core.config import TarjomehConfig

logger = logging.getLogger(__name__)


_SCHEMA_REPAIR_OPERATIONS = {
    "critique_json_repair",
    "refinement_json_repair",
}
_QUALITY_RECOVERY_OPERATIONS = {
    "critique",
    "critique_json_repair",
    "refinement",
    "refinement_json_repair",
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


class LLMClient:
    """OpenAI-compatible client for communicating with OpenRouter or Ollama.

    Provides sync and async interfaces, key rotation, retries, and token counting.
    """

    def __init__(self, config: TarjomehConfig) -> None:
        self.config = config
        self._client = httpx.Client(timeout=180.0)
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

    @property
    def _aclient(self) -> httpx.AsyncClient:
        """Get or create an AsyncClient bound to the active event loop for the current thread."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop_id = id(loop)
        if not hasattr(self._thread_local, "clients"):
            self._thread_local.clients = {}
            
        if loop_id not in self._thread_local.clients or self._thread_local.clients[loop_id].is_closed:
            self._thread_local.clients[loop_id] = httpx.AsyncClient(timeout=180.0)
            
        return self._thread_local.clients[loop_id]

    def close(self) -> None:
        """Close the sync HTTP client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close all thread-local async HTTP clients."""
        if hasattr(self._thread_local, "clients"):
            for client in self._thread_local.clients.values():
                if not client.is_closed:
                    await client.aclose()

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
            payload = {
                "model": self.config.llm.model,
                "messages": final_messages,
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens,
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
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Update payload with custom kwargs
        payload.update(kwargs)
        return url, headers, payload

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
        """Parse a JSON response or assemble an OpenAI-compatible SSE stream."""
        res_text = response_text.strip()
        try:
            return json.loads(res_text)
        except json.JSONDecodeError as first_error:
            if res_text.startswith("data:"):
                return LLMClient._assemble_sse_chat_completion(res_text)

            last_brace = res_text.rfind("}")
            if last_brace != -1:
                try:
                    return json.loads(res_text[:last_brace + 1])
                except json.JSONDecodeError:
                    pass

            snippet = res_text[:300].replace("\n", "\\n")
            raise MalformedLLMResponseError(
                f"LLM endpoint returned malformed JSON response: {snippet!r}"
            ) from first_error

    @staticmethod
    def _assemble_sse_chat_completion(response_text: str) -> dict[str, Any]:
        """Combine ``chat.completion.chunk`` SSE events into one response."""
        result: dict[str, Any] = {}
        content_parts: list[str] = []
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        saw_event = False
        saw_done = False

        for raw_line in response_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(":") or line.startswith("event:"):
                continue
            if not line.startswith("data:"):
                raise MalformedLLMResponseError(
                    "LLM endpoint returned an invalid SSE chat-completion stream."
                )

            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                continue

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                snippet = data[:300].replace("\n", "\\n")
                raise MalformedLLMResponseError(
                    f"LLM endpoint returned malformed SSE data: {snippet!r}"
                ) from exc
            if not isinstance(chunk, dict):
                raise MalformedLLMResponseError(
                    "LLM endpoint returned a non-object SSE data event."
                )
            if chunk.get("error"):
                raise IncompleteCompletionError(
                    f"LLM SSE stream reported an error: {chunk['error']}"
                )

            saw_event = True
            for key in ("id", "object", "created", "model", "system_fingerprint"):
                if key in chunk and key not in result:
                    result[key] = chunk[key]

            raw_usage = chunk.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage

            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            message = choice.get("message")
            part = None
            if isinstance(delta, dict):
                part = delta.get("content")
            elif isinstance(message, dict):
                part = message.get("content")
            if isinstance(part, str):
                content_parts.append(part)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

        if not saw_event:
            raise MalformedLLMResponseError(
                "LLM endpoint returned an empty SSE chat-completion stream."
            )
        if not saw_done and finish_reason is None:
            raise IncompleteCompletionError(
                "LLM SSE stream ended without [DONE] or a finish reason."
            )

        result["choices"] = [{
            "index": 0,
            "message": {"role": "assistant", "content": "".join(content_parts)},
            "finish_reason": finish_reason,
        }]
        if usage:
            result["usage"] = usage
        return result

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
    ) -> dict[str, Any]:
        """Build recovery without mutating the unchanged first-attempt payload."""
        payload = dict(original)
        recovery = self.config.llm.recovery

        if attempt >= 2 and recovery.model.strip():
            payload["model"] = recovery.model.strip()

        if self.config.llm.provider.lower() == "openrouter" and failure_reason == "length":
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
            prompt_chars = sum(
                len(str(message.get("content", "")))
                for message in payload.get("messages", [])
                if isinstance(message, dict)
            )
            estimated_visible = max(2048, int(prompt_chars / 2.5) + 1024)
            original_max = int(
                original.get("max_tokens", self.config.llm.max_tokens)
            )
            recovery_ceiling = max(original_max, int(recovery.max_tokens))
            payload["max_tokens"] = (
                recovery_ceiling
                if final_expanded or no_reasoning_recovery
                else max(original_max, min(estimated_visible, recovery_ceiling))
            )
        return payload

    def _initial_payload_for_operation(
        self,
        original: dict[str, Any],
        operation: str,
    ) -> dict[str, Any]:
        """Constrain schema-repair calls without changing normal operations."""
        if (
            self.config.llm.provider.lower() != "openrouter"
            or operation not in _SCHEMA_REPAIR_OPERATIONS
        ):
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
        failure_reason: str = "",
        error: str = "",
    ) -> dict[str, Any]:
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
                        if attempt >= 2 and self.config.llm.recovery.model.strip()
                        else "same_model"
                    )
                )
            ),
            "success": success,
            "finish_reason": finish_reason,
            "status_code": status_code,
            "model": payload.get("model"),
            "max_tokens": payload.get("max_tokens"),
            "reasoning": payload.get("reasoning"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        if failure_reason:
            event["failure_reason"] = failure_reason
        if error:
            event["error"] = error[:1000]
        return event

    def complete(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronously complete one bounded logical request."""
        operation = str(kwargs.pop("_operation", "completion"))
        url, headers, original_payload = self._prepare_request(
            messages, system_prompt, **kwargs
        )
        original_payload = self._initial_payload_for_operation(
            original_payload, operation
        )
        max_attempts = self._max_call_attempts()
        failure_reason = ""

        for attempt in range(max_attempts):
            self._thread_local.last_call_attempts = attempt + 1
            payload = (
                original_payload
                if attempt == 0
                else self._recovery_payload(
                    original_payload,
                    attempt,
                    max_attempts,
                    failure_reason,
                    operation,
                )
            )
            started = time.monotonic()
            finish_reason = None
            status_code = None
            usage: dict[str, Any] = {}
            try:
                response = self._client.post(url, headers=headers, json=payload)
                status_code = response.status_code
                response.raise_for_status()
                res_json = self._parse_response_json(response.text)
                self._update_usage(response, res_json)
                raw_usage = res_json.get("usage", {})
                usage = raw_usage if isinstance(raw_usage, dict) else {}
                choices = res_json.get("choices", [])
                if not choices:
                    raise MalformedLLMResponseError(
                        f"Empty choices in response: {res_json}"
                    )

                finish_reason = choices[0].get("finish_reason")
                content = choices[0].get("message", {}).get("content")
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

                self._emit_attempt(self._attempt_event(
                    operation=operation,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    payload=payload,
                    started=started,
                    success=True,
                    finish_reason=finish_reason,
                    status_code=status_code,
                    usage=usage,
                ))
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
                if isinstance(exc, TruncatedCompletionError):
                    failure_reason = "length"
                elif isinstance(exc, IncompleteCompletionError):
                    failure_reason = str(finish_reason or "error")
                elif isinstance(exc, EmptyCompletionError):
                    failure_reason = "empty"
                elif isinstance(exc, MalformedLLMResponseError):
                    failure_reason = "malformed_response"
                else:
                    failure_reason = (
                        f"http_{status_code}" if status_code else "transport_error"
                    )

                self._emit_attempt(self._attempt_event(
                    operation=operation,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    payload=payload,
                    started=started,
                    success=False,
                    finish_reason=finish_reason,
                    status_code=status_code,
                    usage=usage,
                    failure_reason=failure_reason,
                    error=str(exc),
                ))
                should_retry = (
                    attempt + 1 < max_attempts
                    and self._retryable_exception(exc, status_code)
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
                if self.config.llm.provider.lower() == "openrouter":
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
        url, headers, original_payload = self._prepare_request(
            messages, system_prompt, **kwargs
        )
        original_payload = self._initial_payload_for_operation(
            original_payload, operation
        )
        max_attempts = self._max_call_attempts()
        failure_reason = ""

        for attempt in range(max_attempts):
            self._thread_local.last_call_attempts = attempt + 1
            payload = (
                original_payload
                if attempt == 0
                else self._recovery_payload(
                    original_payload,
                    attempt,
                    max_attempts,
                    failure_reason,
                    operation,
                )
            )
            started = time.monotonic()
            finish_reason = None
            status_code = None
            usage: dict[str, Any] = {}
            try:
                response = await self._aclient.post(
                    url, headers=headers, json=payload
                )
                status_code = response.status_code
                response.raise_for_status()
                res_json = self._parse_response_json(response.text)
                self._update_usage(response, res_json)
                raw_usage = res_json.get("usage", {})
                usage = raw_usage if isinstance(raw_usage, dict) else {}
                choices = res_json.get("choices", [])
                if not choices:
                    raise MalformedLLMResponseError(
                        f"Empty choices in response: {res_json}"
                    )

                finish_reason = choices[0].get("finish_reason")
                content = choices[0].get("message", {}).get("content")
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

                self._emit_attempt(self._attempt_event(
                    operation=operation,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    payload=payload,
                    started=started,
                    success=True,
                    finish_reason=finish_reason,
                    status_code=status_code,
                    usage=usage,
                ))
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
                if isinstance(exc, TruncatedCompletionError):
                    failure_reason = "length"
                elif isinstance(exc, IncompleteCompletionError):
                    failure_reason = str(finish_reason or "error")
                elif isinstance(exc, EmptyCompletionError):
                    failure_reason = "empty"
                elif isinstance(exc, MalformedLLMResponseError):
                    failure_reason = "malformed_response"
                else:
                    failure_reason = (
                        f"http_{status_code}" if status_code else "transport_error"
                    )

                self._emit_attempt(self._attempt_event(
                    operation=operation,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    payload=payload,
                    started=started,
                    success=False,
                    finish_reason=finish_reason,
                    status_code=status_code,
                    usage=usage,
                    failure_reason=failure_reason,
                    error=str(exc),
                ))
                should_retry = (
                    attempt + 1 < max_attempts
                    and self._retryable_exception(exc, status_code)
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
                if self.config.llm.provider.lower() == "openrouter":
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
