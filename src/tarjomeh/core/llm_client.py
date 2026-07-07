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


class EmptyCompletionError(Exception):
    """Raised when the LLM returns an empty completion."""
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
            api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1").rstrip("/")
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

    def complete(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronously complete a chat request with key rotation and retries."""
        url, headers, payload = self._prepare_request(messages, system_prompt, **kwargs)
        max_retries = self.config.retry.max_retries
        base_delay = self.config.retry.base_delay
        max_delay = self.config.retry.max_delay
        jitter = self.config.retry.jitter

        for attempt in range(max_retries + 1):
            try:
                response = self._client.post(url, headers=headers, json=payload)
                
                # Check for rate limits or server errors explicitly
                if response.status_code in (429, 500, 502, 503, 504):
                    response.raise_for_status()
                
                response.raise_for_status()
                
                # Safe JSON parsing that handles trailing garbage (like "data: [DONE]")
                res_text = response.text.strip()
                try:
                    res_json = json.loads(res_text)
                except json.JSONDecodeError:
                    last_brace = res_text.rfind("}")
                    if last_brace != -1:
                        try:
                            res_json = json.loads(res_text[:last_brace + 1])
                        except json.JSONDecodeError:
                            raise
                    else:
                        raise
                
                self._update_usage(response, res_json)
                
                choices = res_json.get("choices", [])
                if not choices:
                    raise ValueError(f"Empty choices in response: {res_json}")
                
                content = choices[0].get("message", {}).get("content")
                if not content or not content.strip():
                    raise EmptyCompletionError("LLM returned an empty or null translation completion.")
                
                return content

            except (httpx.HTTPStatusError, httpx.RequestError, EmptyCompletionError) as exc:
                status_code = getattr(exc.response, "status_code", None) if hasattr(exc, "response") else None
                
                # Check if we should retry
                should_retry = (
                    attempt < max_retries and
                    (isinstance(exc, EmptyCompletionError) or status_code is None or status_code in (429, 500, 502, 503, 504))
                )

                if should_retry:
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    if jitter:
                        delay = delay / 2 + random.uniform(0, delay / 2)
                    
                    logger.warning(
                        "LLM call failed (attempt %d/%d, status=%s): %s. Retrying in %.2fs...",
                        attempt + 1, max_retries, status_code, exc, delay
                    )
                    time.sleep(delay)

                    # For OpenRouter, rotate API key on retry
                    if self.config.llm.provider.lower() == "openrouter":
                        headers["Authorization"] = f"Bearer {self._get_next_api_key()}"
                else:
                    logger.error("LLM call failed permanently after %d retries: %s", attempt, exc)
                    raise

        raise RuntimeError("LLM request failed after max retries without returning a response.")

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously complete a chat request with key rotation and retries."""
        url, headers, payload = self._prepare_request(messages, system_prompt, **kwargs)
        max_retries = self.config.retry.max_retries
        base_delay = self.config.retry.base_delay
        max_delay = self.config.retry.max_delay
        jitter = self.config.retry.jitter

        for attempt in range(max_retries + 1):
            try:
                response = await self._aclient.post(url, headers=headers, json=payload)
                
                # Check for rate limits or server errors explicitly
                if response.status_code in (429, 500, 502, 503, 504):
                    response.raise_for_status()
                
                response.raise_for_status()
                
                # Safe JSON parsing that handles trailing garbage (like "data: [DONE]")
                res_text = response.text.strip()
                try:
                    res_json = json.loads(res_text)
                except json.JSONDecodeError:
                    last_brace = res_text.rfind("}")
                    if last_brace != -1:
                        try:
                            res_json = json.loads(res_text[:last_brace + 1])
                        except json.JSONDecodeError:
                            raise
                    else:
                        raise
                
                self._update_usage(response, res_json)
                
                choices = res_json.get("choices", [])
                if not choices:
                    raise ValueError(f"Empty choices in response: {res_json}")
                
                content = choices[0].get("message", {}).get("content")
                if not content or not content.strip():
                    raise EmptyCompletionError("LLM returned an empty or null translation completion.")
                
                return content

            except (httpx.HTTPStatusError, httpx.RequestError, EmptyCompletionError) as exc:
                status_code = getattr(exc.response, "status_code", None) if hasattr(exc, "response") else None
                
                # Check if we should retry
                should_retry = (
                    attempt < max_retries and
                    (isinstance(exc, EmptyCompletionError) or status_code is None or status_code in (429, 500, 502, 503, 504))
                )

                if should_retry:
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    if jitter:
                        delay = delay / 2 + random.uniform(0, delay / 2)
                    
                    logger.warning(
                        "LLM call failed (attempt %d/%d, status=%s): %s. Retrying in %.2fs...",
                        attempt + 1, max_retries, status_code, exc, delay
                    )
                    await asyncio.sleep(delay)

                    # For OpenRouter, rotate API key on retry
                    if self.config.llm.provider.lower() == "openrouter":
                        headers["Authorization"] = f"Bearer {self._get_next_api_key()}"
                else:
                    logger.error("LLM call failed permanently after %d retries: %s", attempt, exc)
                    raise

        raise RuntimeError("LLM request failed after max retries without returning a response.")

    async def chat(self, prompt: str) -> str:
        """Compatibility method for quality tools that expect an async chat(prompt) -> str interface."""
        return await self.acomplete(messages=[{"role": "user", "content": prompt}])
