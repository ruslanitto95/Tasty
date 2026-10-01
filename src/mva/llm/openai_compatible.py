"""OpenAI-compatible Chat Completions client (OpenAI, Azure-compatible gateways,
GigaChat/YandexGPT proxies, local Ollama / LM Studio / vLLM servers)."""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

import httpx

from mva.config import DEFAULTS, LLMDefaults
from mva.llm.base import (
    ChatMessage,
    LLMAuthError,
    LLMCancelled,
    LLMInvalidResponse,
    LLMProvider,
    LLMTimeout,
    LLMUnavailable,
    is_local_url,
)

log = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_s: float = 120.0,
        limits: LLMDefaults = DEFAULTS.llm,
        transport: httpx.BaseTransport | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("base_url must be http(s)")
        if base_url.startswith("http://") and not is_local_url(base_url):
            raise ValueError("plain http is allowed only for local servers")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._limits = limits
        self._timeout = httpx.Timeout(
            connect=limits.connect_timeout_s, read=timeout_s, write=30.0, pool=10.0
        )
        self._transport = transport
        self._sleep = sleep
        self._json_schema_supported = True

    @property
    def is_cloud(self) -> bool:
        return not is_local_url(self.base_url)

    def _client(self) -> httpx.Client:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return httpx.Client(timeout=self._timeout, headers=headers, transport=self._transport)

    def complete_json(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any] | None = None,
        schema_name: str = "result",
        cancel: threading.Event | None = None,
    ) -> str:
        deadline = time.monotonic() + self._limits.overall_timeout_s
        last: Exception | None = None
        for attempt in range(1, self._limits.max_attempts + 1):
            if cancel is not None and cancel.is_set():
                raise LLMCancelled()
            if time.monotonic() > deadline:
                raise LLMTimeout("overall deadline exceeded")
            try:
                return self._once(messages, schema, schema_name)
            except (LLMAuthError, LLMInvalidResponse):
                raise
            except (LLMTimeout, LLMUnavailable) as exc:
                last = exc
                if attempt == self._limits.max_attempts:
                    break
                delay = self._limits.backoff_base_s * (2 ** (attempt - 1)) + random.uniform(0, 0.3)  # noqa: S311
                log.warning(
                    "LLM attempt %d failed (%s); retrying in %.1fs", attempt, exc.code, delay
                )
                end = time.monotonic() + delay
                while time.monotonic() < end:
                    if cancel is not None and cancel.is_set():
                        raise LLMCancelled() from exc
                    self._sleep(min(0.1, delay))
        assert last is not None
        raise last

    def _payload(
        self, messages: list[ChatMessage], schema: dict[str, Any] | None, schema_name: str
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if schema is not None and self._json_schema_supported:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": False},
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _once(
        self, messages: list[ChatMessage], schema: dict[str, Any] | None, schema_name: str
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        try:
            with self._client() as client:
                resp = client.post(url, json=self._payload(messages, schema, schema_name))
                if resp.status_code == 400 and schema is not None and self._json_schema_supported:
                    # Server without structured-output support: fall back to JSON mode.
                    self._json_schema_supported = False
                    resp = client.post(url, json=self._payload(messages, schema, schema_name))
        except httpx.TimeoutException as exc:
            raise LLMTimeout(type(exc).__name__) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(type(exc).__name__) from exc
        if resp.status_code in (401, 403):
            raise LLMAuthError(f"HTTP {resp.status_code}")
        if resp.status_code == 429 or resp.status_code >= 500:
            raise LLMUnavailable(f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise LLMInvalidResponse(f"HTTP {resp.status_code}")
        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMInvalidResponse("unexpected response shape") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMInvalidResponse("empty content")
        return content
