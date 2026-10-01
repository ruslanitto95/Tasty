"""Scripted LLM for tests and developer mode. Never calls the network."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

from mva.llm.base import ChatMessage, LLMError, LLMProvider

Responder = Callable[[list[ChatMessage]], str | dict[str, Any] | LLMError]


class MockLLMProvider(LLMProvider):
    name = "mock"

    def __init__(self, responses: list[str | dict[str, Any] | LLMError] | Responder) -> None:
        self._responses = responses
        self.requests: list[list[ChatMessage]] = []

    @property
    def is_cloud(self) -> bool:
        return False

    def complete_json(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any] | None = None,
        schema_name: str = "result",
        cancel: threading.Event | None = None,
    ) -> str:
        self.requests.append(messages)
        if callable(self._responses):
            item = self._responses(messages)
        else:
            if not self._responses:
                raise LLMError("mock exhausted")
            item = self._responses.pop(0)
        if isinstance(item, LLMError):
            raise item
        return item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
