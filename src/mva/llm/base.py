"""LLM abstraction. The clinical pipeline depends only on this interface."""

from __future__ import annotations

import ipaddress
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


class LLMError(Exception):
    code = "llm_unavailable"


class LLMUnavailable(LLMError):
    code = "llm_unavailable"


class LLMAuthError(LLMError):
    code = "invalid_credentials"


class LLMTimeout(LLMError):
    code = "timeout"


class LLMInvalidResponse(LLMError):
    code = "invalid_structured_response"


class LLMCancelled(LLMError):
    code = "cancelled"


@dataclass
class ChatMessage:
    role: str
    content: str


class LLMProvider(ABC):
    name: str = "llm"

    @property
    @abstractmethod
    def is_cloud(self) -> bool:
        """True if text leaves this computer (shown to the doctor in the UI)."""

    @abstractmethod
    def complete_json(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any] | None = None,
        schema_name: str = "result",
        cancel: threading.Event | None = None,
    ) -> str:
        """Return the raw JSON text produced by the model."""

    def test_connection(self) -> None:
        self.complete_json([ChatMessage("user", 'Reply with {"ok": true}')])


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").strip("[]").lower()
    if host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local
