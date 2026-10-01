"""Versioned prompts. The version is recorded in technical metadata, never in the record."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path


def _dir() -> Path:
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        return Path(frozen) / "mva" / "clinical" / "prompts"
    return Path(__file__).resolve().parent


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    return (_dir() / f"{name}.md").read_text(encoding="utf-8")
