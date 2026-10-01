"""LLM fact extraction with strict schema parsing and bounded repair attempts."""

from __future__ import annotations

import json
import logging
import threading
from datetime import date

from pydantic import ValidationError

from mva.clinical.formatter import _strip_fences
from mva.clinical.prompts import load_prompt
from mva.clinical.schemas import ExtractionResult
from mva.llm.base import ChatMessage, LLMInvalidResponse, LLMProvider
from mva.transcription.models import Transcript, format_ts

log = logging.getLogger(__name__)

PROMPT = "clinical_extraction_v1"


class ExtractionFailed(Exception):
    code = "invalid_structured_response"


def transcript_payload(transcript: Transcript, visit_date: date | None) -> str:
    return json.dumps(
        {
            "visit_date": visit_date.isoformat() if visit_date else None,
            "segments": [
                {"id": s.id, "t": format_ts(s.start_ms), "text": s.text}
                for s in transcript.segments
                if s.text.strip()
            ],
        },
        ensure_ascii=False,
    )


class FactExtractor:
    def __init__(self, llm: LLMProvider, max_repairs: int = 2) -> None:
        self.llm = llm
        self.max_repairs = max_repairs

    def extract(
        self,
        transcript: Transcript,
        visit_date: date | None = None,
        cancel: threading.Event | None = None,
    ) -> ExtractionResult:
        messages = [
            ChatMessage("system", load_prompt(PROMPT)),
            ChatMessage("user", transcript_payload(transcript, visit_date)),
        ]
        schema = ExtractionResult.model_json_schema()
        for attempt in range(self.max_repairs + 1):
            raw = self.llm.complete_json(messages, schema, "clinical_facts", cancel)
            try:
                return ExtractionResult.model_validate_json(_strip_fences(raw))
            except (ValidationError, ValueError) as exc:
                log.warning(
                    "Extraction output invalid (attempt %d): %s", attempt + 1, type(exc).__name__
                )
                problems = _describe(exc)
                messages = [
                    *messages[:2],
                    ChatMessage("assistant", raw[:20000]),
                    ChatMessage(
                        "user",
                        "Your previous output did not match the required JSON schema: "
                        f'{problems}. Return ONLY a corrected JSON object {{"facts": [...]}}.',
                    ),
                ]
        raise ExtractionFailed("structured output invalid after repairs")


def _describe(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:8]
        )
    return "invalid JSON"


__all__ = ["ExtractionFailed", "FactExtractor", "LLMInvalidResponse", "transcript_payload"]
