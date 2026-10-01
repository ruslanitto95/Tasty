"""A scripted 'LLM' that answers for the bundled self-test dialogue by locating segments.

It lets end-to-end tests exercise the real wiring (GigaAM transcript -> extraction ->
validation -> formatting) without a paid API. It reads segment ids from the request, so
the facts it returns must survive validation against whatever GigaAM actually produced.
"""

from __future__ import annotations

import json

from mva.llm.base import ChatMessage
from mva.llm.mock import MockLLMProvider


def _segments(messages: list[ChatMessage]) -> list[dict]:
    for m in messages:
        if m.role == "user" and m.content.startswith("{") and "segments" in m.content:
            return json.loads(m.content)["segments"]
    return []


def _find(segments: list[dict], *needles: str) -> dict | None:
    for s in segments:
        low = s["text"].lower()
        if all(n in low for n in needles):
            return s
    return None


def respond(messages: list[ChatMessage]) -> dict:
    segments = _segments(messages)
    facts = []
    nose = _find(segments, "нос", "справа")
    if nose:
        facts.append({"id": "f1", "category": "complaint", "value": "затруднение носового дыхания, преимущественно справа",
                      "temporality": "current", "evidence_segment_ids": [nose["id"]], "evidence_quote": nose["text"]})
    week = _find(segments, "неделю")
    if week:
        facts.append({"id": "f2", "category": "duration", "value": "около 7 дней", "statement": "Считает себя больным около 7 дней.",
                      "evidence_segment_ids": [week["id"]], "evidence_quote": week["text"]})
    temp = _find(segments, "первые два дня")
    if temp:
        facts.append({"id": "f3", "category": "temperature", "value": "37,5 °C в первые двое суток", "temporality": "past",
                      "statement": "В первые двое суток отмечал повышение температуры тела до 37,5 °C.",
                      "evidence_segment_ids": [temp["id"]], "evidence_quote": temp["text"]})
    # A hallucination the safety layer must drop: nothing in the dialogue supports it.
    if segments:
        facts.append({"id": "f4", "category": "treatment", "value": "Називин", "statement": "Самостоятельно применял Називин.",
                      "evidence_segment_ids": [segments[0]["id"]], "evidence_quote": segments[0]["text"]})
    return {"facts": facts}


def scripted_llm() -> MockLLMProvider:
    return MockLLMProvider(respond)
