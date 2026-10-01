"""Formatters. Both receive ONLY validated facts.

* DeterministicFormatter — always available, one sentence per fact, never adds content.
* LLMFormatter — stylistic smoothing; its output must pass FinalFactChecker.
"""

from __future__ import annotations

import json
import threading

from pydantic import ValidationError

from mva.clinical.prompts import load_prompt
from mva.clinical.schemas import (
    ClinicalFact,
    FactCategory,
    FormattedDocument,
    Polarity,
    Temporality,
)
from mva.llm.base import ChatMessage, LLMInvalidResponse, LLMProvider

HISTORY_ORDER = [
    FactCategory.DURATION,
    FactCategory.ONSET,
    FactCategory.PROGRESSION,
    FactCategory.COMPLAINT,  # only past/resolved symptoms land in history
    FactCategory.TEMPERATURE,
    FactCategory.TREATMENT,
    FactCategory.TREATMENT_EFFECT,
    FactCategory.PREVIOUS_VISIT,
    FactCategory.INVESTIGATION,
    FactCategory.OTHER_HISTORY,
]


def _sentence(text: str) -> str:
    text = text.strip().rstrip(";,")
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def is_complaint(fact: ClinicalFact) -> bool:
    return (
        fact.category == FactCategory.COMPLAINT
        and fact.polarity == Polarity.POSITIVE
        and fact.temporality != Temporality.PAST
    )


def in_history(fact: ClinicalFact) -> bool:
    if fact.category == FactCategory.COMPLAINT:
        return fact.temporality == Temporality.PAST and bool(fact.statement)
    if fact.category == FactCategory.OTHER_HISTORY and not fact.relevant_to_current:
        return False
    return True


def history_sentence(fact: ClinicalFact) -> str:
    if fact.statement.strip():
        return _sentence(fact.statement)
    value = fact.value.strip()
    if fact.category == FactCategory.DURATION:
        return _sentence(f"Считает себя больным {value}")
    return _sentence(value)


class DeterministicFormatter:
    def format(
        self, facts: list[ClinicalFact], order: dict[str, int] | None = None
    ) -> tuple[FormattedDocument, dict[str, list[str]]]:
        order = order or {}

        def seq(f: ClinicalFact) -> int:
            return min((order.get(s, 10**9) for s in f.evidence_segment_ids), default=10**9)

        trace: dict[str, list[str]] = {}
        complaints = [f for f in sorted(facts, key=seq) if is_complaint(f)]
        complaint_parts = []
        for fact in complaints:
            part = fact.value.strip().rstrip(".;")
            if part and part.lower() not in (p.lower() for p in complaint_parts):
                complaint_parts.append(part)
        complaints_text = _sentence(
            "; ".join(p[0].lower() + p[1:] if i else p for i, p in enumerate(complaint_parts))
        )
        if complaints_text:
            trace[complaints_text] = [f.id for f in complaints]

        history: list[str] = []
        for category in HISTORY_ORDER:
            for fact in sorted(
                (f for f in facts if f.category == category and in_history(f)), key=seq
            ):
                sentence = history_sentence(fact)
                if sentence and sentence not in history:
                    history.append(sentence)
                    trace[sentence] = [fact.id]
        return FormattedDocument(
            complaints_text=complaints_text, history_text=" ".join(history)
        ), trace


class LLMFormatter:
    PROMPT = "medical_formatter_v1"

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def format(
        self,
        facts: list[ClinicalFact],
        cancel: threading.Event | None = None,
        repair_notes: list[str] | None = None,
        previous: FormattedDocument | None = None,
    ) -> FormattedDocument:
        payload = [
            {
                "id": f.id,
                "category": f.category.value,
                "value": f.value,
                "statement": f.statement,
                "polarity": f.polarity.value,
                "temporality": f.temporality.value,
                "for_complaints": is_complaint(f),
                "for_history": in_history(f) and not is_complaint(f),
            }
            for f in facts
        ]
        messages = [
            ChatMessage("system", load_prompt(self.PROMPT)),
            ChatMessage("user", json.dumps({"validated_facts": payload}, ensure_ascii=False)),
        ]
        if repair_notes and previous is not None:
            messages.append(ChatMessage("assistant", previous.model_dump_json()))
            messages.append(
                ChatMessage(
                    "user",
                    "The text above failed the fact check. Problems:\n- "
                    + "\n- ".join(repair_notes)
                    + "\nRewrite it using ONLY the validated facts. Return the same JSON shape.",
                )
            )
        raw = self.llm.complete_json(
            messages, FormattedDocument.model_json_schema(), "formatted_document", cancel
        )
        try:
            return FormattedDocument.model_validate_json(_strip_fences(raw))
        except ValidationError as exc:
            raise LLMInvalidResponse("formatter returned invalid JSON") from exc


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start >= 0 and end > start else text
