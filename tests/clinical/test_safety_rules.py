"""Extra safety rules: drug identity (brand vs INN), negative complaints, clause scoping."""

from __future__ import annotations

from typing import Any

import pytest

from mva.clinical.lexicon import load_lexicon
from mva.clinical.text import durations, has_absence_negation, looks_like_question, negated_stems
from mva.transcription.models import segment_id

from .loader import run_case


def case(transcript: list[str], *facts: dict[str, Any]) -> dict[str, Any]:
    llm_facts = [
        {
            "id": f"f{i}",
            "evidence_segment_ids": [segment_id(1)],
            "evidence_quote": transcript[0],
            **f,
        }
        for i, f in enumerate(facts)
    ]
    return {"transcript": transcript, "llm_facts": llm_facts}


@pytest.mark.parametrize(
    ("spoken", "written"),
    [("Сумамед", "Азитромицин"), ("Азитромицин", "Сумамед"), ("Називин", "оксиметазолин")],
)
def test_brand_and_inn_are_not_interchangeable(spoken, written):
    draft = run_case(
        case(
            [f"{spoken} пил."],
            {
                "category": "treatment",
                "value": written,
                "statement": f"Принимал {written}.",
                "evidence_quote": f"{spoken} пил",
            },
        )
    )
    assert not draft.accepted_facts
    assert draft.rejected_facts[0].reason == "medication_not_in_evidence"


def test_inn_is_its_own_lexicon_entity():
    lexicon = load_lexicon()
    names = {m.canonical for m in lexicon.find_medications("азитромицин и сумамед")}
    assert len(names) == 2


def test_negative_complaint_is_rendered_in_history():
    draft = run_case(
        case(
            ["Нос заложен.", "Выделения есть?", "Нет, выделений нет."],
            {
                "category": "complaint",
                "value": "затруднение носового дыхания",
                "evidence_quote": "Нос заложен",
            },
            {
                "category": "complaint",
                "polarity": "negative",
                "value": "выделений из носа нет",
                "evidence_segment_ids": [segment_id(2), segment_id(3)],
                "evidence_quote": "Нет, выделений нет",
            },
        )
    )
    assert draft.complaints_text == "Затруднение носового дыхания."
    assert draft.history_text == "Выделений из носа нет."


def test_question_without_mark_does_not_affirm_a_denied_symptom():
    draft = run_case(
        case(
            ["Кашель есть", "Нет, кашля нет."],
            {
                "category": "complaint",
                "value": "кашель",
                "evidence_segment_ids": [segment_id(1), segment_id(2)],
                "evidence_quote": "Нет, кашля нет",
            },
        )
    )
    assert not draft.accepted_facts


@pytest.mark.parametrize(
    ("text", "expected"),
    [("раз в 2 дня", []), ("каждые три дня", []), ("в течение недели", [7.0])],
)
def test_frequency_phrases_are_not_durations(text, expected):
    assert sorted(round(d.days, 3) for d in durations(text)) == expected


def test_lists_share_one_negation():
    assert {"кашл", "темпер"} <= negated_stems("Кашля, температуры нет")


def test_bare_answer_negates_the_question():
    assert "отриви" in negated_stems("Отривин пробовали? Нет.")


def test_correction_opener_is_not_a_denial():
    assert not has_absence_negation("Нет, точнее левое")


@pytest.mark.parametrize("text", ["Температура была?", "Есть ли кашель", "Сколько дней"])
def test_looks_like_question(text):
    assert looks_like_question(text)
