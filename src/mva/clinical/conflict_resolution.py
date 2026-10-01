"""Resolve duplicates and contradictions between validated facts.

Rule: an explicit correction by the patient wins; otherwise conflicting facts are
removed and the doctor is warned. Less is better than invented.
"""

from __future__ import annotations

from mva.clinical.lexicon import Lexicon, stems_overlap
from mva.clinical.schemas import ClinicalFact, FactCategory, Polarity, ReviewWarning
from mva.clinical.text import (
    content_stems,
    durations,
    has_correction_marker,
    laterality,
    norm,
    temperatures,
)
from mva.clinical.validation import PHRASING_STEMS
from mva.transcription.models import Transcript

_MESSAGES = {
    "duration_conflict": "Неоднозначно указана длительность заболевания.",
    "laterality_conflict": "Возможное противоречие стороны.",
    "polarity_conflict": "Противоречивые сведения о наличии симптома.",
    "temperature_conflict": "Противоречивые сведения о температуре.",
}


def _first_seq(fact: ClinicalFact, order: dict[str, int]) -> int:
    return min((order.get(s, 10**9) for s in fact.evidence_segment_ids), default=10**9)


class ConflictResolver:
    def __init__(self, lexicon: Lexicon) -> None:
        self.lexicon = lexicon

    def resolve(
        self, facts: list[ClinicalFact], transcript: Transcript
    ) -> tuple[list[ClinicalFact], list[ReviewWarning]]:
        order = {s.id: s.seq for s in transcript.segments}
        text_of = {s.id: s.text for s in transcript.segments}
        facts = sorted(facts, key=lambda f: _first_seq(f, order))
        facts = self._dedupe(facts)
        warnings: list[ReviewWarning] = []
        removed: set[str] = set()

        def is_correction(f: ClinicalFact) -> bool:
            return f.is_correction or any(
                has_correction_marker(text_of.get(s, "")) for s in f.evidence_segment_ids
            )

        def settle(group: list[ClinicalFact], code: str) -> None:
            latest = group[-1]
            if is_correction(latest):
                removed.update(f.id for f in group[:-1])
                return
            removed.update(f.id for f in group)
            ids = sorted(
                {s for f in group for s in f.evidence_segment_ids}, key=lambda s: order.get(s, 0)
            )
            warnings.append(ReviewWarning(code=code, message=_MESSAGES[code], segment_ids=ids))

        # Durations of the illness: there should be one.
        durs = [f for f in facts if f.category == FactCategory.DURATION]
        distinct: list[ClinicalFact] = []
        for fact in durs:
            d = durations(f"{fact.value} {fact.statement}")
            if not distinct:
                distinct.append(fact)
                continue
            prev = durations(f"{distinct[-1].value} {distinct[-1].statement}")
            same = (d and prev and any(a.close_to(b) for a in d for b in prev)) or norm(
                fact.value
            ) == norm(distinct[-1].value)
            if not same:
                distinct.append(fact)
            else:
                removed.add(fact.id)
        if len(distinct) > 1:
            settle(distinct, "duration_conflict")

        # Temperature maxima that disagree (same period not tracked: be conservative).
        temps = [
            f
            for f in facts
            if f.category == FactCategory.TEMPERATURE and f.polarity == Polarity.POSITIVE
        ]
        temp_values = [temperatures(f"{f.value} {f.statement}") for f in temps]
        if len({frozenset(t) for t in temp_values if t}) > 1 and any(
            has_correction_marker(text_of.get(s, "")) for f in temps for s in f.evidence_segment_ids
        ):
            settle(
                [f for f, t in zip(temps, temp_values, strict=True) if t], "temperature_conflict"
            )

        # Same symptom: different side or opposite polarity.
        symptomatic = [
            f
            for f in facts
            if f.category
            in (FactCategory.COMPLAINT, FactCategory.TEMPERATURE, FactCategory.PROGRESSION)
        ]
        for i, a in enumerate(symptomatic):
            for b in symptomatic[i + 1 :]:
                if a.id in removed or b.id in removed:
                    continue
                sa = content_stems(a.value) - PHRASING_STEMS
                sb = content_stems(b.value) - PHRASING_STEMS
                if not stems_overlap(sa, sb):
                    continue
                if a.polarity != b.polarity and a.temporality == b.temporality:
                    settle([a, b], "polarity_conflict")
                    continue
                la, lb = laterality(a.value), laterality(b.value)
                if la and lb and la != lb and a.category == b.category == FactCategory.COMPLAINT:
                    settle([a, b], "laterality_conflict")
        kept = [f for f in facts if f.id not in removed]
        return kept, warnings

    @staticmethod
    def _dedupe(facts: list[ClinicalFact]) -> list[ClinicalFact]:
        result: list[ClinicalFact] = []
        index: dict[tuple[str, str, str], int] = {}
        for fact in facts:
            key = (fact.category.value, fact.polarity.value, norm(fact.value))
            if key in index:
                kept = result[index[key]]
                merged = list(dict.fromkeys(kept.evidence_segment_ids + fact.evidence_segment_ids))
                result[index[key]] = kept.model_copy(update={"evidence_segment_ids": merged})
                continue
            index[key] = len(result)
            result.append(fact)
        return result
