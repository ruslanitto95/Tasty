"""Deterministic evidence validation of LLM-extracted facts.

A fact enters the document only if its evidence exists in the transcript and its
value/statement does not contradict that evidence on the critical dimensions:
negation, numbers, duration, laterality, medication names and qualifiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mva.clinical.lexicon import Lexicon, stems_overlap
from mva.clinical.schemas import (
    Certainty,
    ClinicalFact,
    FactCategory,
    Polarity,
    RejectedFact,
    ReviewWarning,
)
from mva.clinical.text import (
    stem,
    content_stems,
    durations,
    fuzzy_contains,
    has_absence_negation,
    is_question,
    laterality,
    laterality_compatible,
    negated_stems,
    norm,
    numbers,
    temperatures,
    tokens,
)
from mva.transcription.models import Transcript, format_ts

# Words the formatter/LLM may use to phrase a fact without them counting as new content.
PHRASING_WORDS = """
считает себя больным больной больна болен заболевание заболевания течение отмечает отмечал
отмечала отмечались жалобы около преимущественно также период момент обращения присоединились
присоединилось присоединился появление наличие стороны области характер эпизод самостоятельно
указывает сообщает пациент пациентка ребенок ребенка мать мама родитель родители слов со тела
повышение повышения до дней дня день суток сутки недели неделю недель месяц месяца месяцев назад
тому началось начался началась начало через после затем далее впоследствии кратковременный
кратковременным эффект эффектом эффекта применял применяла применяли принимал принимала
препарат препарата препараты лечение уточнено название не без отрицает отсутствие наблюдалось
динамика положительная сохраняется сохранялось беспокоит беспокоят беспокоило анамнез текущего
настоящего несколько отмечается отмечалось первые первых двое
"""
PHRASING_STEMS = {stem(w) for w in PHRASING_WORDS.split()}


@dataclass
class ValidationOutcome:
    accepted: list[ClinicalFact] = field(default_factory=list)
    rejected: list[RejectedFact] = field(default_factory=list)
    warnings: list[ReviewWarning] = field(default_factory=list)


def _warning(code: str, message: str, fact: ClinicalFact | None = None) -> ReviewWarning:
    return ReviewWarning(
        code=code, message=message, segment_ids=list(fact.evidence_segment_ids) if fact else []
    )


class EvidenceValidator:
    def __init__(self, lexicon: Lexicon) -> None:
        self.lexicon = lexicon

    def validate(self, facts: list[ClinicalFact], transcript: Transcript) -> ValidationOutcome:
        out = ValidationOutcome()
        by_id = transcript.by_id()
        seen_ids: set[str] = set()
        for index, fact in enumerate(facts):
            if not fact.id or fact.id in seen_ids:
                fact = fact.model_copy(update={"id": f"f{index + 1}"})
            seen_ids.add(fact.id)
            reason, fixed, warnings = self._check(fact, by_id)
            out.warnings.extend(warnings)
            if reason is None and fixed is not None:
                out.accepted.append(fixed)
            else:
                out.rejected.append(RejectedFact(fact=fact, reason=reason or "rejected"))
        return out

    # ------------------------------------------------------------------------------
    def _check(
        self, fact: ClinicalFact, by_id: dict[str, object]
    ) -> tuple[str | None, ClinicalFact | None, list[ReviewWarning]]:
        warnings: list[ReviewWarning] = []
        if not fact.value.strip():
            return "empty_value", None, warnings
        if not fact.evidence_segment_ids or not fact.evidence_quote.strip():
            return "no_evidence", None, warnings
        segments = [by_id.get(sid) for sid in fact.evidence_segment_ids]
        if any(s is None for s in segments):
            return "unknown_segment", None, warnings
        evidence_text = " ".join(s.text for s in segments)  # type: ignore[union-attr]
        if not fuzzy_contains(evidence_text, fact.evidence_quote):
            return "quote_not_in_transcript", None, warnings
        quote = fact.evidence_quote
        # Evaluate against the cited segments (quote may legitimately be an excerpt).
        evidence = evidence_text
        claim = f"{fact.value} {fact.statement}"

        if fact.certainty == Certainty.AMBIGUOUS:
            when = format_ts(segments[0].start_ms)  # type: ignore[union-attr]
            note = fact.ambiguity_note or fact.value
            warnings.append(_warning("ambiguous_fact", f"Неоднозначно ({when}): {note}", fact))
            return "ambiguous", None, warnings

        if all(is_question(s.text) for s in segments):  # type: ignore[union-attr]
            return "evidence_is_question", None, warnings

        # -- medications (may restore the spoken form of a misrecognised name) -----------
        fixed = fact
        if fact.category in (
            FactCategory.TREATMENT,
            FactCategory.TREATMENT_EFFECT,
            FactCategory.OTHER_HISTORY,
            FactCategory.PREVIOUS_VISIT,
        ):
            reason, fixed, med_warnings = self._check_medications(fact, evidence)
            warnings.extend(med_warnings)
            if reason:
                return reason, None, warnings
            fact = fixed
            claim = f"{fact.value} {fact.statement}"

        # -- numbers / temperature / duration -------------------------------------------
        claim_temps = temperatures(claim)
        ev_temps = temperatures(evidence)
        if claim_temps - ev_temps:
            warnings.append(
                _warning("number_unconfirmed", "Не удалось подтвердить значение температуры.", fact)
            )
            return "temperature_mismatch", None, warnings
        claim_durs = durations(claim)
        ev_durs = durations(evidence)
        for dur in claim_durs:
            if not any(dur.close_to(e) for e in ev_durs):
                warnings.append(
                    _warning("number_unconfirmed", "Не удалось подтвердить длительность.", fact)
                )
                return "duration_mismatch", None, warnings
        other_claim = {n for n in numbers(claim) if not 34 <= n <= 43}
        other_ev = set(numbers(evidence))
        dur_values = {round(d.days, 3) for d in claim_durs}
        unexplained = {
            n for n in other_claim if n not in other_ev and round(n, 3) not in dur_values
        }
        if unexplained and not claim_durs:
            return "number_mismatch", None, warnings

        # -- support: the claim must be about what the evidence talks about -----------
        claim_stems = content_stems(fact.value) - PHRASING_STEMS
        ev_stems = self.lexicon.expand_stems(content_stems(evidence), evidence)
        claim_expanded = self.lexicon.expand_stems(claim_stems, fact.value)
        numeric = fact.category in (FactCategory.DURATION, FactCategory.TEMPERATURE) and bool(
            durations(claim) or temperatures(claim)
        )
        if (
            claim_stems
            and not numeric
            and not (
                stems_overlap(claim_stems, ev_stems) or stems_overlap(claim_expanded, ev_stems)
            )
        ):
            return "value_not_supported", None, warnings

        # -- negation -------------------------------------------------------------------
        ev_negated = has_absence_negation(evidence)
        if fact.polarity == Polarity.NEGATIVE:
            if not ev_negated:
                return "negation_not_in_evidence", None, warnings
            if fact.statement and not has_absence_negation(fact.statement):
                return "statement_drops_negation", None, warnings
        else:
            if has_absence_negation(fact.value) or (
                fact.statement and has_absence_negation(fact.statement)
            ):
                return "positive_fact_phrased_negative", None, warnings
            if ev_negated:
                negated = self.lexicon.expand_stems(negated_stems(evidence))
                core = (
                    claim_stems - {"температ", "темпер"}
                    if fact.category != FactCategory.TEMPERATURE
                    else claim_stems
                )
                if (
                    core
                    and stems_overlap(core, negated)
                    and not self._affirmed_elsewhere(core, evidence)
                ):
                    return "evidence_negates_fact", None, warnings

        # -- laterality -----------------------------------------------------------------
        if not laterality_compatible(laterality(claim), laterality(evidence)):
            warnings.append(
                _warning("laterality_mismatch", "Возможное противоречие стороны.", fact)
            )
            return "laterality_mismatch", None, warnings

        # -- qualifiers that change clinical meaning (e.g. "гнойные") ---------------------
        ev_tokens = tokens(evidence)
        for qualifier in self.lexicon.risky_in(claim):
            if not any(t.startswith(qualifier) for t in ev_tokens):
                return f"unsupported_qualifier:{qualifier}", None, warnings

        # -- diagnoses / forbidden content ------------------------------------------------
        if self.lexicon.forbidden_in(claim):
            return "forbidden_content", None, warnings
        diagnoses = self.lexicon.diagnoses_in(claim)
        if diagnoses:
            if fact.category not in (
                FactCategory.OTHER_HISTORY,
                FactCategory.PREVIOUS_VISIT,
                FactCategory.INVESTIGATION,
            ):
                warnings.append(
                    _warning(
                        "diagnosis_mentioned",
                        "В разговоре упоминается диагноз — в жалобы/анамнез не включён.",
                        fact,
                    )
                )
                return "diagnosis_not_allowed", None, warnings
            if not all(any(t.startswith(d) for t in ev_tokens) for d in diagnoses):
                return "diagnosis_not_in_evidence", None, warnings

        if fact.category == FactCategory.TREATMENT and all(
            is_question(s.text) for s in segments[-1:]
        ):  # type: ignore[union-attr]
            return "treatment_from_question", None, warnings
        if quote and fixed is not None and fixed.evidence_quote != quote:
            fixed = fixed.model_copy(update={"evidence_quote": quote})
        return None, fixed, warnings

    def _affirmed_elsewhere(self, core: set[str], evidence: str) -> bool:
        """True if the symptom is also mentioned outside any negation scope."""
        parts = re.split(r"[.!?;]|,\s*(?:а|но)\s", norm(evidence))
        for part in parts:
            if not part.strip() or has_absence_negation(part):
                continue
            if stems_overlap(core, self.lexicon.expand_stems(content_stems(part), part)):
                return True
        return False

    def _check_medications(
        self, fact: ClinicalFact, evidence: str
    ) -> tuple[str | None, ClinicalFact, list[ReviewWarning]]:
        warnings: list[ReviewWarning] = []
        claim = f"{fact.value} {fact.statement}"
        ev_meds = self.lexicon.find_medications(evidence, cutoff=0.7)
        ev_tokens = set(tokens(evidence))
        value, statement = fact.value, fact.statement
        for match in self.lexicon.find_medications(claim):
            surface_in_ev = all(part in ev_tokens for part in match.surface.split())
            if surface_in_ev:
                continue
            spoken = next((m for m in ev_meds if m.canonical == match.canonical), None)
            if spoken is None:
                warnings.append(
                    _warning(
                        "medication_not_in_evidence",
                        "Название препарата не подтверждено расшифровкой.",
                        fact,
                    )
                )
                return "medication_not_in_evidence", fact, warnings
            # LLM "corrected" an STT spelling: restore what was actually said, suggest the name.
            pattern = re.compile(re.escape(match.surface), re.IGNORECASE)
            value = pattern.sub(spoken.surface, value)
            statement = pattern.sub(spoken.surface, statement)
            warnings.append(
                _warning(
                    "medication_uncertain",
                    f"Название препарата распознано неоднозначно: «{spoken.surface}» (возможно, {match.canonical}?)",
                    fact,
                )
            )
        # Spoken near-miss that the LLM kept verbatim: still flag it for the doctor.
        for spoken in ev_meds:
            if spoken.similarity < 1.0 and spoken.surface in norm(claim):
                if not any(w.code == "medication_uncertain" for w in warnings):
                    warnings.append(
                        _warning(
                            "medication_uncertain",
                            f"Название препарата распознано неоднозначно: «{spoken.surface}» "
                            f"(возможно, {spoken.canonical}?)",
                            fact,
                        )
                    )
        return None, fact.model_copy(update={"value": value, "statement": statement}), warnings
