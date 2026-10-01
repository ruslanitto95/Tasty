"""Deterministic evidence validation of LLM-extracted facts.

A fact enters the document only if its evidence exists in the transcript and its
value/statement does not contradict that evidence on the critical dimensions:
negation, numbers, duration, laterality, medication names and qualifiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import pairwise

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
    content_stems,
    durations,
    fuzzy_contains,
    has_absence_negation,
    has_correction_marker,
    is_question,
    laterality,
    laterality_compatible_any,
    looks_like_question,
    negated_stems,
    norm,
    numbers,
    split_clauses,
    stem,
    strip_durations,
    temperatures,
    tokens,
)
from mva.transcription.models import Transcript, TranscriptSegment, format_ts

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

# A diagnosis voiced as a guess is a hypothesis, never a fact about the patient.
_HEDGE = re.compile(
    r"\b(?:похоже|наверное|скорее всего|может быть|возможно|подозр\w*|думаю|предположительно|"
    r"вероятно|не исключ\w*|кажется|полагаю|вроде)\b"
)
_SYMPTOM_CATEGORIES = (FactCategory.COMPLAINT, FactCategory.ONSET, FactCategory.PROGRESSION)
_BARE_YES = re.compile(r"^(?:да|ага|угу|конечно|точно)\W*$")
_COMPLAINT_QUESTION = re.compile(r"беспокоит|жалуетесь|жалобы|что болит|что случилось")
_ANSWER_NO = re.compile(r"^\s*(?:нет|никогда|не было|не бывает|не принимал\w*|не давали)\b")


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
        ordered = sorted(transcript.segments, key=lambda s: s.seq)
        previous = {cur.id: prev.text for prev, cur in pairwise(ordered)}
        seen_ids: set[str] = set()
        for index, fact in enumerate(facts):
            if not fact.id or fact.id in seen_ids:
                fact = fact.model_copy(update={"id": f"f{index + 1}"})
            seen_ids.add(fact.id)
            reason, fixed, warnings = self._check(fact, by_id, previous)
            out.warnings.extend(warnings)
            if reason is None and fixed is not None:
                out.accepted.append(fixed)
            else:
                out.rejected.append(RejectedFact(fact=fact, reason=reason or "rejected"))
        return out

    # ------------------------------------------------------------------------------
    def _check(
        self, fact: ClinicalFact, by_id: dict[str, TranscriptSegment], previous: dict[str, str]
    ) -> tuple[str | None, ClinicalFact | None, list[ReviewWarning]]:
        warnings: list[ReviewWarning] = []
        if not fact.value.strip():
            return "empty_value", None, warnings
        if not fact.evidence_segment_ids or not fact.evidence_quote.strip():
            return "no_evidence", None, warnings
        found = [by_id.get(sid) for sid in fact.evidence_segment_ids]
        segments = [s for s in found if s is not None]
        if len(segments) != len(found):
            return "unknown_segment", None, warnings
        seg_texts = [s.text for s in segments]
        evidence_text = " ".join(seg_texts)
        if not fuzzy_contains(evidence_text, fact.evidence_quote):
            return "quote_not_in_transcript", None, warnings
        quote = fact.evidence_quote
        # Evaluate against the cited segments (quote may legitimately be an excerpt).
        evidence = evidence_text
        claim = f"{fact.value} {fact.statement}"

        if fact.certainty == Certainty.AMBIGUOUS:
            when = format_ts(segments[0].start_ms)
            note = fact.ambiguity_note or fact.value
            warnings.append(_warning("ambiguous_fact", f"Неоднозначно ({when}): {note}", fact))
            return "ambiguous", None, warnings

        if all(is_question(t) for t in seg_texts):
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
        # Numbers outside duration phrases (doses, counts) must come from the speech too.
        other_claim = {n for n in numbers(strip_durations(claim)) if not 34 <= n <= 43}
        if other_claim - set(numbers(evidence)):
            return "number_mismatch", None, warnings

        # -- support: the claim must be about what the evidence talks about -----------
        support = evidence
        context = [
            previous.get(seg.id, "")
            for seg in segments
            if has_correction_marker(seg.text) or _BARE_YES.match(norm(seg.text))
        ]
        if context:
            # «Нет, точнее левое» / «Да.»: the symptom itself was named in the previous segment.
            support = " ".join([*context, evidence])
        if (
            fact.polarity == Polarity.POSITIVE
            and self.lexicon.heads_in(claim)
            and not self.lexicon.heads_in(support)
        ):
            # Elliptical answer («Ночью хуже»): the symptom is the one just discussed.
            # Denials must cite the question themselves.
            support = f"{previous.get(segments[0].id, '')} {support}"
        claim_stems = content_stems(fact.value) - PHRASING_STEMS
        ev_stems = self.lexicon.expand_stems(content_stems(support), support)
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
                    and not self._affirmed_elsewhere(core, seg_texts)
                ):
                    return "evidence_negates_fact", None, warnings

        # -- laterality: per symptom/clause, also against the cited quote ----------------
        if not self._laterality_ok(fact, claim, evidence):
            warnings.append(
                _warning("laterality_mismatch", "Возможное противоречие стороны.", fact)
            )
            return "laterality_mismatch", None, warnings

        # -- qualifiers that change clinical meaning (e.g. "гнойные") ---------------------
        ev_tokens = tokens(evidence)
        for qualifier in self.lexicon.risky_in(claim):
            if not any(t.startswith(qualifier) for t in ev_tokens):
                return f"unsupported_qualifier:{qualifier}", None, warnings

        if not numeric and not self._symptom_supported(fact, claim, support):
            return "value_not_supported", None, warnings

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
            if self._only_hedged(diagnoses, evidence):
                warnings.append(
                    _warning(
                        "diagnosis_mentioned",
                        "В разговоре упоминается предположительный диагноз — в документ не включён.",
                        fact,
                    )
                )
                return "diagnosis_hedged", None, warnings
            if not all(any(t.startswith(d) for t in ev_tokens) for d in diagnoses):
                return "diagnosis_not_in_evidence", None, warnings

        if fact.category == FactCategory.TREATMENT and looks_like_question(segments[-1].text):
            return "treatment_from_question", None, warnings
        if quote and fixed is not None and fixed.evidence_quote != quote:
            fixed = fixed.model_copy(update={"evidence_quote": quote})
        return None, fixed, warnings

    # ------------------------------------------------------------------------------
    def _symptom_supported(self, fact: ClinicalFact, claim: str, support: str) -> bool:
        """The symptom HEAD (pain / congestion / tickle ...) and body region must be spoken:
        sharing only «ухо» does not turn «заложило ухо» into «боль в ухе»."""
        missing = self.lexicon.heads_in(claim) - self.lexicon.heads_in(support)
        if missing and (
            self.lexicon.heads_in(support) or not _COMPLAINT_QUESTION.search(norm(support))
        ):
            # Another symptom was spoken, or nothing but a body part without an open question
            # («Что беспокоит? — Горло»): the claimed head is not established.
            return False
        if fact.category not in _SYMPTOM_CATEGORIES:
            return True
        # A region may be implied («Левая сторона не дышит»), but another one must not be spoken.
        spoken = self.lexicon.locations_in(support)
        return not (spoken and self.lexicon.locations_in(fact.value) - spoken)

    def _laterality_ok(self, fact: ClinicalFact, claim: str, evidence: str) -> bool:
        claimed = laterality(claim)
        if not claimed:
            return True
        if not self._side_fits(claimed, claim, evidence):
            return False
        # The quote is what the extractor pointed at: it must not name the opposite side.
        return not laterality(fact.evidence_quote) or self._side_fits(
            claimed, claim, fact.evidence_quote
        )

    def _side_fits(self, claimed: set[str], claim: str, text: str) -> bool:
        heads = self.lexicon.heads_in(claim)
        locations = self.lexicon.locations_in(claim)

        def about_claim(clause: str) -> bool:
            if heads:
                return bool(heads & self.lexicon.heads_in(clause))
            if locations:
                return bool(locations & self.lexicon.locations_in(clause))
            return True

        relevant = [
            sides
            for clause in split_clauses(text)
            if (sides := laterality(clause)) and about_claim(clause)
        ]
        return laterality_compatible_any(claimed, relevant or [laterality(text)])

    def _only_hedged(self, diagnoses: list[str], evidence: str) -> bool:
        sentences = [
            s
            for s in re.split(r"[.!?;]", norm(evidence))
            if any(
                d in s if " " in d else any(t.startswith(d) for t in tokens(s)) for d in diagnoses
            )
        ]
        return bool(sentences) and all(_HEDGE.search(s) for s in sentences)

    def _affirmed_elsewhere(self, core: set[str], segment_texts: list[str]) -> bool:
        """True if the symptom is also stated by the patient outside any negation scope.

        A doctor's question («Кашель есть?») never affirms anything, even if it lost its «?».
        """
        for index, text in enumerate(segment_texts):
            answered_no = index + 1 < len(segment_texts) and _ANSWER_NO.match(
                norm(segment_texts[index + 1])
            )
            if looks_like_question(text) or answered_no:
                continue
            for sentence in re.findall(r"[^.!?;]+[.!?;]?", text):
                if sentence.strip().endswith("?") or looks_like_question(sentence):
                    continue
                for clause in split_clauses(sentence):
                    if has_absence_negation(clause):
                        continue
                    if stems_overlap(
                        core, self.lexicon.expand_stems(content_stems(clause), clause)
                    ):
                        return True
        return False

    def _check_medications(
        self, fact: ClinicalFact, evidence: str
    ) -> tuple[str | None, ClinicalFact, list[ReviewWarning]]:
        warnings: list[ReviewWarning] = []
        claim = f"{fact.value} {fact.statement}"
        spoken_by_name = {m.canonical: m for m in self.lexicon.find_medications(evidence)}
        ev_tokens = set(tokens(evidence))
        value, statement = fact.value, fact.statement
        for match in self.lexicon.find_medications(claim):
            spoken = spoken_by_name.get(match.canonical)
            if spoken is None:
                warnings.append(
                    _warning(
                        "medication_not_in_evidence",
                        "Название препарата не подтверждено расшифровкой.",
                        fact,
                    )
                )
                return "medication_not_in_evidence", fact, warnings
            verbatim = all(part in ev_tokens for part in match.surface.split())
            if verbatim:
                if match.kind == "fuzzy":
                    # Spoken near-miss that the LLM kept verbatim: still flag it for the doctor.
                    warnings.append(_uncertain_medication(fact, spoken.surface, spoken.canonical))
                continue
            if spoken.kind != "fuzzy":
                continue  # same drug in another case form: keep the text as written
            # LLM "corrected" an STT spelling: restore what was actually said, suggest the name.
            pattern = re.compile(re.escape(match.surface), re.IGNORECASE)
            value = pattern.sub(spoken.surface, value)
            statement = pattern.sub(spoken.surface, statement)
            warnings.append(_uncertain_medication(fact, spoken.surface, match.canonical))
        return None, fact.model_copy(update={"value": value, "statement": statement}), warnings


def _uncertain_medication(fact: ClinicalFact, surface: str, canonical: str) -> ReviewWarning:
    return _warning(
        "medication_uncertain",
        f"Название препарата распознано неоднозначно: «{surface}» (возможно, {canonical}?)",
        fact,
    )
