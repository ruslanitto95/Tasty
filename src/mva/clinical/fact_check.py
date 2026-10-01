"""Final check of formatter output against validated facts (NO NEW FACTS)."""

from __future__ import annotations

from dataclasses import dataclass

from mva.clinical.lexicon import Lexicon, stems_overlap
from mva.clinical.schemas import ClinicalFact, FormattedDocument, Polarity, SentenceTrace
from mva.clinical.text import (
    content_stems,
    durations,
    has_absence_negation,
    laterality,
    laterality_compatible,
    norm,
    numbers,
    split_sentences,
    temperatures,
    tokens,
)
from mva.clinical.validation import PHRASING_STEMS


@dataclass
class Violation:
    section: str
    sentence: str
    problem: str

    def describe(self) -> str:
        return f"[{self.section}] {self.problem}: {self.sentence}"


class FinalFactChecker:
    def __init__(self, lexicon: Lexicon, min_coverage: float = 0.75) -> None:
        self.lexicon = lexicon
        self.min_coverage = min_coverage

    def check(self, doc: FormattedDocument, facts: list[ClinicalFact]) -> list[Violation]:
        violations: list[Violation] = []
        for section, text in (("complaints", doc.complaints_text), ("history", doc.history_text)):
            for sentence in split_sentences(text):
                for clause in sentence.split(";") if section == "complaints" else [sentence]:
                    if clause.strip():
                        violations.extend(self._check_sentence(section, clause.strip(), facts))
        return violations

    def _fact_text(self, fact: ClinicalFact) -> str:
        return f"{fact.value} {fact.statement}"

    def _check_sentence(
        self, section: str, sentence: str, facts: list[ClinicalFact]
    ) -> list[Violation]:
        out: list[Violation] = []
        all_text = " ".join(self._fact_text(f) for f in facts)
        sent_stems = content_stems(sentence) - PHRASING_STEMS
        supporting = [
            f
            for f in facts
            if stems_overlap(
                sent_stems,
                self.lexicon.expand_stems(content_stems(self._fact_text(f)), self._fact_text(f)),
            )
        ]
        if sent_stems and not supporting:
            return [Violation(section, sentence, "no supporting fact")]
        support_text = " ".join(self._fact_text(f) for f in supporting) or all_text
        support_stems = self.lexicon.expand_stems(content_stems(support_text), support_text)
        if sent_stems:
            covered = sum(1 for s in sent_stems if stems_overlap({s}, support_stems))
            if covered / len(sent_stems) < self.min_coverage:
                out.append(Violation(section, sentence, "content not covered by facts"))
        if temperatures(sentence) - temperatures(all_text):
            out.append(Violation(section, sentence, "temperature not in facts"))
        fact_durs = durations(all_text)
        for dur in durations(sentence):
            if not any(dur.close_to(f) for f in fact_durs):
                out.append(Violation(section, sentence, "duration not in facts"))
        nums = {n for n in numbers(sentence) if not 34 <= n <= 43}
        if nums - set(numbers(all_text)) and not durations(sentence):
            out.append(Violation(section, sentence, "number not in facts"))
        if not laterality_compatible(laterality(sentence), laterality(support_text)):
            out.append(Violation(section, sentence, "laterality not in facts"))
        if has_absence_negation(sentence):
            negatives = [f for f in supporting if f.polarity == Polarity.NEGATIVE]
            symptom_neg = any(has_absence_negation(self._fact_text(f)) for f in supporting)
            if not negatives and not symptom_neg:
                out.append(Violation(section, sentence, "negation without negative fact"))
        elif any(f.polarity == Polarity.NEGATIVE for f in supporting) and len(supporting) == 1:
            out.append(Violation(section, sentence, "negative fact rendered as positive"))
        fact_tokens = set(tokens(all_text))
        for med in self.lexicon.find_medications(sentence):
            if not all(part in fact_tokens for part in med.surface.split()):
                out.append(Violation(section, sentence, f"medication not in facts: {med.surface}"))
        for qualifier in self.lexicon.risky_in(sentence):
            if not any(t.startswith(qualifier) for t in fact_tokens):
                out.append(Violation(section, sentence, f"qualifier not in facts: {qualifier}"))
        for diagnosis in self.lexicon.diagnoses_in(sentence):
            if not any(t.startswith(diagnosis) for t in fact_tokens):
                out.append(Violation(section, sentence, f"diagnosis: {diagnosis}"))
        for phrase in self.lexicon.forbidden_in(sentence):
            if phrase not in norm(all_text):
                out.append(Violation(section, sentence, f"forbidden phrase: {phrase}"))
        return out

    def trace(self, doc: FormattedDocument, facts: list[ClinicalFact]) -> list[SentenceTrace]:
        result: list[SentenceTrace] = []
        for section, text in (("complaints", doc.complaints_text), ("history", doc.history_text)):
            for sentence in split_sentences(text):
                clauses = sentence.split(";") if section == "complaints" else [sentence]
                ids: list[str] = []
                for clause in clauses:
                    stems = content_stems(clause) - PHRASING_STEMS
                    scored = []
                    for fact in facts:
                        ft = self._fact_text(fact)
                        fs = self.lexicon.expand_stems(content_stems(ft), ft)
                        score = sum(1 for s in stems if stems_overlap({s}, fs))
                        if score:
                            scored.append((score, fact.id))
                    scored.sort(reverse=True)
                    if scored:
                        top = scored[0][0]
                        ids.extend(fid for score, fid in scored if score == top and fid not in ids)
                result.append(SentenceTrace(section=section, sentence=sentence, fact_ids=ids))
        return result
