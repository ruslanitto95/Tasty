"""STT quality metrics for benchmarks: WER and clinically weighted error rates."""

from __future__ import annotations

from dataclasses import dataclass

from mva.clinical.lexicon import Lexicon
from mva.clinical.text import durations, laterality, norm, numbers, tokens


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = tokens(reference), tokens(hypothesis)
    return _edit_distance(ref, hyp) / len(ref) if ref else float(bool(hyp))


def _term_error(ref_terms: list[str], hyp_text: str) -> tuple[int, int]:
    hyp = norm(hyp_text)
    missed = sum(1 for t in ref_terms if t not in hyp)
    return missed, len(ref_terms)


@dataclass
class ClinicalSTTMetrics:
    wer: float
    medical_term_error_rate: float | None
    medication_error_rate: float | None
    negation_error_rate: float | None
    number_error_rate: float | None
    laterality_error_rate: float | None


def _rate(missed: int, total: int) -> float | None:
    return missed / total if total else None


def clinical_metrics(reference: str, hypothesis: str, lexicon: Lexicon) -> ClinicalSTTMetrics:
    ref = norm(reference)
    terms = [t for t in lexicon.ent_terms if t in ref]
    meds = [m.surface for m in lexicon.find_medications(reference, cutoff=0.99)]
    neg_ref = [t for t in tokens(reference) if t in ("не", "нет", "без", "ни")]
    neg_hyp = [t for t in tokens(hypothesis) if t in ("не", "нет", "без", "ни")]
    ref_nums = sorted(numbers(reference)) + sorted(round(d.days, 2) for d in durations(reference))
    hyp_nums = sorted(numbers(hypothesis)) + sorted(round(d.days, 2) for d in durations(hypothesis))
    ref_side, hyp_side = laterality(reference), laterality(hypothesis)
    t_missed, t_total = _term_error(terms, hypothesis)
    m_missed, m_total = _term_error(meds, hypothesis)
    n_total = len(neg_ref)
    n_missed = max(0, n_total - len(neg_hyp)) + max(0, len(neg_hyp) - n_total)
    num_missed = sum(1 for n in ref_nums if n not in hyp_nums)
    side_total = len(ref_side)
    side_missed = len(ref_side ^ hyp_side) if side_total else 0
    return ClinicalSTTMetrics(
        wer=wer(reference, hypothesis),
        medical_term_error_rate=_rate(t_missed, t_total),
        medication_error_rate=_rate(m_missed, m_total),
        negation_error_rate=_rate(min(n_missed, n_total), n_total),
        number_error_rate=_rate(num_missed, len(ref_nums)),
        laterality_error_rate=_rate(min(side_missed, side_total), side_total),
    )
