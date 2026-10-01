"""Extensible medical lexicon (ENT first). Used for STT-error suggestions, safety checks
and benchmarks — never for automatic clinical conclusions."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

from mva.clinical.text import norm, stem, tokens
from mva.paths import resources_dir

# Case endings that turn «Мирамистин» into «Мирамистином»: same drug, not an STT error.
_MED_ENDINGS = (
    "ами",
    "ями",
    "ом",
    "ем",
    "ой",
    "ою",
    "ов",
    "ам",
    "ям",
    "ах",
    "ях",
    "а",
    "у",
    "ы",
    "и",
    "е",
    "ю",
    "я",
)
_MIN_FUZZY_LEN = 6


@dataclass(frozen=True)
class Medication:
    name: str
    inn: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class MedicationMatch:
    surface: str  # as written in the text
    canonical: str  # lexicon name
    similarity: float
    kind: str = (
        "exact"  # exact | inflected (case form of a known name) | fuzzy (probable STT error)
    )


def _med_base(word: str) -> str:
    for ending in _MED_ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 5:
            return word[: -len(ending)]
    return word


def _matches(token: str, patterns: list[str]) -> bool:
    return any(token.startswith(p[:-1]) if p.endswith("*") else token == p for p in patterns)


class Lexicon:
    def __init__(self, data: dict[str, object]) -> None:
        meds = data.get("medications", [])
        assert isinstance(meds, list)
        self.medications = [
            Medication(m["name"], m.get("inn", ""), tuple(m.get("aliases", []))) for m in meds
        ]
        self._med_forms: dict[str, str] = {}
        for med in self.medications:
            for form in (med.name, *med.aliases):
                for part in form.split("+"):
                    if part.strip():
                        self._med_forms[norm(part.strip())] = med.name
        # An INN is its own entity: «азитромицин» is never silently the same as «Сумамед».
        for med in self.medications:
            for part in med.inn.split("+"):
                form = norm(part.strip())
                if form and form not in self._med_forms:
                    self._med_forms[form] = form
        self._med_bases: dict[str, str] = {}
        for form, canonical in self._med_forms.items():
            if " " not in form:
                self._med_bases.setdefault(_med_base(form), canonical)
        self.ent_terms = _normed(data, "ent_terms")
        self.diagnoses = _normed(data, "diagnoses")
        self.risky_qualifiers = _normed(data, "risky_qualifiers")
        self.forbidden_phrases = _normed(data, "forbidden_phrases")
        groups = data.get("synonym_groups", [])
        assert isinstance(groups, list)
        self.synonym_groups = [[norm(x) for x in g] for g in groups]
        self.symptom_heads = _pattern_groups(data.get("symptom_heads", {}))
        self.locations = _pattern_groups(data.get("locations", {}))
        words = _normed(data, "non_medication_words")
        self._common_words = [w for w in words if w]
        for phrase in (*self.ent_terms, *self.diagnoses):
            self._common_words.extend(tokens(phrase))

    # -- medications ---------------------------------------------------------------
    def _is_common_word(self, cand: str) -> bool:
        return any(_matches(w, self._common_words) for w in cand.split())

    def find_medications(self, text: str, cutoff: float = 0.85) -> list[MedicationMatch]:
        """Exact, inflected and (conservatively) near-exact lexicon hits."""
        toks = tokens(text)
        found: list[MedicationMatch] = []
        forms = list(self._med_forms)
        candidates = toks + [f"{a} {b}" for a, b in pairwise(toks)]
        for cand in candidates:
            if len(cand) < 4 or cand[0].isdigit():
                continue
            if cand in self._med_forms:
                found.append(MedicationMatch(cand, self._med_forms[cand], 1.0))
                continue
            if self._is_common_word(cand):
                continue
            if " " not in cand and _med_base(cand) in self._med_bases:
                found.append(
                    MedicationMatch(cand, self._med_bases[_med_base(cand)], 1.0, "inflected")
                )
                continue
            if len(cand) < _MIN_FUZZY_LEN:
                continue
            close = difflib.get_close_matches(cand, forms, n=3, cutoff=cutoff)
            for form in close:
                if len(form) >= _MIN_FUZZY_LEN and abs(len(form) - len(cand)) <= 2:
                    ratio = difflib.SequenceMatcher(None, cand, form).ratio()
                    found.append(MedicationMatch(cand, self._med_forms[form], ratio, "fuzzy"))
                    break
        # Keep the best hit per canonical name.
        best: dict[str, MedicationMatch] = {}
        for match in found:
            if match.canonical not in best or match.similarity > best[match.canonical].similarity:
                best[match.canonical] = match
        return list(best.values())

    def unsupported_medications(self, text: str, reference: str) -> list[str]:
        """Drug names in ``text`` that ``reference`` does not contain (fuzzy ones: verbatim only)."""
        ref_tokens = set(tokens(reference))
        ref_known = {m.canonical for m in self.find_medications(reference) if m.kind != "fuzzy"}
        return [
            m.surface
            for m in self.find_medications(text)
            if not all(part in ref_tokens for part in m.surface.split())
            and not (m.kind != "fuzzy" and m.canonical in ref_known)
        ]

    def suggest_medication(self, word: str, cutoff: float = 0.85) -> str | None:
        w = norm(word)
        if w in self._med_forms or len(w) < _MIN_FUZZY_LEN or self._is_common_word(w):
            return None
        close = difflib.get_close_matches(w, list(self._med_forms), n=1, cutoff=cutoff)
        return self._med_forms[close[0]] if close else None

    def is_medication_word(self, word: str) -> bool:
        return norm(word) in self._med_forms

    # -- groups ----------------------------------------------------------------------
    def expand_stems(self, stems: set[str], text: str = "") -> set[str]:
        """Add synonym-group members for every stem/phrase present."""
        low = norm(text)
        out = set(stems)
        for group in self.synonym_groups:
            hit = any(
                (" " in g and g in low)
                or any(s.startswith(g) or g.startswith(s) for s in stems if len(s) >= 3)
                for g in group
            )
            if hit:
                out.update(g for g in group if " " not in g)
        return out

    def heads_in(self, text: str) -> set[str]:
        """Symptom heads (pain, congestion, discharge, ...) named in the text."""
        toks = tokens(text)
        return {
            n
            for n, patterns in self.symptom_heads.items()
            if any(_matches(t, patterns) for t in toks)
        }

    def locations_in(self, text: str) -> set[str]:
        """Body regions (nose, throat, ear, ...) named in the text."""
        toks = tokens(text)
        return {
            n for n, patterns in self.locations.items() if any(_matches(t, patterns) for t in toks)
        }

    def diagnoses_in(self, text: str) -> list[str]:
        low = norm(text)
        return [
            d
            for d in self.diagnoses
            if any(t.startswith(d) for t in tokens(low)) or (" " in d and d in low)
        ]

    def risky_in(self, text: str) -> list[str]:
        toks = tokens(text)
        return [q for q in self.risky_qualifiers if any(t.startswith(q) for t in toks)]

    def forbidden_in(self, text: str) -> list[str]:
        low = norm(text)
        return [p for p in self.forbidden_phrases if p in low]


def _normed(data: dict[str, object], key: str) -> list[str]:
    raw = data.get(key, [])
    assert isinstance(raw, list)
    return [norm(str(t)) for t in raw]


def _pattern_groups(raw: object) -> dict[str, list[str]]:
    assert isinstance(raw, dict)
    return {str(k): [norm(p) for p in v] for k, v in raw.items()}


def stems_overlap(a: set[str], b: set[str]) -> bool:
    return any(
        x == y or (len(x) >= 4 and len(y) >= 4 and (x.startswith(y) or y.startswith(x)))
        for x in a
        for y in b
    )


@lru_cache(maxsize=4)
def load_lexicon(path: Path | None = None) -> Lexicon:
    target = path or resources_dir() / "lexicon" / "ent_ru.json"
    return Lexicon(json.loads(target.read_text(encoding="utf-8")))


__all__ = ["Lexicon", "MedicationMatch", "load_lexicon", "stem", "stems_overlap"]
