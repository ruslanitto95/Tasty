"""Extensible medical lexicon (ENT first). Used for STT-error suggestions, safety checks
and benchmarks — never for automatic clinical conclusions."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from mva.clinical.text import norm, stem, tokens
from mva.paths import resources_dir


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


class Lexicon:
    def __init__(self, data: dict[str, object]) -> None:
        meds = data.get("medications", [])
        assert isinstance(meds, list)
        self.medications = [
            Medication(m["name"], m.get("inn", ""), tuple(m.get("aliases", []))) for m in meds
        ]
        self._med_forms: dict[str, str] = {}
        for med in self.medications:
            for form in (med.name, med.inn, *med.aliases):
                for part in form.split("+"):
                    if part.strip():
                        self._med_forms[norm(part.strip())] = med.name
        self.ent_terms = [norm(t) for t in data.get("ent_terms", [])]  # type: ignore[union-attr]
        self.diagnoses = [norm(t) for t in data.get("diagnoses", [])]  # type: ignore[union-attr]
        self.risky_qualifiers = [norm(t) for t in data.get("risky_qualifiers", [])]  # type: ignore[union-attr]
        self.forbidden_phrases = [norm(t) for t in data.get("forbidden_phrases", [])]  # type: ignore[union-attr]
        groups = data.get("synonym_groups", [])
        assert isinstance(groups, list)
        self.synonym_groups = [[norm(x) for x in g] for g in groups]

    # -- medications ---------------------------------------------------------------
    def find_medications(self, text: str, cutoff: float = 0.86) -> list[MedicationMatch]:
        """Exact and near-exact lexicon hits (single and two-word names)."""
        toks = tokens(text)
        found: list[MedicationMatch] = []
        forms = list(self._med_forms)
        candidates = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:], strict=False)]
        for cand in candidates:
            if len(cand) < 4 or cand[0].isdigit():
                continue
            if cand in self._med_forms:
                found.append(MedicationMatch(cand, self._med_forms[cand], 1.0))
                continue
            close = difflib.get_close_matches(cand, forms, n=1, cutoff=cutoff)
            if close:
                ratio = difflib.SequenceMatcher(None, cand, close[0]).ratio()
                found.append(MedicationMatch(cand, self._med_forms[close[0]], ratio))
        # Keep the best hit per canonical name.
        best: dict[str, MedicationMatch] = {}
        for match in found:
            if match.canonical not in best or match.similarity > best[match.canonical].similarity:
                best[match.canonical] = match
        return list(best.values())

    def suggest_medication(self, word: str, cutoff: float = 0.7) -> str | None:
        w = norm(word)
        if w in self._med_forms:
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
