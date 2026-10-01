"""Deterministic Russian text analysis used by validation and fact checking.

Deliberately conservative: when a rule cannot confirm something, callers omit or warn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-zа-яё]+|\d+(?:[.,]\d+)?", re.IGNORECASE)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(norm(text))


_ENDINGS = sorted(
    [
        "ями",
        "ами",
        "его",
        "ого",
        "ему",
        "ому",
        "ыми",
        "ими",
        "иях",
        "ях",
        "ах",
        "ов",
        "ев",
        "ей",
        "ой",
        "ий",
        "ый",
        "ая",
        "яя",
        "ое",
        "ее",
        "ые",
        "ие",
        "ую",
        "юю",
        "ом",
        "ем",
        "ам",
        "ям",
        "ия",
        "ие",
        "ью",
        "ться",
        "тся",
        "ешь",
        "ет",
        "ут",
        "ют",
        "ит",
        "ат",
        "ят",
        "ил",
        "ыл",
        "ла",
        "ли",
        "ло",
        "ть",
        "ти",
        "а",
        "я",
        "о",
        "е",
        "ы",
        "и",
        "у",
        "ю",
        "ь",
        "й",
    ],
    key=len,
    reverse=True,
)


def stem(word: str) -> str:
    word = norm(word)
    if word.isdigit() or len(word) <= 3:
        return word
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            word = word[: -len(ending)]
            break
    return word[:6]


STOPWORDS = set(
    [
        "и",
        "а",
        "но",
        "или",
        "в",
        "во",
        "на",
        "с",
        "со",
        "к",
        "ко",
        "по",
        "о",
        "об",
        "от",
        "до",
        "из",
        "у",
        "за",
        "при",
        "для",
        "без",
        "над",
        "под",
        "про",
        "через",
        "что",
        "как",
        "так",
        "же",
        "ли",
        "бы",
        "то",
        "это",
        "этот",
        "эта",
        "эти",
        "тот",
        "та",
        "те",
        "там",
        "тут",
        "здесь",
        "уже",
        "еще",
        "ещё",
        "вот",
        "ну",
        "да",
        "нет",
        "не",
        "ни",
        "я",
        "мы",
        "вы",
        "ты",
        "он",
        "она",
        "оно",
        "они",
        "мне",
        "меня",
        "мой",
        "моя",
        "мое",
        "мои",
        "его",
        "ее",
        "её",
        "их",
        "вас",
        "вам",
        "нас",
        "нам",
        "был",
        "была",
        "было",
        "были",
        "есть",
        "быть",
        "будет",
        "очень",
        "всё",
        "все",
        "весь",
        "когда",
        "где",
        "куда",
        "чем",
        "чего",
        "кто",
        "который",
        "которая",
        "которые",
        "потом",
        "сначала",
        "теперь",
        "сейчас",
        "тогда",
        "просто",
        "короче",
        "наверное",
        "типа",
        "вроде",
        "как-то",
        "какой",
        "какая",
        "какие",
        "какое",
        "такой",
        "такая",
        "такие",
        "такое",
        "доктор",
    ]
)


def content_stems(text: str) -> set[str]:
    return {stem(t) for t in tokens(text) if t not in STOPWORDS and not t[0].isdigit()}


# ---- numbers -----------------------------------------------------------------
_UNITS = {
    "ноль": 0, "один": 1, "одна": 1, "одну": 1, "одного": 1, "одной": 1, "одни": 1,
    "два": 2, "две": 2, "двух": 2, "три": 3, "трех": 3, "четыре": 4, "четырех": 4,
    "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7, "восемь": 8,
    "восьми": 8, "девять": 9, "девяти": 9, "десять": 10, "десяти": 10,
    "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
    "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
    "девятнадцать": 19, "пару": 2, "пара": 2, "пары": 2, "двое": 2, "трое": 3,
    "четверо": 4, "пятеро": 5, "оба": 2, "обе": 2,
}  # fmt: skip
_TENS = {
    "двадцать": 20, "двадцати": 20, "тридцать": 30, "тридцати": 30, "сорок": 40,
    "сорока": 40, "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70,
    "восемьдесят": 80, "девяносто": 90,
}  # fmt: skip


def words_to_digits(text: str) -> str:
    """'тридцать семь и пять' -> '37.5'; '37 и 5' -> '37.5'; 'полтора' -> '1.5'."""
    toks = norm(text).split()
    out: list[str] = []
    i = 0
    while i < len(toks):
        tok = toks[i].strip(",.;:!?")
        value: float | None = None
        consumed = 1
        if tok in _TENS:
            value = _TENS[tok]
            if i + 1 < len(toks) and toks[i + 1].strip(",.;:!?") in _UNITS:
                value += _UNITS[toks[i + 1].strip(",.;:!?")]
                consumed = 2
        elif tok in _UNITS:
            value = _UNITS[tok]
        elif tok in ("полтора", "полторы"):
            value = 1.5
        elif re.fullmatch(r"\d+(?:[.,]\d+)?", tok):
            value = float(tok.replace(",", "."))
        if value is None:
            out.append(toks[i])
            i += 1
            continue
        j = i + consumed
        rest = [t.strip(",.;:!?") for t in toks[j : j + 3]]
        if rest[:2] == ["с", "половиной"]:
            value += 0.5
            consumed += 2
        elif (
            34 <= value <= 42
            and float(value).is_integer()
            and len(rest) >= 2
            and rest[0] in ("и", "и,")
        ):
            frac = rest[1]
            digit = _UNITS.get(frac) if frac in _UNITS else (int(frac) if frac.isdigit() else None)
            if digit is not None and 0 <= digit <= 9:
                value += digit / 10
                consumed += 2
        out.append(_fmt_num(value))
        i += consumed
    return " ".join(out)


def _fmt_num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def numbers(text: str) -> list[float]:
    converted = words_to_digits(text)
    return [float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", converted)]


def temperatures(text: str) -> set[float]:
    return {round(n, 1) for n in numbers(text) if 34.0 <= n <= 43.0}


# ---- durations -----------------------------------------------------------------
@dataclass(frozen=True)
class Duration:
    days: float

    def close_to(self, other: Duration) -> bool:
        return abs(self.days - other.days) <= max(0.5, 0.15 * max(self.days, other.days))


_UNIT_DAYS = [
    (r"сут(?:ки|ок)", 1.0),
    (r"(?:день|дня|дней|дн)", 1.0),
    (r"недел\w*", 7.0),
    (r"месяц\w*", 30.0),
    (r"(?:год|года|лет)", 365.0),
    (r"час\w*", 1 / 24),
]


def durations(text: str) -> list[Duration]:
    t = words_to_digits(text)
    found: list[Duration] = []
    taken: list[tuple[int, int]] = []
    for unit, days in _UNIT_DAYS:
        for m in re.finditer(rf"(\d+(?:\.\d+)?)\s*(?:-|–|до)?\s*(?:\d+\s*)?{unit}\b", t):
            found.append(Duration(float(m.group(1)) * days))
            taken.append(m.span())
    for m in re.finditer(r"\bполгода\b", t):
        found.append(Duration(182.0))
        taken.append(m.span())
    # Bare unit without a number: "неделю", "месяц", "сутки".
    for unit, days in _UNIT_DAYS[:4]:
        for m in re.finditer(
            rf"(?<![\w.])(?:уже\s+|около\s+|почти\s+|целую\s+|с\s+)?({unit})\b", t
        ):
            if any(a <= m.start(1) < b for a, b in taken):
                continue
            if re.search(
                r"(несколько|пару|пара|каждый|каждую|в\s+день|прошл\w*|следующ\w*)\s*$",
                t[: m.start(1)],
            ):
                continue
            found.append(Duration(days))
    return found


# ---- laterality --------------------------------------------------------------
_RIGHT = re.compile(r"\b(справа|право\w*|прав(?:ый|ая|ое|ого|ой|ом|ому|ую|ые|ых|ым|ыми))\b")
_LEFT = re.compile(r"\b(слева|лево\w*|лев(?:ый|ая|ое|ого|ой|ом|ому|ую|ые|ых|ым|ыми))\b")
_BOTH = re.compile(
    r"\b(с\s+обеих\s+сторон|с\s+двух\s+сторон|обе\s+стороны|двусторонн\w*|оба\s+уха|"
    r"обоих\s+уш\w*|обе\s+ноздри|в\s+обоих|обеих|оба)\b"
)


def laterality(text: str) -> set[str]:
    t = norm(text)
    sides = set()
    if _BOTH.search(t):
        sides.add("bilateral")
    if _RIGHT.search(t):
        sides.add("right")
    if _LEFT.search(t):
        sides.add("left")
    if {"right", "left"} <= sides and re.search(
        r"(справа|прав\w*)\s+и\s+(слева|лев\w*)|(слева|лев\w*)\s+и\s+(справа|прав\w*)", t
    ):
        sides.add("bilateral")
    return sides


def laterality_compatible(claimed: set[str], evidence: set[str]) -> bool:
    """A claimed side must be stated in the evidence; "both sides" == "right and left"."""
    if not claimed:
        return True
    ev = set(evidence)
    if {"right", "left"} <= ev:
        ev.add("bilateral")
    cl = set(claimed)
    if "bilateral" in ev and {"right", "left"} <= cl:
        cl = (cl - {"right", "left"}) | {"bilateral"}
    return cl <= ev


# ---- negation ----------------------------------------------------------------
# "не + verb" phrases that describe a PRESENT symptom rather than its absence.
_SYMPTOM_NEG = r"(?:дыш\w*|слыш\w*|проход\w*|мог\w*|могу|чувству\w*|чувств\w*|спит|сплю|спал\w*|ест|ем|глота\w*|прош\w*|помога\w*|помог\w*|различа\w*|разговарива\w*|сморка\w*|отход\w*)"
_ABSENCE = re.compile(
    rf"\bне\s+(?!{_SYMPTOM_NEG}\b)[а-я]+|\bнет\b|\bотрица\w*|\bотсутств\w*|\bникак\w*|\bни\s+разу\b|\bбез\s+[а-я]+"
)


def has_absence_negation(text: str) -> bool:
    return bool(_ABSENCE.search(norm(text)))


def negated_stems(text: str, window: int = 3) -> set[str]:
    """Content stems within ``window`` tokens of an absence negation."""
    t = norm(text)
    toks = tokens(t)
    result: set[str] = set()
    for i, tok in enumerate(toks):
        is_neg = tok in ("нет", "без") or tok.startswith(("отрица", "отсутств", "никак"))
        if tok == "не" and i + 1 < len(toks) and not re.fullmatch(_SYMPTOM_NEG, toks[i + 1]):
            is_neg = True
        if not is_neg:
            continue
        for j in range(max(0, i - window), min(len(toks), i + window + 1)):
            if j != i and toks[j] not in STOPWORDS and not toks[j][0].isdigit():
                result.add(stem(toks[j]))
    return result


CORRECTION_MARKERS = re.compile(
    r"\b(нет,|ой|точнее|вернее|то есть|а нет|извините|ошибся|ошиблась|поправлю|на самом деле|"
    r"не\s+\w+,\s+а)\b"
)


def has_correction_marker(text: str) -> bool:
    return bool(CORRECTION_MARKERS.search(norm(text)))


def is_question(text: str) -> bool:
    return text.strip().endswith("?")


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def fuzzy_contains(haystack: str, needle: str, threshold: float = 0.8) -> bool:
    """True if most needle tokens appear in order inside haystack (tolerates STT/LLM noise)."""
    h = tokens(haystack)
    n = tokens(needle)
    if not n:
        return False
    hs = [stem(x) for x in h]
    ns = [stem(x) for x in n]
    matched = 0
    pos = 0
    for token in ns:
        try:
            pos = hs.index(token, pos) + 1
            matched += 1
        except ValueError:
            continue
    return matched / len(ns) >= threshold
