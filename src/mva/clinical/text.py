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


_NUMBER_WORD = "|".join(sorted([*_UNITS, *_TENS], key=len, reverse=True))
_NUMBER_RANGE = re.compile(rf"\b((?:\d+|{_NUMBER_WORD}))-((?:\d+|{_NUMBER_WORD}))\b")


def _split_number_ranges(text: str) -> str:
    """'три-четыре' -> 'три - четыре' so that both ends are read as numbers."""
    return _NUMBER_RANGE.sub(r"\1 - \2", text)


def words_to_digits(text: str) -> str:
    """'тридцать семь и пять' -> '37.5'; '37 и 5' -> '37.5'; 'полтора' -> '1.5'."""
    toks = _split_number_ranges(norm(text)).split()
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
        elif tok in ("полтора", "полторы", "полутора"):
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
# Genitive forms for the inverted colloquial order «дня три», «недели две».
_UNIT_DAYS_GENITIVE = [
    (r"(?:дня|дней|суток)", 1.0),
    (r"(?:недели|недель)", 7.0),
    (r"(?:месяца|месяцев)", 30.0),
    (r"(?:года|лет)", 365.0),
]
_ORDINALS = {
    "перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5,
    "шест": 6, "седьм": 7, "восьм": 8, "девят": 9, "десят": 10,
}  # fmt: skip
_ORDINAL_ENDINGS = r"(?:ый|ий|ая|ья|ую|ью|ой|ое|ье|ого|его|ом|ем)"
_NUM = r"\d+(?:\.\d+)?"
# «раз в 2 дня», «в день», «каждые два дня»: a frequency, not how long the illness lasts.
_FREQUENCY_BEFORE = re.compile(r"(?:^|\s)(?:в|кажд\w+)\s+$")
_NOT_A_BARE_DURATION = re.compile(
    r"(?:^|\s)(?:несколько|пару|пара|кажд\w+|в|прошл\w*|следующ\w*)\s*$"
)
_DOSE_AFTER = r"(?!\s*(?:раз|капл|кап|таблет|мг|мл|штук|доз|упаков|ампул))"


def _scan_durations(t: str) -> list[tuple[Duration, tuple[int, int]]]:
    """Duration expressions of an already digit-converted, normalised text with their spans."""
    found: list[tuple[Duration, tuple[int, int]]] = []
    frequencies: list[tuple[int, int]] = []

    def taken(pos: int) -> bool:
        return any(a <= pos < b for _, (a, b) in found) or any(a <= pos < b for a, b in frequencies)

    ordinals = "|".join(_ORDINALS)
    for unit, days in _UNIT_DAYS[:5]:
        for m in re.finditer(rf"\b({ordinals}){_ORDINAL_ENDINGS}\s+({unit})\b", t):
            found.append((Duration(_ORDINALS[m.group(1)] * days), m.span()))
    for unit, days in _UNIT_DAYS:
        for m in re.finditer(
            rf"(?<![\d.,])({_NUM})(?:\s*(?:-|–|до|или)\s*({_NUM}))?\s*{unit}\b", t
        ):
            if taken(m.start()):
                continue
            if _FREQUENCY_BEFORE.search(t[: m.start()]):
                frequencies.append(m.span())
                continue
            first = float(m.group(1))
            last = float(m.group(2)) if m.group(2) else first
            found.append((Duration((first + last) / 2 * days), m.span()))
    for unit, days in _UNIT_DAYS_GENITIVE:
        for m in re.finditer(rf"(?<![\w.])({unit})\s+(\d{{1,2}})(?![.,]?\d)\b{_DOSE_AFTER}", t):
            if taken(m.start()) or not 1 <= float(m.group(2)) <= 12:
                continue
            found.append((Duration(float(m.group(2)) * days), m.span()))
    for m in re.finditer(r"\bпол(?:года|угода)\b", t):
        found.append((Duration(182.0), m.span()))
    # Bare unit without a number: "неделю", "месяц", "сутки".
    for unit, days in _UNIT_DAYS[:4]:
        for m in re.finditer(
            rf"(?<![\w.])(?:уже\s+|около\s+|почти\s+|целую\s+|с\s+)?({unit})\b", t
        ):
            if taken(m.start(1)) or _NOT_A_BARE_DURATION.search(t[: m.start(1)]):
                continue
            found.append((Duration(days), m.span(1)))
    return found


def durations(text: str) -> list[Duration]:
    return [d for d, _ in _scan_durations(words_to_digits(text))]


def strip_durations(text: str) -> str:
    """Digit-converted text with every duration expression removed (dose numbers remain)."""
    t = words_to_digits(text)
    for _, (a, b) in sorted((s for s in _scan_durations(t)), key=lambda x: -x[1][0]):
        t = f"{t[:a]} {t[b:]}"
    return t


# ---- laterality --------------------------------------------------------------
_SIDE_ENDINGS = r"(?:ый|ая|ое|ого|ой|ом|ому|ую|ые|ых|ым|ыми)"
_RIGHT_WORD = rf"(?:справа|прав{_SIDE_ENDINGS})"
_LEFT_WORD = rf"(?:слева|лев{_SIDE_ENDINGS})"
_RIGHT = re.compile(rf"\b{_RIGHT_WORD}\b")
_LEFT = re.compile(rf"\b{_LEFT_WORD}\b")
_RIGHT_AND_LEFT = re.compile(
    rf"\b{_RIGHT_WORD}\s*,?\s*и\s+{_LEFT_WORD}\b|\b{_LEFT_WORD}\s*,?\s*и\s+{_RIGHT_WORD}\b"
)
_SIDE_COMMA_AND = re.compile(
    rf"\b({_RIGHT_WORD}|{_LEFT_WORD})\s*,\s*и\s+(?={_RIGHT_WORD}|{_LEFT_WORD})"
)
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
    if {"right", "left"} <= sides and _RIGHT_AND_LEFT.search(t):
        sides.add("bilateral")
    return sides


def laterality_compatible(claimed: set[str], evidence: set[str]) -> bool:
    """A claimed side must be stated in the evidence and must not narrow «both sides»."""
    if not claimed:
        return True
    ev = set(evidence)
    if {"right", "left"} <= ev:
        ev.add("bilateral")
    cl = set(claimed)
    if "bilateral" in ev:
        if {"right", "left"} <= cl:
            cl = (cl - {"right", "left"}) | {"bilateral"}
        if cl != {"bilateral"}:
            return False
    elif "bilateral" in cl and {"right", "left"} <= ev:
        cl = (cl - {"bilateral"}) | {"right", "left"}
    return cl <= ev


def laterality_compatible_any(claimed: set[str], evidence_sets: list[set[str]]) -> bool:
    """Claim must fit one evidence set; sides spread over separate sets may be combined
    only when none of them says «both sides»."""
    if not claimed:
        return True
    if any(laterality_compatible(claimed, ev) for ev in evidence_sets):
        return True
    union: set[str] = set().union(*evidence_sets) if evidence_sets else set()
    if "bilateral" in union:
        return False
    return claimed <= union and "bilateral" not in claimed


# ---- negation ----------------------------------------------------------------
# "не + verb" phrases that describe a PRESENT symptom rather than its absence.
_SYMPTOM_NEG = r"(?:дыш\w*|слыш\w*|проход\w*|мог\w*|могу|чувству\w*|чувств\w*|спит|сплю|спал\w*|ест|ем|глота\w*|прош\w*|помога\w*|помог\w*|различа\w*|разговарива\w*|сморка\w*|отход\w*)"
# "не знаю / не помню / не мерил / не уточнено / не меньше": uncertainty or a qualifier, never absence.
_NOT_ABSENCE = (
    r"(?:" + _SYMPTOM_NEG + r"|знаю|знает|знаем|помню|помнит|помним|вспомню|вспомнит|уверен\w*"
    r"|мерил\w*|измерял\w*|измеряла|замерял\w*|уточнен\w*|уточнял\w*|указан\w*|запомнил\w*"
    r"|меньше|менее|больше|более)"
)
_ABSENCE = re.compile(
    rf"\bне\s+(?!{_NOT_ABSENCE}\b)[а-я]+|\bнет\b|\bнету\b|\bотрица\w*|\bотсутств\w*|\bникак\w*"
    r"|\bни\s+разу\b|\bбез\s+[а-я]+"
)
_CORRECTION_OPENER = re.compile(
    r"\bнет,\s*(?=точнее|вернее|то есть|на самом деле|ой\b|извин|ошиб|не\s+[а-я]+,?\s+а\s)"
)
_NEGATION_WORD_PREFIXES = ("отрица", "отсутств", "никак", "никогд", "нету")
_CLAUSE_SPLIT = re.compile(r"([.!?;…]+|,|\s(?:а|но|зато|однако)\s)")


def _strip_correction_opener(t: str) -> str:
    """«Нет, точнее левое» corrects an earlier answer; its «нет» is not a denial."""
    return _CORRECTION_OPENER.sub("", t)


def has_absence_negation(text: str) -> bool:
    return bool(_ABSENCE.search(_strip_correction_opener(norm(text))))


def _is_negation_token(tokens_: list[str], i: int) -> bool:
    tok = tokens_[i]
    if tok in ("нет", "нету", "без") or tok.startswith(_NEGATION_WORD_PREFIXES):
        return True
    return tok == "не" and i + 1 < len(tokens_) and not re.fullmatch(_NOT_ABSENCE, tokens_[i + 1])


def _content_tokens(toks: list[str]) -> list[str]:
    return [
        t
        for t in toks
        if t not in STOPWORDS
        and t != "без"
        and not t[0].isdigit()
        and not t.startswith(_NEGATION_WORD_PREFIXES)
    ]


def split_clauses(text: str) -> list[str]:
    """Normalised clauses split at sentence ends, commas and «а/но».

    A lone word before a comma («Кашля, температуры нет») stays with the next clause.
    """
    t = _SIDE_COMMA_AND.sub(r"\1 и ", _strip_correction_opener(norm(text)))
    pieces = _CLAUSE_SPLIT.split(t)
    clauses: list[str] = []
    carry = ""
    for index in range(0, len(pieces), 2):
        piece = f"{carry} {pieces[index].strip()}".strip()
        carry = ""
        sep = pieces[index + 1] if index + 1 < len(pieces) else ""
        if (
            sep == ","
            and len(_content_tokens(tokens(piece))) == 1
            and not has_absence_negation(piece)
        ):
            carry = piece
            continue
        if piece:
            clauses.append(piece)
    if carry:
        clauses.append(carry)
    return clauses


def negated_stems(text: str, window: int = 3) -> set[str]:
    """Content stems within ``window`` tokens of an absence negation, inside the same clause.

    A bare answer («Нет.», «Не было.») negates the clause before it (the doctor's question).
    """
    result: set[str] = set()
    previous = ""
    for clause in split_clauses(text):
        toks = tokens(clause)
        positions = [i for i in range(len(toks)) if _is_negation_token(toks, i)]
        if not positions:
            previous = clause
            continue
        if not _content_tokens(toks):
            result |= {stem(t) for t in _content_tokens(tokens(previous))}
            continue
        for i in positions:
            for j in range(max(0, i - window), min(len(toks), i + window + 1)):
                if j != i and toks[j] in _content_tokens([toks[j]]):
                    result.add(stem(toks[j]))
        previous = clause
    return result


CORRECTION_MARKERS = re.compile(
    r"\b(нет,|ой|точнее|вернее|то есть|а нет|извините|ошибся|ошиблась|поправлю|на самом деле|"
    r"не\s+\w+,\s+а)\b"
)


def has_correction_marker(text: str) -> bool:
    return bool(CORRECTION_MARKERS.search(norm(text)))


def is_question(text: str) -> bool:
    return text.strip().endswith("?")


_QUESTION_STARTS = frozenset(
    [
        "что",
        "как",
        "где",
        "когда",
        "сколько",
        "чем",
        "почему",
        "зачем",
        "какой",
        "какая",
        "какие",
        "какое",
        "куда",
        "откуда",
        "давно",
        "часто",
        "есть",
        "были",
        "принимали",
        "пробовали",
        "лечились",
        "лечили",
        "делали",
        "беспокоит",
    ]
)


def looks_like_question(text: str) -> bool:
    """A doctor's question, also when the ASR dropped the question mark."""
    t = norm(text)
    if t.endswith("?"):
        return True
    words = t.split()
    return bool(words) and ("ли" in tokens(t) or words[0].strip(",") in _QUESTION_STARTS)


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
