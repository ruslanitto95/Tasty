"""Unit tests for the deterministic Russian text rules in mva.clinical.text."""

from __future__ import annotations

import pytest

from mva.clinical.text import (
    Duration,
    durations,
    has_absence_negation,
    has_correction_marker,
    laterality,
    laterality_compatible,
    negated_stems,
    numbers,
    stem,
    temperatures,
    words_to_digits,
)


def days(text: str) -> list[float]:
    return sorted(round(d.days, 3) for d in durations(text))


# ---- words_to_digits / numbers ------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("тридцать семь и пять", [37.5]),
        ("37 и 5", [37.5]),
        ("37,5", [37.5]),
        ("37.5", [37.5]),
        ("38.2", [38.2]),
        ("тридцать восемь и два", [38.2]),
        ("тридцать шесть и шесть", [36.6]),
        ("тридцать шесть и девять", [36.9]),
        ("тридцать девять", [39.0]),
        ("тридцать семь", [37.0]),
        ("38 и 0", [38.0]),
        ("Температура 37 и 5, пульс 80", [37.5, 80.0]),
        ("три", [3.0]),
        ("две", [2.0]),
        ("двадцать один", [21.0]),
        ("сорок пять", [45.0]),
        ("семнадцать", [17.0]),
        ("полтора", [1.5]),
        ("две с половиной", [2.5]),
        ("пару", [2.0]),
        ("Нос заложен", []),
        ("1 и 5", [1.0, 5.0]),
        ("три недели", [3.0]),
    ],
)
def test_numbers(text, expected):
    assert numbers(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("тридцать семь и пять", "37.5"),
        ("37 и 5", "37.5"),
        ("тридцать восемь и два", "38.2"),
        ("пару дней", "2 дней"),
        ("полтора месяца", "1.5 месяца"),
        ("две с половиной недели", "2.5 недели"),
        ("нос заложен", "нос заложен"),
    ],
)
def test_words_to_digits(text, expected):
    assert words_to_digits(text) == expected


def test_temperatures_filters_physiological_range():
    assert temperatures("было тридцать семь и пять, пульс 80, три дня") == {37.5}
    assert temperatures("Температура 36,6") == {36.6}
    assert temperatures("три дня") == set()


# ---- durations --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("неделю", [7.0]),
        ("Получается уже неделю", [7.0]),
        ("две недели", [14.0]),
        ("2 недели", [14.0]),
        ("месяц", [30.0]),
        ("полтора месяца", [45.0]),
        ("три дня", [3.0]),
        ("5 дней", [5.0]),
        ("пару дней", [2.0]),
        ("двое суток", [2.0]),
        ("сутки", [1.0]),
        ("два месяца назад", [60.0]),
        ("пять лет", [1825.0]),
        ("полгода", [182.0]),
        ("двадцать один день", [21.0]),
        ("две с половиной недели", [17.5]),
        ("около 7 дней", [7.0]),
        ("1,5 месяца", [45.0]),
    ],
)
def test_durations_positive(text, expected):
    assert days(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "несколько дней",
        "через несколько дней нос заложило",
        "каждый день",
        "на прошлой неделе",
        "в прошлом месяце",
        "Нос заложен",
        "37,5",
    ],
)
def test_durations_none(text):
    assert days(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("сегодня третий день", [3.0], id="ordinal_third_day"),
        pytest.param("вторая неделя", [14.0], id="ordinal_second_week"),
        pytest.param("дня три", [3.0], id="inverted_dnya_tri"),
        pytest.param("недели две", [14.0], id="inverted_nedeli_dve"),
        pytest.param("около полутора месяцев", [45.0], id="genitive_polutora"),
        pytest.param("полутора недель", [10.5], id="genitive_polutora_weeks"),
    ],
)
def test_durations_spoken_forms(text, expected):
    assert days(text) == expected


@pytest.mark.parametrize("text", ["по две капли три раза в день", "2 раза в день"])
def test_per_day_dose_is_not_a_duration(text):
    assert days(text) == []


def test_hyphenated_spoken_range():
    assert set(days("три-четыре дня")) & {3.0, 3.5, 4.0}


def test_duration_close_to():
    assert Duration(7).close_to(Duration(7.5))
    assert Duration(30).close_to(Duration(33))
    assert not Duration(7).close_to(Duration(14))
    assert not Duration(7).close_to(Duration(49))
    assert not Duration(3).close_to(Duration(30))


# ---- laterality -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("правое ухо", {"right"}),
        ("в правом ухе", {"right"}),
        ("ухо болит справа", {"right"}),
        ("правая половина носа", {"right"}),
        ("правую ноздрю заложило", {"right"}),
        ("левое ухо", {"left"}),
        ("в левом ухе", {"left"}),
        ("болит слева", {"left"}),
        ("левая ноздря", {"left"}),
        ("левой стороной не слышу", {"left"}),
        ("с обеих сторон", {"bilateral"}),
        ("с двух сторон", {"bilateral"}),
        ("оба уха", {"bilateral"}),
        ("в обоих ушах", {"bilateral"}),
        ("обе ноздри", {"bilateral"}),
        ("двусторонний шум", {"bilateral"}),
        ("и справа и слева", {"right", "left", "bilateral"}),
        ("слева и справа", {"right", "left", "bilateral"}),
        ("справа ухо болит, а слева заложено", {"right", "left"}),
        ("Нос заложен", set()),
        ("правильно, правда", set()),
        ("справка и направление", set()),
    ],
)
def test_laterality(text, expected):
    assert laterality(text) == expected


@pytest.mark.parametrize("text", ["Левофлоксацин принимал", "Левомицетин капал"])
def test_laterality_ignores_drug_names(text):
    assert laterality(text) == set()


@pytest.mark.parametrize(
    ("claimed", "evidence", "ok"),
    [
        (set(), {"right"}, True),
        (set(), set(), True),
        ({"right"}, {"right"}, True),
        ({"left"}, {"left"}, True),
        ({"right"}, {"left"}, False),
        ({"left"}, {"right"}, False),
        ({"right"}, set(), False),
        ({"bilateral"}, set(), False),
        ({"bilateral"}, {"right"}, False),
        ({"right"}, {"bilateral"}, False),
        ({"bilateral"}, {"bilateral"}, True),
        ({"bilateral"}, {"right", "left"}, True),
        ({"right", "left"}, {"bilateral"}, True),
        ({"right", "left"}, {"right"}, False),
        pytest.param(
            {"left"},
            {"right", "left"},
            False,
            id="narrowing_left_of_both",
        ),
        pytest.param(
            {"right"},
            {"right", "left"},
            False,
            id="narrowing_right_of_both",
        ),
    ],
)
def test_laterality_compatible(claimed, evidence, ok):
    assert laterality_compatible(claimed, evidence) is ok


# ---- negation ------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "Температуры не было",
        "Кашля нет",
        "Антибиотик не принимал",
        "Без температуры",
        "Аллергию отрицает",
        "Ни разу не было температуры",
        "Никаких выделений",
        "Выделения отсутствуют",
        "Ничем не лечился",
        "Нет, выделений нет",
    ],
)
def test_absence_negation_detected(text):
    assert has_absence_negation(text)


@pytest.mark.parametrize(
    "text",
    [
        "Нос не дышит",
        "Правым ухом не слышу",
        "Насморк не проходит",
        "Не могу глотать",
        "Ночью не сплю",
        "Называн не помог",
        "Лекарство не помогает",
        "Запахи не чувствую",
        "Нос дышит",
        "Нос заложен",
        "Ухо болит",
    ],
)
def test_symptom_negation_is_not_absence(text):
    assert not has_absence_negation(text)


@pytest.mark.parametrize(
    "text", ["Не знаю, мерил ли температуру", "Не помню, принимал ли антибиотик"]
)
def test_not_knowing_is_not_absence(text):
    assert not has_absence_negation(text)


def test_negated_stems_contains_negated_topic():
    assert stem("температуры") in negated_stems("Температуры не было")
    assert stem("кашля") in negated_stems("Кашля нет")
    assert stem("антибиотик") in negated_stems("Антибиотик не принимал")


def test_negated_stems_empty_for_symptom_negation():
    assert negated_stems("Нос не дышит") == set()
    assert negated_stems("Правым ухом не слышу") == set()
    assert negated_stems("Насморк не проходит") == set()
    assert negated_stems("Нос заложен") == set()


@pytest.mark.parametrize(
    ("text", "affirmed"),
    [
        ("Температуры не было, нос заложен", "заложен"),
        ("Кашля нет, горло болит", "горло"),
        ("Нос заложен, температуры нет", "заложен"),
        ("Температуры нет, но нос заложен", "заложен"),
    ],
)
def test_negation_scope_does_not_swallow_affirmed_clause(text, affirmed):
    assert stem(affirmed) not in negated_stems(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Нет, точнее пять дней", True),
        ("Ой, не три, а пять", True),
        ("Извините, левое", True),
        ("На самом деле неделю", True),
        ("Нос заложен", False),
        ("Три дня", False),
    ],
)
def test_correction_marker(text, expected):
    assert has_correction_marker(text) is expected
