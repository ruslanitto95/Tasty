"""Suite-level guarantees: dataset size per tag and zero false positives per tag."""

from __future__ import annotations

from collections import defaultdict

import pytest

from .loader import load_cases, run_case, tag_counts

CASES = load_cases()
MINIMUMS = {
    "negation": 20,
    "numbers": 20,
    "laterality": 20,
    "medication": 30,
    "hallucination": 5,
    "ent_nose": 5,
    "ent_ear": 5,
    "throat": 5,
    "pediatrics": 3,
    "therapy": 3,
    "conversation_noise": 8,
    "contradiction": 5,
}


def test_dataset_is_large_enough():
    assert len(CASES) >= 100


def test_case_ids_are_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(("tag", "minimum"), sorted(MINIMUMS.items()))
def test_tag_coverage(tag, minimum):
    assert tag_counts(CASES)[tag] >= minimum


def test_cases_are_well_formed():
    for case in CASES:
        expected = case["expected"]
        fact_ids = {f["id"] for f in case["llm_facts"]}
        assert set(expected["accepted"]) <= fact_ids, case["id"]
        assert set(expected["rejected"]) <= fact_ids, case["id"]
        assert not set(expected["accepted"]) & set(expected["rejected"]), case["id"]
        assert case["transcript"], case["id"]


def test_false_positive_metrics_per_tag():
    """Facts accepted by the pipeline that the dataset says must not be accepted, per tag."""
    false_positives: dict[str, int] = defaultdict(int)
    checked: dict[str, int] = defaultdict(int)
    offenders: dict[str, list[str]] = defaultdict(list)
    for case in CASES:
        draft = run_case(case)
        accepted = {f.id for f in draft.accepted_facts}
        extra = accepted - set(case["expected"]["accepted"])
        for tag in case["tags"]:
            checked[tag] += len(case["llm_facts"])
            false_positives[tag] += len(extra)
            if extra:
                offenders[tag].append(case["id"])
    report = {tag: (false_positives[tag], checked[tag]) for tag in sorted(checked)}
    unexpected = {tag: ids for tag, ids in offenders.items() if ids}
    assert not unexpected, f"false positives: {unexpected}; metrics={report}"
