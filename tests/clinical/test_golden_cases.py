"""Golden clinical cases: scripted LLM facts -> deterministic safety layer -> exact text."""

from __future__ import annotations

import pytest

from .loader import load_cases, run_case

CASES = load_cases()


def _params():
    params = []
    for case in CASES:
        params.append(pytest.param(case, id=case["id"]))
    return params


@pytest.mark.parametrize("case", _params())
def test_golden_case(case):
    draft = run_case(case)
    expected = case["expected"]
    accepted = {f.id for f in draft.accepted_facts}
    rejected = {r.fact.id: r.reason for r in draft.rejected_facts}

    false_positives = accepted - set(expected["accepted"])
    assert not false_positives, f"false-positive facts accepted: {sorted(false_positives)}"
    assert accepted == set(expected["accepted"]), "a supported fact was wrongly dropped"

    for fact_id, prefix in expected["rejected"].items():
        assert fact_id in rejected, (
            f"{fact_id} should be rejected ({prefix}), got {sorted(rejected)}"
        )
        assert rejected[fact_id].startswith(prefix), f"{fact_id}: {rejected[fact_id]} !~ {prefix}"

    assert draft.complaints_text == expected["complaints"]
    assert draft.history_text == expected["history"]
    codes = {w.code for w in draft.warnings}
    assert codes == set(expected["warning_codes"]), f"warnings: {sorted(codes)}"

    document = f"{draft.complaints_text} {draft.history_text}".lower()
    for phrase in case.get("must_not_contain", []):
        assert phrase.lower() not in document, f"forbidden phrase present: {phrase}"
