"""Load clinical regression cases and run them through the pipeline with a scripted LLM."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mva.clinical.pipeline import ClinicalPipeline
from mva.clinical.schemas import DocumentDraft
from mva.llm.mock import MockLLMProvider
from mva.transcription.models import Transcript, TranscriptSegment, segment_id

CASES_DIR = Path(__file__).parent / "cases"
_SHORT_ID = re.compile(r"s(\d{1,3})")


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(CASES_DIR.glob("*.json")):
        for case in json.loads(path.read_text(encoding="utf-8")):
            case["_file"] = path.stem
            cases.append(case)
    return cases


def real_segment_id(short: str) -> str:
    """Cases cite segments as s1, s2, ... (1-based); the pipeline uses s0001, s0002, ..."""
    match = _SHORT_ID.fullmatch(short)
    return segment_id(int(match.group(1))) if match else short


def build_transcript(case: dict[str, Any]) -> Transcript:
    segments = []
    for i, text in enumerate(case["transcript"]):
        start = i * 5000
        segments.append(
            TranscriptSegment(
                id=segment_id(i + 1), seq=i + 1, start_ms=start, end_ms=start + 4000, text=text
            )
        )
    return Transcript(segments=segments)


def scripted_extraction(case: dict[str, Any]) -> dict[str, Any]:
    facts = []
    for fact in case["llm_facts"]:
        fact = dict(fact)
        fact["evidence_segment_ids"] = [real_segment_id(s) for s in fact["evidence_segment_ids"]]
        facts.append(fact)
    return {"facts": facts}


def run_case(case: dict[str, Any]) -> DocumentDraft:
    llm = MockLLMProvider([scripted_extraction(case)])
    return ClinicalPipeline(llm, use_llm_formatter=False).run(build_transcript(case))


def tag_counts(cases: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for case in cases:
        counts.update(set(case["tags"]))
    return counts
