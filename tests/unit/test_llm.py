from __future__ import annotations

import json
import threading

import httpx
import pytest

from mva.clinical.extraction import ExtractionFailed, FactExtractor
from mva.clinical.pipeline import ClinicalPipeline
from mva.llm.base import ChatMessage, LLMAuthError, LLMCancelled, LLMTimeout, is_local_url
from mva.llm.mock import MockLLMProvider
from mva.llm.openai_compatible import OpenAICompatibleProvider
from tests.helpers import make_transcript


def provider(handler, **kw) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        "https://llm.test/v1",
        "m",
        "sk-test",
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
        **kw,
    )


def ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_success_sends_auth_and_structured_format():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return ok('{"ok": true}')

    assert (
        provider(handler).complete_json([ChatMessage("user", "x")], {"type": "object"})
        == '{"ok": true}'
    )
    body = json.loads(seen[0].content)
    assert seen[0].headers["Authorization"] == "Bearer sk-test"
    assert body["temperature"] == 0 and body["response_format"]["type"] == "json_schema"


def test_retries_with_backoff_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else ok("{}")

    assert provider(handler).complete_json([ChatMessage("user", "x")]) == "{}"
    assert calls["n"] == 3


def test_retries_are_bounded_and_timeout_is_reported():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(LLMTimeout):
        provider(handler).complete_json([ChatMessage("user", "x")])
    assert calls["n"] == 3


def test_auth_error_not_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(401)

    with pytest.raises(LLMAuthError):
        provider(handler).complete_json([ChatMessage("user", "x")])
    assert calls["n"] == 1


def test_json_schema_unsupported_falls_back_to_json_mode():
    formats = []

    def handler(req):
        fmt = json.loads(req.content)["response_format"]["type"]
        formats.append(fmt)
        return httpx.Response(400) if fmt == "json_schema" else ok("{}")

    p = provider(handler)
    p.complete_json([ChatMessage("user", "x")], {"type": "object"})
    assert formats == ["json_schema", "json_object"]


def test_cancel_before_request():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(LLMCancelled):
        provider(lambda r: ok("{}")).complete_json([ChatMessage("user", "x")], cancel=cancel)


def test_url_policy_and_cloud_detection():
    assert is_local_url("http://localhost:11434/v1") and is_local_url("http://192.168.1.10:8000/v1")
    assert not is_local_url("https://api.openai.com/v1")
    with pytest.raises(ValueError):
        OpenAICompatibleProvider("http://api.example.com/v1", "m", None)
    assert OpenAICompatibleProvider("https://api.example.com/v1", "m", None).is_cloud
    assert not OpenAICompatibleProvider("http://127.0.0.1:1234/v1", "m", None).is_cloud


TRANSCRIPT = make_transcript(["Что беспокоит?", "Нос заложен три дня."])
GOOD = {
    "facts": [
        {
            "id": "f1",
            "category": "complaint",
            "value": "затруднение носового дыхания",
            "evidence_segment_ids": ["s0002"],
            "evidence_quote": "Нос заложен",
        }
    ]
}


def test_invalid_json_is_repaired_once():
    llm = MockLLMProvider(["not json", GOOD])
    result = FactExtractor(llm).extract(TRANSCRIPT)
    assert len(result.facts) == 1
    assert "did not match" in llm.requests[1][-1].content


def test_invalid_json_gives_up_without_fallback_document():
    llm = MockLLMProvider(["x", '{"facts": "nope"}', "{"])
    with pytest.raises(ExtractionFailed):
        FactExtractor(llm).extract(TRANSCRIPT)


def test_llm_formatter_new_fact_is_repaired_or_replaced():
    hallucinating = {
        "complaints_text": "Затруднение носового дыхания, гнойные выделения.",
        "history_text": "Температуру отрицает.",
    }
    llm = MockLLMProvider([GOOD, hallucinating, hallucinating])
    draft = ClinicalPipeline(llm, use_llm_formatter=True).run(TRANSCRIPT)
    assert draft.formatter == "deterministic"
    assert draft.complaints_text == "Затруднение носового дыхания."
    assert "гнойн" not in draft.complaints_text and "отрицает" not in draft.history_text
    assert any(w.code == "formatter_fallback" for w in draft.warnings)
    assert "failed the fact check" in llm.requests[2][-1].content


def test_llm_formatter_clean_output_is_used():
    clean = {"complaints_text": "Заложенность носа.", "history_text": ""}
    draft = ClinicalPipeline(MockLLMProvider([GOOD, clean]), use_llm_formatter=True).run(TRANSCRIPT)
    assert draft.formatter == "llm" and draft.complaints_text == "Заложенность носа."


def test_extraction_request_contains_no_identifiers_beyond_transcript():
    llm = MockLLMProvider([GOOD])
    ClinicalPipeline(llm, use_llm_formatter=False).run(TRANSCRIPT)
    payload = json.loads(llm.requests[0][1].content)
    assert set(payload) == {"visit_date", "segments"}
    assert [s["id"] for s in payload["segments"]] == ["s0001", "s0002"]
    assert (
        "Never diagnose" in llm.requests[0][0].content
        or "do NOT diagnose" in llm.requests[0][0].content
    )


def test_llm_formatter_dropping_content_falls_back():
    empty = {"complaints_text": "", "history_text": ""}
    t = make_transcript(["Что беспокоит?", "Нос заложен три дня, справа."])
    facts = {
        "facts": [
            {
                "id": "f1",
                "category": "complaint",
                "value": "заложенность носа справа",
                "evidence_segment_ids": ["s0002"],
                "evidence_quote": "Нос заложен",
            },
            {
                "id": "f2",
                "category": "duration",
                "value": "около 3 дней",
                "statement": "Считает себя больным около 3 дней.",
                "evidence_segment_ids": ["s0002"],
                "evidence_quote": "три дня",
            },
        ]
    }
    draft = ClinicalPipeline(
        MockLLMProvider(
            [facts, empty, {"complaints_text": "Заложенность носа.", "history_text": "Болеет."}]
        ),
        use_llm_formatter=True,
    ).run(t)
    assert draft.formatter == "deterministic"
    assert draft.complaints_text == "Заложенность носа справа."
    assert draft.history_text == "Считает себя больным около 3 дней."
