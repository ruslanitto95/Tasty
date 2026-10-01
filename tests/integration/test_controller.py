"""Controller integration: real Silero VAD + scripted capture + fake STT (fast, CI-safe)."""

from __future__ import annotations

import gc
import threading

import psutil
import pytest

from mva.core.state_machine import AppState
from mva.llm.base import LLMUnavailable
from mva.llm.mock import MockLLMProvider
from mva.storage.app_settings import SettingsStore
from mva.storage.temp_sessions import TempSessionManager
from mva.transcription.fake import FakeTranscriptionProvider
from mva.transcription.model_manager import ModelManager, load_manifest
from mva.ui.controller import AppController
from tests.helpers import FileCapture
from tests.scripted_llm import respond

TEXTS = {
    1: "Что вас беспокоит?",
    2: "Нос плохо дышит, особенно справа.",
    3: "Давно?",
    4: "Получается уже неделю.",
    5: "Температура была?",
    6: "Первые два дня 37 и 5.",
}


@pytest.fixture
def make_controller(qtbot, tmp_path):
    created = []

    def factory(llm=None, capture=None, ai_enabled=True):
        store = SettingsStore(tmp_path / "settings.json")
        provider = FakeTranscriptionProvider(texts=lambda s: TEXTS.get(s.seq, ""))
        provider.load()
        c = AppController(
            store,
            provider=provider,
            manager=ModelManager(load_manifest(), tmp_path / "models"),
            llm_factory=(lambda _s, _k: llm) if ai_enabled else (lambda _s, _k: None),
            capture=capture or FileCapture(speed=20.0),
            temp_manager=TempSessionManager(tmp_path / "temp"),
        )
        c.startup()
        assert c.state == AppState.IDLE
        created.append(c)
        return c

    yield factory
    for c in created:
        c.shutdown()


def record_visit(qtbot, c: AppController, seconds: float = 1.3) -> None:
    assert c.start_visit()
    qtbot.wait(int(seconds * 1000))
    c.stop_visit()
    qtbot.waitUntil(lambda: c.state == AppState.REVIEW, timeout=20000)


def test_full_visit_produces_validated_draft(qtbot, make_controller, tmp_path):
    c = make_controller(MockLLMProvider(respond))
    drafts = []
    c.draft_ready.connect(drafts.append)
    record_visit(qtbot, c)
    qtbot.waitUntil(lambda: bool(drafts), timeout=5000)
    draft = drafts[0]
    assert draft.complaints_text == "Затруднение носового дыхания, преимущественно справа."
    assert draft.history_text.startswith("Считает себя больным около 7 дней.")
    assert "37,5" in draft.history_text
    assert "Називин" not in draft.history_text  # hallucinated drug rejected
    assert c.visit.transcript is not None and len(c.visit.transcript.segments) == 6
    assert TempSessionManager(tmp_path / "temp").count() == 0  # audio deleted after STT


def test_double_start_creates_one_session(qtbot, make_controller):
    capture = FileCapture(speed=20.0)
    c = make_controller(MockLLMProvider(respond), capture=capture)
    assert c.start_visit()
    assert not c.start_visit()
    c.toggle_recording()  # hotkey: stop
    qtbot.waitUntil(lambda: c.state == AppState.REVIEW, timeout=20000)
    assert capture.starts == 1


def test_offline_llm_keeps_transcript_and_allows_retry(qtbot, make_controller):
    responses = [LLMUnavailable("offline")]
    c = make_controller(MockLLMProvider(lambda m: responses.pop(0) if responses else respond(m)))
    failures, drafts = [], []
    c.analysis_failed.connect(failures.append)
    c.draft_ready.connect(drafts.append)
    record_visit(qtbot, c)
    qtbot.waitUntil(lambda: bool(failures), timeout=5000)
    assert failures == ["llm_unavailable"]
    assert c.visit.transcript is not None and not c.visit.transcript.is_empty()
    c.retry_analysis()
    qtbot.waitUntil(lambda: bool(drafts), timeout=5000)
    assert drafts[0].complaints_text


def test_without_ai_only_transcript(qtbot, make_controller):
    c = make_controller(ai_enabled=False)
    failures = []
    c.analysis_failed.connect(failures.append)
    record_visit(qtbot, c)
    qtbot.waitUntil(lambda: bool(failures), timeout=5000)
    assert failures == ["llm_not_configured"]


def test_microphone_lost_stops_safely(qtbot, make_controller):
    c = make_controller(MockLLMProvider(respond), capture=FileCapture(speed=20.0, lose_after_s=8.0))
    notices = []
    c.notice.connect(lambda code, _d: notices.append(code))
    assert c.start_visit()
    qtbot.waitUntil(lambda: c.state == AppState.REVIEW, timeout=20000)
    assert "microphone_lost" in notices
    assert c.visit.transcript is not None and len(c.visit.transcript.segments) >= 2


def test_cancel_recording_discards_everything(qtbot, make_controller, tmp_path):
    c = make_controller(MockLLMProvider(respond))
    assert c.start_visit()
    qtbot.wait(300)
    c.cancel_recording()
    assert c.state == AppState.IDLE and c.visit is None
    assert TempSessionManager(tmp_path / "temp").count() == 0


def test_cancel_analysis_keeps_transcript(qtbot, make_controller):
    gate = threading.Event()

    def slow(messages):
        gate.wait(5)
        return respond(messages)

    c = make_controller(MockLLMProvider(slow))
    failures = []
    c.analysis_failed.connect(failures.append)
    assert c.start_visit()
    qtbot.wait(1300)
    c.stop_visit()
    qtbot.waitUntil(lambda: c.state == AppState.ANALYZING, timeout=20000)
    c.cancel_analysis()
    gate.set()
    assert c.state == AppState.REVIEW and failures == ["cancelled"]
    assert c.visit.transcript is not None


def test_long_workday_50_visits_no_leaks(qtbot, make_controller, tmp_path):
    c = make_controller(MockLLMProvider(respond), capture=FileCapture(speed=60.0))
    proc = psutil.Process()
    record_visit(qtbot, c, 0.4)
    c.new_visit()
    gc.collect()
    base_threads = threading.active_count()
    base_rss = proc.memory_info().rss
    for _ in range(50):
        record_visit(qtbot, c, 0.4)
        assert c.visit.draft is not None
        c.new_visit()
        assert c.state == AppState.IDLE and c.visit is None
    gc.collect()
    qtbot.wait(300)
    assert threading.active_count() <= base_threads + 2
    assert TempSessionManager(tmp_path / "temp").count() == 0
    growth_mb = (proc.memory_info().rss - base_rss) / 1e6
    assert growth_mb < 80, f"RSS grew by {growth_mb:.1f} MB over 50 visits"
