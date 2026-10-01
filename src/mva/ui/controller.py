"""Application controller: the only place that wires services together.

All heavy work (download, model load, audio, STT, LLM) runs on worker threads; results
are delivered to the GUI thread through Qt signals (queued connections).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from mva.audio.capture import AudioCaptureService, MicrophoneError, resolve_device
from mva.audio.levels import LevelReading
from mva.clinical.pipeline import ClinicalPipeline
from mva.clinical.schemas import DocumentDraft, ReviewWarning
from mva.config import DEFAULTS
from mva.core.state_machine import AppState, InvalidTransition, StateMachine
from mva.llm.base import LLMError, LLMProvider
from mva.llm.openai_compatible import OpenAICompatibleProvider
from mva.paths import dev_recordings_dir, models_dir, temp_root
from mva.security.credentials import CredentialStore
from mva.storage.app_settings import AppSettings, SettingsStore
from mva.storage.temp_sessions import TempSession, TempSessionManager
from mva.transcription.base import TranscriptionProvider
from mva.transcription.gigaam_provider import GigaAMTranscriptionProvider
from mva.transcription.model_manager import (
    DownloadProgress,
    ModelError,
    ModelManager,
    ModelStatus,
    load_manifest,
)
from mva.transcription.models import Transcript, TranscriptSegment
from mva.transcription.session import SessionTranscriber

log = logging.getLogger(__name__)


@dataclass
class Visit:
    """Everything about the current visit lives here and is dropped on «Новый приём»."""

    started_at: datetime = field(default_factory=datetime.now)
    started_monotonic: float = field(default_factory=time.monotonic)
    stopped_monotonic: float | None = None
    temp: TempSession | None = None
    transcriber: SessionTranscriber | None = None
    transcript: Transcript | None = None
    draft: DocumentDraft | None = None
    analysis_error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def elapsed_s(self) -> float:
        end = self.stopped_monotonic or time.monotonic()
        return end - self.started_monotonic


class AppController(QObject):
    state_changed = Signal(str, str)  # previous, current
    model_progress = Signal(str, float)  # stage, fraction (-1 = indeterminate)
    model_ready = Signal(str)  # device
    error = Signal(str, str)  # code, detail
    notice = Signal(str, str)  # code, detail (non-fatal warnings)
    level = Signal(float, bool)  # peak 0..1, low level warning
    segments_changed = Signal(int)
    lagging = Signal(bool)
    finalize_progress = Signal(int, int)  # done, total
    stage = Signal(str)
    draft_ready = Signal(object)  # DocumentDraft | None
    analysis_failed = Signal(str)  # error code
    settings_changed = Signal()

    def __init__(
        self,
        settings_store: SettingsStore,
        credentials: CredentialStore | None = None,
        provider: TranscriptionProvider | None = None,
        manager: ModelManager | None = None,
        llm_factory: Callable[[AppSettings, str | None], LLMProvider | None] | None = None,
        capture: AudioCaptureService | None = None,
        temp_manager: TempSessionManager | None = None,
    ) -> None:
        super().__init__()
        self.settings_store = settings_store
        self.settings = settings_store.load()
        self.credentials = credentials or CredentialStore()
        self.manager = manager or ModelManager(
            load_manifest(self.settings.speech.model_name), models_dir()
        )
        self.provider = provider or GigaAMTranscriptionProvider(
            self.manager, self.settings.speech.device
        )
        self._llm_factory = llm_factory or default_llm_factory
        self.capture = capture or AudioCaptureService()
        self.temp = temp_manager or TempSessionManager(temp_root())
        self.sm = StateMachine()
        self.sm.subscribe(lambda a, b: self.state_changed.emit(a.value, b.value))
        self.visit: Visit | None = None
        self.error_codes: list[str] = []
        self._model_cancel = threading.Event()
        self._threads: list[threading.Thread] = []
        self._poll = QTimer(self)
        self._poll.setInterval(500)
        self._poll.timeout.connect(self._poll_progress)

    # ---- lifecycle -----------------------------------------------------------------
    @property
    def state(self) -> AppState:
        return self.sm.state

    def startup(self) -> None:
        self.temp.cleanup_orphans()
        self.load_model()

    def load_model(self) -> None:
        if self.provider.is_loaded:
            self._set(AppState.IDLE)
            self.model_ready.emit(self.provider.info().device)
            return
        if not self.sm.try_transition(
            {AppState.INITIALIZING, AppState.ERROR, AppState.IDLE}, AppState.MODEL_LOADING
        ):
            return
        self._model_cancel.clear()
        self._spawn("model-init", self._model_worker)

    def _model_worker(self) -> None:
        try:
            if self.manager.status() != ModelStatus.READY:
                self.model_progress.emit("download", 0.0)

                def progress(p: DownloadProgress) -> None:
                    self.model_progress.emit("download", p.fraction)

                self.manager.ensure_downloaded(progress, self._model_cancel)
            self.model_progress.emit("load", -1.0)
            self.provider.load(lambda stage: self.model_progress.emit(stage, -1.0))
        except ModelError as exc:
            self._fail(exc.code, type(exc).__name__)
            return
        except Exception as exc:
            code = getattr(exc, "code", "gigaam_load_failed")
            log.exception("Model initialisation failed")
            self._fail(code, type(exc).__name__)
            return
        self.model_ready.emit(self.provider.info().device)
        self._set(AppState.IDLE)

    def shutdown(self) -> None:
        self._model_cancel.set()
        self._poll.stop()
        if self.capture.running:
            self.capture.stop()
        if self.visit is not None:
            self.visit.cancel.set()
            if self.visit.transcriber is not None:
                self.visit.transcriber.cancel()
        self._drop_visit()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self.temp.cleanup_orphans()

    # ---- recording -------------------------------------------------------------------
    def toggle_recording(self) -> None:
        if self.state == AppState.RECORDING:
            self.stop_visit()
        elif self.state == AppState.IDLE:
            self.start_visit()

    def start_visit(self) -> bool:
        if not self.provider.is_loaded:
            self.notice.emit("model_unavailable", "")
            return False
        if not self.sm.try_transition({AppState.IDLE}, AppState.RECORDING):
            return False  # double click / repeated hotkey
        visit = Visit()
        try:
            visit.temp = self.temp.create()
            dev_dir = (
                dev_recordings_dir()
                if self.settings.developer.enabled and self.settings.developer.save_test_audio
                else None
            )
            visit.transcriber = SessionTranscriber(
                self.provider,
                visit.temp,
                self.settings.speech.vad,
                on_segment=self._on_segment,
                dev_audio_dir=dev_dir,
            )
            visit.transcriber.start()
            device = resolve_device(
                self.settings.audio.device_name, self.settings.audio.device_hostapi
            )
            self.visit = visit
            self.capture.start(
                device, visit.transcriber.feed, self._on_level, self._on_capture_error
            )
        except MicrophoneError as exc:
            self._abort_visit(visit)
            self._set(AppState.IDLE)
            self.error.emit(exc.code, "")
            self.error_codes.append(exc.code)
            return False
        except Exception as exc:
            log.exception("Failed to start visit")
            self._abort_visit(visit)
            self._set(AppState.IDLE)
            self.error.emit("microphone_unavailable", type(exc).__name__)
            return False
        self._poll.start()
        self.segments_changed.emit(0)
        return True

    def stop_visit(self) -> None:
        if not self.sm.try_transition({AppState.RECORDING}, AppState.STOPPING):
            return
        self._spawn("visit-stop", self._stop_worker)

    def _stop_worker(self) -> None:
        visit = self.visit
        assert visit is not None
        assert visit.transcriber is not None
        self.capture.stop()
        visit.stopped_monotonic = time.monotonic()
        self._set(AppState.FINALIZING_STT)
        self.stage.emit("finalizing")
        transcript = visit.transcriber.finish()
        visit.transcript = transcript
        self._delete_audio(visit)
        if visit.cancel.is_set():
            return
        self._after_transcription(visit)

    def cancel_recording(self) -> None:
        if self.state not in (AppState.RECORDING, AppState.STOPPING, AppState.FINALIZING_STT):
            return
        visit = self.visit
        if visit is not None:
            visit.cancel.set()
        self.capture.stop()
        if visit is not None:
            self._abort_visit(visit)
        self._poll.stop()
        self._set(AppState.IDLE)

    def _on_segment(self, _segment: TranscriptSegment) -> None:
        visit = self.visit
        if visit is not None and visit.transcriber is not None:
            self.segments_changed.emit(visit.transcriber.merger.count())

    def _on_level(self, reading: LevelReading) -> None:
        self.level.emit(reading.peak, reading.low_level)

    def _on_capture_error(self, exc: MicrophoneError) -> None:
        self.error_codes.append(exc.code)
        self.notice.emit(exc.code, "")
        if self.state == AppState.RECORDING:
            self.stop_visit()  # keep everything recognised so far

    def _poll_progress(self) -> None:
        visit = self.visit
        if visit is None or visit.transcriber is None:
            self._poll.stop()
            return
        stats = visit.transcriber.stats()
        self.lagging.emit(stats.pending_audio_s > DEFAULTS.stt_queue.lag_warning_s)
        if self.state == AppState.FINALIZING_STT:
            self.finalize_progress.emit(stats.segments_done, stats.segments_detected)
        elif self.state not in (AppState.RECORDING, AppState.STOPPING):
            self._poll.stop()

    # ---- developer modes -----------------------------------------------------------
    def analyze_file(self, path: Path) -> bool:
        if not self.provider.is_loaded or not self.sm.try_transition(
            {AppState.IDLE}, AppState.FINALIZING_STT
        ):
            return False
        visit = Visit()
        self.visit = visit

        def work() -> None:
            from mva.transcription.session import transcribe_file

            try:
                visit.temp = self.temp.create()
                visit.transcript, _ = transcribe_file(
                    self.provider, path, visit.temp, self.settings.speech.vad
                )
            except Exception as exc:
                log.exception("File transcription failed")
                self._fail(getattr(exc, "code", "stt_failed"), type(exc).__name__)
                return
            finally:
                self._delete_audio(visit)
            visit.stopped_monotonic = time.monotonic()
            self._after_transcription(visit)

        self.stage.emit("finalizing")
        self._spawn("file-stt", work)
        return True

    def analyze_text(self, text: str) -> bool:
        if not self.sm.try_transition({AppState.IDLE}, AppState.FINALIZING_STT):
            return False
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        segments = [
            TranscriptSegment(
                id=f"s{i + 1:04d}", seq=i + 1, start_ms=i * 5000, end_ms=i * 5000 + 4000, text=line
            )
            for i, line in enumerate(lines)
        ]
        visit = Visit(transcript=Transcript(segments=segments))
        visit.stopped_monotonic = time.monotonic()
        self.visit = visit
        self._spawn("text-analysis", lambda: self._after_transcription(visit))
        return True

    # ---- analysis --------------------------------------------------------------------
    def llm(self) -> LLMProvider | None:
        return self._llm_factory(self.settings, self.credentials.get_api_key())

    def _after_transcription(self, visit: Visit) -> None:
        transcript = visit.transcript
        if transcript is None or transcript.is_empty():
            visit.draft = DocumentDraft(
                warnings=[
                    ReviewWarning(
                        code="no_speech", message="Речь не распознана — документ не сформирован."
                    )
                ]
            )
            self._set(AppState.REVIEW)
            self.draft_ready.emit(visit.draft)
            return
        llm = self.llm()
        if llm is None:
            # Degraded mode: transcript is kept; doctor can copy it or retry once AI is configured.
            visit.analysis_error = "llm_not_configured"
            self._set(AppState.REVIEW)
            self.analysis_failed.emit("llm_not_configured")
            return
        self._analyze(visit, llm)

    def retry_analysis(self) -> None:
        visit = self.visit
        if visit is None or visit.transcript is None:
            return
        llm = self.llm()
        if llm is None:
            self.analysis_failed.emit("llm_not_configured")
            return
        if not self.sm.try_transition({AppState.REVIEW}, AppState.ANALYZING):
            return
        visit.cancel = threading.Event()
        self._spawn("analysis", lambda: self._analyze(visit, llm, already_analyzing=True))

    def _analyze(self, visit: Visit, llm: LLMProvider, already_analyzing: bool = False) -> None:
        if not already_analyzing:
            self._set(AppState.ANALYZING)
        self.stage.emit("analyzing")
        assert visit.transcript is not None
        pipeline = ClinicalPipeline(llm, use_llm_formatter=self.settings.ai.use_llm_formatter)
        try:
            draft = pipeline.run(visit.transcript, visit.started_at.date(), visit.cancel)
        except LLMError as exc:
            if visit.cancel.is_set():
                return
            visit.analysis_error = exc.code
            self.error_codes.append(exc.code)
            self._set(AppState.REVIEW)
            self.analysis_failed.emit(exc.code)
            return
        except Exception as exc:
            if visit.cancel.is_set():
                return
            code = getattr(exc, "code", "invalid_structured_response")
            log.error("Analysis failed: %s", type(exc).__name__)
            visit.analysis_error = code
            self.error_codes.append(code)
            self._set(AppState.REVIEW)
            self.analysis_failed.emit(code)
            return
        if visit.cancel.is_set() or self.visit is not visit:
            return
        visit.draft = draft
        visit.analysis_error = None
        self._set(AppState.REVIEW)
        self.draft_ready.emit(draft)

    def cancel_analysis(self) -> None:
        visit = self.visit
        if visit is None or self.state != AppState.ANALYZING:
            return
        visit.cancel.set()
        visit.analysis_error = "cancelled"
        self._set(AppState.REVIEW)
        self.analysis_failed.emit("cancelled")

    # ---- new visit / settings ------------------------------------------------------
    def new_visit(self) -> None:
        if self.state in (AppState.RECORDING, AppState.STOPPING, AppState.FINALIZING_STT):
            self.cancel_recording()
            return
        if self.state == AppState.ANALYZING and self.visit is not None:
            self.visit.cancel.set()
        self._drop_visit()
        if self.state in (AppState.REVIEW, AppState.ANALYZING):
            self._set(AppState.IDLE)
        self.segments_changed.emit(0)

    def save_settings(self, settings: AppSettings) -> None:
        self.settings = settings
        self.settings_store.save(settings)
        self.settings_changed.emit()

    # ---- helpers ---------------------------------------------------------------------
    def _set(self, state: AppState) -> None:
        try:
            self.sm.transition(state)
        except InvalidTransition:
            log.warning("Ignored transition %s -> %s", self.sm.state.value, state.value)

    def _fail(self, code: str, detail: str) -> None:
        self.error_codes.append(code)
        self._set(AppState.ERROR)
        self.error.emit(code, detail)

    def _spawn(self, name: str, target: Callable[[], Any]) -> None:
        def run() -> None:
            try:
                target()
            except Exception as exc:
                log.exception("Worker %s crashed", name)
                self._fail("internal_error", type(exc).__name__)

        self._threads = [t for t in self._threads if t.is_alive()]
        thread = threading.Thread(target=run, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _delete_audio(self, visit: Visit) -> None:
        if visit.temp is not None:
            visit.temp.destroy()
            visit.temp = None

    def _abort_visit(self, visit: Visit) -> None:
        visit.cancel.set()
        if visit.transcriber is not None:
            visit.transcriber.cancel()
        self._delete_audio(visit)
        if self.visit is visit:
            self.visit = None

    def _drop_visit(self) -> None:
        visit, self.visit = self.visit, None
        if visit is None:
            return
        if visit.transcriber is not None and visit.transcriber.worker.alive:
            visit.transcriber.cancel()
        self._delete_audio(visit)
        visit.transcriber = None
        visit.transcript = None
        visit.draft = None


def default_llm_factory(settings: AppSettings, api_key: str | None) -> LLMProvider | None:
    ai = settings.ai
    if not ai.enabled or not ai.base_url or not ai.model:
        return None
    try:
        return OpenAICompatibleProvider(ai.base_url, ai.model, api_key, ai.timeout_s)
    except ValueError:
        return None
