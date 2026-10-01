"""Deterministic provider for tests and UI development (never used in production paths)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from mva.transcription.base import LoadProgress, ProviderInfo, TranscriptionProvider
from mva.transcription.models import AudioSegment, TranscriptSegment


class FakeTranscriptionProvider(TranscriptionProvider):
    def __init__(
        self,
        texts: list[str] | Callable[[AudioSegment], str] | None = None,
        delay_s: float = 0.0,
        fail_on: set[int] | None = None,
    ) -> None:
        self._texts = texts
        self._delay = delay_s
        self._fail_on = fail_on or set()
        self._loaded = False
        self.calls = 0
        self._lock = threading.Lock()

    def load(self, progress: LoadProgress | None = None) -> None:
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def info(self) -> ProviderInfo:
        return ProviderInfo(name="fake", model="fake", revision="0", device="cpu")

    def transcribe(self, segment: AudioSegment) -> TranscriptSegment:
        with self._lock:
            index = self.calls
            self.calls += 1
        if self._delay:
            time.sleep(self._delay)
        if segment.seq in self._fail_on:
            raise RuntimeError("fake failure")
        if callable(self._texts):
            text = self._texts(segment)
        elif self._texts is not None:
            text = self._texts[index] if index < len(self._texts) else ""
        else:
            text = f"фрагмент {segment.seq}"
        return TranscriptSegment(
            id=segment.id,
            seq=segment.seq,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            text=text,
            provider="fake",
            overlap_prev_ms=segment.overlap_prev_ms,
        )
