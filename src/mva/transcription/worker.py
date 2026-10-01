"""Background STT worker: queue -> provider.transcribe -> merger, strictly in audio order."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from mva.audio.levels import normalize_segment
from mva.transcription.base import TranscriptionProvider
from mva.transcription.merger import TranscriptMerger
from mva.transcription.models import AudioSegment, TranscriptSegment
from mva.transcription.queue import QueueClosed, SegmentQueue

log = logging.getLogger(__name__)

SegmentCallback = Callable[[TranscriptSegment], None]


class STTWorker:
    def __init__(
        self,
        provider: TranscriptionProvider,
        queue: SegmentQueue,
        merger: TranscriptMerger,
        on_segment: SegmentCallback | None = None,
        on_audio_done: Callable[[AudioSegment], None] | None = None,
    ) -> None:
        self._provider = provider
        self._queue = queue
        self._merger = merger
        self._on_segment = on_segment
        self._on_audio_done = on_audio_done
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stt-worker", daemon=True)
        self.processed = 0
        self.failed = 0
        self.audio_seconds = 0.0
        self.inference_seconds = 0.0

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._cancel.is_set():
            try:
                segment = self._queue.get(timeout=0.25)
            except QueueClosed:
                break
            if segment is None or self._cancel.is_set():
                continue
            self._handle(segment)

    def _handle(self, segment: AudioSegment) -> None:
        try:
            if segment.samples is not None:
                segment.samples = normalize_segment(segment.samples)
            result = self._provider.transcribe(segment)
            self.audio_seconds += segment.duration_s
            self.inference_seconds += result.inference_ms / 1000
        except Exception as exc:
            log.error("STT failed for segment %s: %s", segment.id, type(exc).__name__)
            self.failed += 1
            result = TranscriptSegment(
                id=segment.id,
                seq=segment.seq,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                text="",
                failed=True,
            )
        finally:
            if self._on_audio_done is not None:
                self._on_audio_done(segment)
            segment.samples = None  # release audio as soon as it is transcribed
        stored = self._merger.add(result)
        self.processed += 1
        if self._on_segment is not None:
            self._on_segment(stored)

    def finish(self, timeout: float | None = None) -> bool:
        """Drain remaining segments, then stop. Returns False on timeout."""
        self._queue.close()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def cancel(self) -> None:
        self._cancel.set()
        self._queue.clear()
        self._queue.close()
        self._thread.join(5.0)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()
