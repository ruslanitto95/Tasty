"""Per-visit transcription: 16 kHz stream -> VAD segmenter -> queue -> STT worker -> merger.

Used identically for the live microphone and for WAV files (developer/self-test mode).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mva.audio.segmenter import SpeechSegmenter
from mva.audio.vad import SileroSpeechModel, SpeechProbabilityModel, VoiceActivityDetector
from mva.audio.wav import read_audio_16k, write_wav_16k
from mva.config import DEFAULTS, SAMPLE_RATE
from mva.storage.app_settings import VADSettings
from mva.storage.temp_sessions import TempSession
from mva.transcription.base import TranscriptionProvider
from mva.transcription.merger import TranscriptMerger
from mva.transcription.models import AudioSegment, Transcript, TranscriptSegment
from mva.transcription.queue import SegmentQueue
from mva.transcription.worker import STTWorker

log = logging.getLogger(__name__)


@dataclass
class TranscriptionStats:
    segments_detected: int
    segments_done: int
    segments_failed: int
    pending: int
    pending_audio_s: float
    audio_seconds: float
    inference_seconds: float
    spilled: int

    @property
    def rtf(self) -> float:
        return self.inference_seconds / self.audio_seconds if self.audio_seconds else 0.0


class SessionTranscriber:
    def __init__(
        self,
        provider: TranscriptionProvider,
        temp: TempSession | None,
        vad_settings: VADSettings | None = None,
        speech_model: SpeechProbabilityModel | None = None,
        on_segment: Callable[[TranscriptSegment], None] | None = None,
        dev_audio_dir: Path | None = None,
    ) -> None:
        vs = vad_settings or VADSettings()
        self.merger = TranscriptMerger()
        self.queue = SegmentQueue(temp, DEFAULTS.stt_queue.max_in_memory_segments)
        self._dev_audio_dir = dev_audio_dir
        self.worker = STTWorker(
            provider,
            self.queue,
            self.merger,
            on_segment=on_segment,
            on_audio_done=self._save_dev_audio if dev_audio_dir else None,
        )
        vad = VoiceActivityDetector(
            speech_model or SileroSpeechModel(),
            threshold=vs.threshold,
            neg_threshold=max(0.05, vs.threshold - 0.15),
            min_silence_ms=vs.min_silence_ms,
        )
        self.segmenter = SpeechSegmenter(
            vad,
            self.queue.put,
            pre_roll_ms=vs.pre_roll_ms,
            post_roll_ms=vs.post_roll_ms,
            max_segment_s=provider.max_segment_s,
        )
        self._feed_lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if not self._started:
            self.worker.start()
            self._started = True

    def feed(self, samples_16k: np.ndarray) -> None:
        with self._feed_lock:
            self.segmenter.feed(samples_16k)

    def finish(self, timeout: float | None = None) -> Transcript:
        with self._feed_lock:
            self.segmenter.flush()
        self.worker.finish(timeout)
        return self.merger.transcript()

    def cancel(self) -> None:
        self.worker.cancel()

    def stats(self) -> TranscriptionStats:
        return TranscriptionStats(
            segments_detected=self.segmenter.segments_emitted,
            segments_done=self.worker.processed,
            segments_failed=self.worker.failed,
            pending=self.queue.pending(),
            pending_audio_s=self.queue.pending_audio_s(),
            audio_seconds=self.worker.audio_seconds,
            inference_seconds=self.worker.inference_seconds,
            spilled=self.queue.spilled_total,
        )

    def _save_dev_audio(self, segment: AudioSegment) -> None:
        if self._dev_audio_dir is not None and segment.samples is not None:
            write_wav_16k(self._dev_audio_dir / f"{segment.id}.wav", segment.samples)


def transcribe_file(
    provider: TranscriptionProvider,
    path: Path,
    temp: TempSession | None = None,
    vad_settings: VADSettings | None = None,
    chunk_s: float = 0.1,
) -> tuple[Transcript, TranscriptionStats]:
    audio = read_audio_16k(path)
    session = SessionTranscriber(provider, temp, vad_settings)
    session.start()
    step = int(chunk_s * SAMPLE_RATE)
    for i in range(0, len(audio), step):
        session.feed(audio[i : i + step])
    transcript = session.finish()
    return transcript, session.stats()
