"""Shared test helpers (importable from any test package)."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from mva.audio.capture import InputDevice, MicrophoneError, MicrophoneLost
from mva.audio.wav import read_audio_16k
from mva.paths import self_test_wav
from mva.transcription.models import Transcript, TranscriptSegment


def make_transcript(lines: list[str]) -> Transcript:
    return Transcript(
        segments=[
            TranscriptSegment(
                id=f"s{i + 1:04d}", seq=i + 1, start_ms=i * 5000, end_ms=i * 5000 + 4000, text=t
            )
            for i, t in enumerate(lines)
        ]
    )


class ScriptedProb:
    """Speech probability model returning a predefined sequence (1 value per 512-sample frame)."""

    def __init__(self, probs: list[float]) -> None:
        self.probs = probs
        self.i = 0

    def probability(self, frame: np.ndarray) -> float:
        value = self.probs[self.i] if self.i < len(self.probs) else 0.0
        self.i += 1
        return value

    def reset(self) -> None:
        pass


class FileCapture:
    """Stands in for AudioCaptureService: streams a WAV in real-time-ish blocks on a thread."""

    def __init__(
        self, path: Path | None = None, speed: float = 8.0, lose_after_s: float | None = None
    ) -> None:
        self.audio = read_audio_16k(path or self_test_wav())
        self.speed = speed
        self.lose_after_s = lose_after_s
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.running = False
        self.starts = 0
        self.device: InputDevice | None = None

    def start(
        self,
        device: InputDevice | None,
        on_block: Callable[[np.ndarray], None],
        on_level: Callable | None = None,
        on_error: Callable[[MicrophoneError], None] | None = None,
    ) -> InputDevice | None:
        if self.running:
            raise RuntimeError("already running")
        self.starts += 1
        self._stop.clear()
        self.running = True

        def pump() -> None:
            step = 1600
            for i in range(0, len(self.audio), step):
                if self._stop.is_set():
                    break
                if self.lose_after_s is not None and i / 16000 >= self.lose_after_s:
                    self.running = False
                    if on_error:
                        on_error(MicrophoneLost("test"))
                    return
                on_block(self.audio[i : i + step])
                time.sleep(step / 16000 / self.speed)
            # Keep "recording" silence until stopped, like a real microphone.
            while not self._stop.is_set():
                on_block(np.zeros(1600, dtype=np.float32))
                time.sleep(0.1 / self.speed)

        self._thread = threading.Thread(target=pump, daemon=True)
        self._thread.start()
        return device

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)
        self.running = False


def models_dir_for_tests() -> Path:
    """Real-model tests share one cache; set MVA_MODELS_DIR to reuse an existing download."""
    import tempfile

    return Path(os.environ.get("MVA_MODELS_DIR", Path(tempfile.gettempdir()) / "mva-test-models"))
