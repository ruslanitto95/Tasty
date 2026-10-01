"""Silero VAD (local, bundled with the silero-vad package) + hysteresis state."""

from __future__ import annotations

import logging
import threading
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from mva.config import SAMPLE_RATE, VAD_FRAME_SAMPLES

log = logging.getLogger(__name__)


class SpeechProbabilityModel(Protocol):
    def probability(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


class SileroSpeechModel:
    """Wraps the JIT Silero model shipped inside the silero-vad wheel (no download)."""

    _lock = threading.Lock()
    _shared: Any = None

    def __init__(self) -> None:
        import torch

        with SileroSpeechModel._lock:
            if SileroSpeechModel._shared is None:
                from silero_vad import load_silero_vad

                SileroSpeechModel._shared = load_silero_vad(onnx=False)
                log.info("Silero VAD loaded")
        # The JIT module keeps recurrent state; each detector gets its own copy.
        self._model = _clone_jit(SileroSpeechModel._shared)
        self._torch = torch

    def probability(self, frame: np.ndarray) -> float:
        if len(frame) != VAD_FRAME_SAMPLES:
            raise ValueError("Silero VAD expects 512-sample frames at 16 kHz")
        with self._torch.inference_mode():
            tensor = self._torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
            return float(self._model(tensor, SAMPLE_RATE).item())

    def reset(self) -> None:
        self._model.reset_states()


def _clone_jit(model: Any) -> Any:
    import copy

    try:
        return copy.deepcopy(model)
    except Exception:
        log.warning("Silero VAD deepcopy unsupported; sharing a single instance")
        return model


class VADEvent(StrEnum):
    SILENCE = "silence"
    SPEECH_STARTED = "speech_started"
    SPEECH_CONTINUES = "speech_continues"
    SPEECH_ENDED = "speech_ended"


class VoiceActivityDetector:
    """Frame-level hysteresis on top of speech probabilities.

    Speech starts when p >= threshold; it ends after ``min_silence_ms`` of frames
    with p < neg_threshold. Frames between the thresholds keep the current state.
    """

    def __init__(
        self,
        model: SpeechProbabilityModel,
        threshold: float = 0.5,
        neg_threshold: float = 0.35,
        min_silence_ms: int = 700,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.neg_threshold = min(neg_threshold, threshold - 0.05)
        frame_ms = VAD_FRAME_SAMPLES * 1000 / SAMPLE_RATE
        self.min_silence_frames = max(1, round(min_silence_ms / frame_ms))
        self.in_speech = False
        self.silence_frames = 0
        self.last_probability = 0.0

    def process(self, frame: np.ndarray) -> VADEvent:
        prob = self.model.probability(frame)
        self.last_probability = prob
        if not self.in_speech:
            if prob >= self.threshold:
                self.in_speech = True
                self.silence_frames = 0
                return VADEvent.SPEECH_STARTED
            return VADEvent.SILENCE
        if prob < self.neg_threshold:
            self.silence_frames += 1
            if self.silence_frames >= self.min_silence_frames:
                self.in_speech = False
                self.silence_frames = 0
                return VADEvent.SPEECH_ENDED
        else:
            self.silence_frames = 0
        return VADEvent.SPEECH_CONTINUES

    def reset(self) -> None:
        self.model.reset()
        self.in_speech = False
        self.silence_frames = 0
