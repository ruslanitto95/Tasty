"""Cuts a continuous 16 kHz stream into speech segments suitable for GigaAM.

* pre-roll keeps the beginning of the first word, post-roll its end;
* segments end on VAD silence;
* monologues longer than ``target_max_s`` are split at the quietest frame of the
  last ``split_search_window_s``; only if no pause exists before ``hard_max_s`` a
  forced split with a short audio overlap is made (the merger removes duplicates).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mva.audio.buffers import RingBuffer
from mva.audio.vad import VADEvent, VoiceActivityDetector
from mva.config import DEFAULTS, SAMPLE_RATE, VAD_FRAME_SAMPLES, SegmentationConfig
from mva.transcription.models import AudioSegment

SegmentSink = Callable[[AudioSegment], None]


def _ms(samples: int) -> int:
    return int(samples * 1000 // SAMPLE_RATE)


class SpeechSegmenter:
    def __init__(
        self,
        vad: VoiceActivityDetector,
        sink: SegmentSink,
        pre_roll_ms: int = 400,
        post_roll_ms: int = 500,
        min_speech_ms: int = 250,
        config: SegmentationConfig = DEFAULTS.segmentation,
        max_segment_s: float | None = None,
    ) -> None:
        self.vad = vad
        self.sink = sink
        self.config = config
        hard_max = config.hard_max_s
        if max_segment_s is not None:
            hard_max = min(hard_max, max_segment_s - 1.0)
        self._hard_max = int(hard_max * SAMPLE_RATE)
        self._target_max = int(min(config.target_max_s, hard_max) * SAMPLE_RATE)
        self._search_frames = max(
            1, int(config.split_search_window_s * SAMPLE_RATE) // VAD_FRAME_SAMPLES
        )
        self._overlap = int(config.forced_split_overlap_ms * SAMPLE_RATE / 1000)
        self._post_roll = int(post_roll_ms * SAMPLE_RATE / 1000)
        self._min_speech = int(min_speech_ms * SAMPLE_RATE / 1000)
        self._min_segment = int(config.min_segment_ms * SAMPLE_RATE / 1000)
        self._pre_roll = RingBuffer(int(pre_roll_ms * SAMPLE_RATE / 1000))
        self._pending = np.zeros(0, dtype=np.float32)
        # Current segment = _head (pre-roll or overlap carry-over) + _frames; _probs is 1:1 with _frames.
        self._active = False
        self._head = np.zeros(0, dtype=np.float32)
        self._frames: list[np.ndarray] = []
        self._probs: list[float] = []
        self._seg_start = 0
        self._speech_samples = 0
        self._trailing_silence = 0
        self._position = 0
        self._seq = 0
        self._overlap_next = 0
        self._forced_next = False
        self.segments_emitted = 0

    @property
    def position_ms(self) -> int:
        return _ms(self._position)

    @property
    def in_speech(self) -> bool:
        return self._active

    def _length(self) -> int:
        return len(self._head) + len(self._frames) * VAD_FRAME_SAMPLES

    def feed(self, samples: np.ndarray) -> None:
        data = np.concatenate([self._pending, np.asarray(samples, dtype=np.float32)])
        n_frames = len(data) // VAD_FRAME_SAMPLES
        for i in range(n_frames):
            self._process_frame(data[i * VAD_FRAME_SAMPLES : (i + 1) * VAD_FRAME_SAMPLES])
        self._pending = data[n_frames * VAD_FRAME_SAMPLES :].copy()

    def _process_frame(self, frame: np.ndarray) -> None:
        event = self.vad.process(frame)
        prob = self.vad.last_probability
        frame_start = self._position
        self._position += len(frame)
        if event == VADEvent.SILENCE:
            self._pre_roll.write(frame)
            return
        if event == VADEvent.SPEECH_STARTED:
            self._head = self._pre_roll.read_all()
            self._pre_roll.clear()
            self._frames = []
            self._probs = []
            self._seg_start = frame_start - len(self._head)
            self._speech_samples = 0
            self._trailing_silence = 0
            self._active = True
        self._frames.append(frame.copy())
        self._probs.append(prob)
        if prob >= self.vad.neg_threshold:
            self._speech_samples += len(frame)
            self._trailing_silence = 0
        else:
            self._trailing_silence += len(frame)
        if event == VADEvent.SPEECH_ENDED:
            self._finish(trim_silence=True)
            return
        if self._length() >= self._target_max:
            self._maybe_split()

    def _maybe_split(self) -> None:
        start = max(0, len(self._probs) - self._search_frames)
        window = self._probs[start:]
        quiet = start + int(np.argmin(window))
        forced = self._probs[quiet] >= self.vad.neg_threshold
        if forced and self._length() < self._hard_max:
            return  # no pause yet: keep waiting until the hard limit
        audio = np.concatenate([self._head, *self._frames])
        frame_offset = len(self._head) + quiet * VAD_FRAME_SAMPLES
        split_at = frame_offset + VAD_FRAME_SAMPLES // 2
        overlap = self._overlap if forced else 0
        carry_from = max(0, split_at - overlap)
        self._emit(audio[:split_at], self._seg_start)
        self._overlap_next = _ms(split_at - carry_from)
        self._forced_next = forced
        self._head = audio[carry_from : frame_offset + VAD_FRAME_SAMPLES].copy()
        self._frames = self._frames[quiet + 1 :]
        self._probs = self._probs[quiet + 1 :]
        self._seg_start += carry_from
        self._speech_samples = self._length()

    def _finish(self, trim_silence: bool) -> None:
        if not self._active:
            return
        audio = np.concatenate([self._head, *self._frames])
        if trim_silence and self._trailing_silence > self._post_roll:
            audio = audio[: len(audio) - (self._trailing_silence - self._post_roll)]
        if self._speech_samples >= self._min_speech and len(audio) >= self._min_segment:
            self._emit(audio, self._seg_start)
        self._active = False
        self._head = np.zeros(0, dtype=np.float32)
        self._frames = []
        self._probs = []
        self._speech_samples = 0
        self._trailing_silence = 0
        self._overlap_next = 0
        self._forced_next = False

    def _emit(self, audio: np.ndarray, start_sample: int) -> None:
        self._seq += 1
        begin = max(0, start_sample)
        segment = AudioSegment(
            seq=self._seq,
            start_ms=_ms(begin),
            end_ms=_ms(begin + len(audio)),
            samples=audio.astype(np.float32, copy=True),
            overlap_prev_ms=self._overlap_next,
            forced_split=self._forced_next,
        )
        self._overlap_next = 0
        self._forced_next = False
        self.segments_emitted += 1
        self.sink(segment)

    def flush(self) -> None:
        """End of stream: emit any speech in progress."""
        if len(self._pending):
            pad = np.zeros(VAD_FRAME_SAMPLES - len(self._pending), dtype=np.float32)
            self._process_frame(np.concatenate([self._pending, pad]))
            self._pending = np.zeros(0, dtype=np.float32)
        self._finish(trim_silence=True)
        self.vad.reset()
        self._pre_roll.clear()
