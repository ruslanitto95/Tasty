"""Thread-safe FIFO for speech segments with bounded RAM.

When STT falls behind, segments beyond ``max_in_memory`` are written to the visit's
temp directory and their samples released; nothing is dropped. Spilled files are
deleted immediately after they are read back.
"""

from __future__ import annotations

import collections
import threading

import numpy as np

from mva.audio.wav import read_audio_16k, write_wav_16k
from mva.storage.temp_sessions import TempSession
from mva.transcription.models import AudioSegment


class QueueClosed(Exception):
    pass


class SegmentQueue:
    def __init__(self, temp: TempSession | None, max_in_memory: int = 24) -> None:
        self._temp = temp
        self._max_in_memory = max_in_memory
        self._items: collections.deque[AudioSegment] = collections.deque()
        self._cond = threading.Condition()
        self._closed = False
        self.spilled_total = 0

    def put(self, segment: AudioSegment) -> None:
        with self._cond:
            if self._closed:
                raise QueueClosed()
            in_memory = sum(1 for s in self._items if s.samples is not None)
            if (
                in_memory >= self._max_in_memory
                and self._temp is not None
                and segment.samples is not None
            ):
                path = self._temp.file(f"{segment.id}.wav")
                write_wav_16k(path, segment.samples)
                segment.spill_path = path
                segment.samples = None
                self.spilled_total += 1
            self._items.append(segment)
            self._cond.notify()

    def get(self, timeout: float | None = None) -> AudioSegment | None:
        """Returns None on timeout; raises QueueClosed when closed and drained."""
        with self._cond:
            if not self._items:
                if self._closed:
                    raise QueueClosed()
                self._cond.wait(timeout)
                if not self._items:
                    if self._closed:
                        raise QueueClosed()
                    return None
            segment = self._items.popleft()
        if segment.samples is None and segment.spill_path is not None:
            try:
                segment.samples = read_audio_16k(segment.spill_path)
            finally:
                segment.spill_path.unlink(missing_ok=True)
                segment.spill_path = None
        return segment

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def clear(self) -> None:
        with self._cond:
            for seg in self._items:
                if seg.spill_path is not None:
                    seg.spill_path.unlink(missing_ok=True)
                seg.samples = np.zeros(0, dtype=np.float32)
            self._items.clear()
            self._cond.notify_all()

    def pending(self) -> int:
        with self._cond:
            return len(self._items)

    def pending_audio_s(self) -> float:
        with self._cond:
            return sum((s.end_ms - s.start_ms) / 1000 for s in self._items)
