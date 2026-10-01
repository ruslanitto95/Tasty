"""Fixed-capacity float32 ring buffer used for VAD pre-roll."""

from __future__ import annotations

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, capacity)
        self._data = np.zeros(self.capacity, dtype=np.float32)
        self._size = 0
        self._pos = 0

    def __len__(self) -> int:
        return self._size

    def write(self, samples: np.ndarray) -> None:
        if self.capacity == 0:
            return
        samples = np.asarray(samples, dtype=np.float32)
        if len(samples) >= self.capacity:
            self._data[:] = samples[-self.capacity :]
            self._pos = 0
            self._size = self.capacity
            return
        end = self._pos + len(samples)
        if end <= self.capacity:
            self._data[self._pos : end] = samples
        else:
            first = self.capacity - self._pos
            self._data[self._pos :] = samples[:first]
            self._data[: end - self.capacity] = samples[first:]
        self._pos = end % self.capacity
        self._size = min(self.capacity, self._size + len(samples))

    def read_all(self) -> np.ndarray:
        if self._size < self.capacity:
            return self._data[self._pos - self._size : self._pos].copy()
        return np.concatenate([self._data[self._pos :], self._data[: self._pos]])

    def clear(self) -> None:
        self._size = 0
        self._pos = 0
        self._data[:] = 0.0
