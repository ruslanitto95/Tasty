"""Signal level metering, silence detection and the per-segment normaliser."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from mva.config import DEFAULTS, LevelConfig


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-9))


@dataclass
class LevelReading:
    rms_dbfs: float
    peak: float
    clipped: bool
    low_level: bool  # signal persistently too quiet (likely wrong mic / muted / no permission)


class LevelMeter:
    def __init__(self, config: LevelConfig = DEFAULTS.levels) -> None:
        self.config = config
        self._peaks: deque[tuple[float, float]] = deque()
        self.clip_events = 0

    def update(self, block: np.ndarray, now: float | None = None) -> LevelReading:
        now = time.monotonic() if now is None else now
        if len(block) == 0:
            return LevelReading(-120.0, 0.0, False, False)
        peak = float(np.max(np.abs(block)))
        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
        clipped = peak >= self.config.clip_threshold
        if clipped:
            self.clip_events += 1
        self._peaks.append((now, peak))
        window = self.config.low_level_window_s
        while self._peaks and now - self._peaks[0][0] > window:
            self._peaks.popleft()
        span = now - self._peaks[0][0] if self._peaks else 0.0
        max_peak = max(p for _, p in self._peaks)
        low = span >= window * 0.9 and dbfs(max_peak) < self.config.low_level_dbfs
        return LevelReading(dbfs(rms), peak, clipped, low)

    def reset(self) -> None:
        self._peaks.clear()
        self.clip_events = 0


def normalize_segment(samples: np.ndarray, config: LevelConfig = DEFAULTS.levels) -> np.ndarray:
    """DC removal + bounded peak normalisation + hard clip guard. Never amplifies silence."""
    x = np.asarray(samples, dtype=np.float32)
    if len(x) == 0:
        return x
    x = x - np.float32(np.mean(x))
    peak = float(np.max(np.abs(x)))
    if peak < 10 ** (config.silent_dbfs / 20):
        return np.clip(x, -1.0, 1.0)
    target = 10 ** (config.target_peak_dbfs / 20)
    gain = min(target / peak, 10 ** (config.max_gain_db / 20))
    if gain < 1.0 or peak < target:
        x = x * np.float32(gain)
    return np.clip(x, -1.0, 1.0).astype(np.float32)
