"""Streaming resampler to the internal 16 kHz mono format."""

from __future__ import annotations

import numpy as np
import soxr

from mva.config import SAMPLE_RATE


def to_mono(block: np.ndarray) -> np.ndarray:
    block = np.asarray(block, dtype=np.float32)
    if block.ndim == 2:
        return block.mean(axis=1) if block.shape[1] > 1 else block[:, 0]
    return block


class StreamResampler:
    def __init__(self, in_rate: int, out_rate: int = SAMPLE_RATE) -> None:
        self.in_rate = int(in_rate)
        self.out_rate = int(out_rate)
        self._stream = (
            None
            if self.in_rate == self.out_rate
            else soxr.ResampleStream(self.in_rate, self.out_rate, 1, dtype="float32", quality="HQ")
        )

    def process(self, block: np.ndarray, last: bool = False) -> np.ndarray:
        mono = to_mono(block)
        if self._stream is None:
            return mono.astype(np.float32, copy=False)
        return np.asarray(self._stream.resample_chunk(mono, last=last), dtype=np.float32)


def resample(samples: np.ndarray, in_rate: int, out_rate: int = SAMPLE_RATE) -> np.ndarray:
    mono = to_mono(samples)
    if in_rate == out_rate:
        return mono.astype(np.float32, copy=False)
    return np.asarray(soxr.resample(mono, in_rate, out_rate, quality="HQ"), dtype=np.float32)
