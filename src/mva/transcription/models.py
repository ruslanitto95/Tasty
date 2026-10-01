"""Transcript data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from mva.config import SAMPLE_RATE


@dataclass
class AudioSegment:
    """A speech segment cut by the VAD, 16 kHz mono float32 in [-1, 1].

    ``samples`` may be None while the segment is spilled to ``spill_path`` on disk.
    """

    seq: int
    start_ms: int
    end_ms: int
    samples: np.ndarray | None
    overlap_prev_ms: int = 0
    forced_split: bool = False
    spill_path: Path | None = field(default=None, repr=False)

    @property
    def id(self) -> str:
        return segment_id(self.seq)

    @property
    def duration_s(self) -> float:
        if self.samples is not None:
            return len(self.samples) / SAMPLE_RATE
        return (self.end_ms - self.start_ms) / 1000.0


def segment_id(seq: int) -> str:
    return f"s{seq:04d}"


class TranscriptSegment(BaseModel):
    id: str
    seq: int
    start_ms: int
    end_ms: int
    text: str
    speaker: str = "unknown"
    provider: str = ""
    inference_ms: int = 0
    overlap_prev_ms: int = 0
    failed: bool = False


class Transcript(BaseModel):
    segments: list[TranscriptSegment] = Field(default_factory=list)

    def by_id(self) -> dict[str, TranscriptSegment]:
        return {s.id: s for s in self.segments}

    def text(self) -> str:
        return " ".join(s.text for s in self.segments if s.text)

    def is_empty(self) -> bool:
        return not any(s.text.strip() for s in self.segments)


def format_ts(ms: int) -> str:
    total = ms // 1000
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
