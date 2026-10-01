"""Developer benchmark: technical timing only (no medical text is stored)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

from mva.transcription.base import TranscriptionProvider
from mva.transcription.session import transcribe_file


@dataclass
class BenchmarkResult:
    timestamp: float
    audio_seconds: float
    transcription_seconds: float
    rtf: float
    rss_mb: float
    device: str
    model: str
    segments: int


def run_benchmark(provider: TranscriptionProvider, wav: Path) -> tuple[BenchmarkResult, str]:
    started = time.monotonic()
    transcript, stats = transcribe_file(provider, wav)
    wall = time.monotonic() - started
    info = provider.info()
    result = BenchmarkResult(
        timestamp=time.time(),
        audio_seconds=round(stats.audio_seconds, 2),
        transcription_seconds=round(wall, 2),
        rtf=round(stats.rtf, 3),
        rss_mb=round(psutil.Process().memory_info().rss / 1e6, 1),
        device=info.device,
        model=info.model,
        segments=len(transcript.segments),
    )
    return result, transcript.text()


def append_benchmark(path: Path, result: BenchmarkResult) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result)) + "\n")
