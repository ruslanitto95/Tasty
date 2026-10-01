"""WAV/FLAC/OGG file input (developer file mode, benchmarks, self-test)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from mva.audio.resample import resample
from mva.config import SAMPLE_RATE


class AudioFileError(Exception):
    code = "audio_file"


def read_audio_16k(path: Path) -> np.ndarray:
    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=False)
    except (RuntimeError, OSError, sf.LibsndfileError) as exc:
        raise AudioFileError(type(exc).__name__) from exc
    return resample(np.asarray(data, dtype=np.float32), int(rate), SAMPLE_RATE)


def write_wav_16k(path: Path, samples: np.ndarray) -> None:
    sf.write(str(path), np.asarray(samples, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")
