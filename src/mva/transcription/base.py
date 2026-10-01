"""Provider-agnostic STT interface. Nothing outside transcription/ imports GigaAM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from mva.transcription.models import AudioSegment, TranscriptSegment


@dataclass
class ProviderInfo:
    name: str
    model: str
    revision: str
    device: str


LoadProgress = Callable[[str], None]


class TranscriptionError(Exception):
    code = "stt_failed"


class ModelLoadError(TranscriptionError):
    code = "gigaam_load_failed"


class TranscriptionProvider(ABC):
    @abstractmethod
    def load(self, progress: LoadProgress | None = None) -> None: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def info(self) -> ProviderInfo: ...

    @property
    def max_segment_s(self) -> float:
        return 25.0

    @abstractmethod
    def transcribe(self, segment: AudioSegment) -> TranscriptSegment: ...

    def unload(self) -> None:
        return None
