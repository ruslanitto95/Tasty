"""Static application configuration (non-user-editable defaults).

User-editable values live in :mod:`mva.storage.app_settings`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

APP_NAME = "Medical Visit Assistant"
APP_SHORT_NAME = "MVA"
APP_DIR_NAME = "MedicalVisitAssistant"
APP_ORG = "MVA"

SAMPLE_RATE = 16_000
# Silero VAD v5+/v6 accepts exactly 512 samples per call at 16 kHz.
VAD_FRAME_SAMPLES = 512


@dataclass(frozen=True)
class VADConfig:
    threshold: float = 0.5
    neg_threshold: float = 0.35
    min_silence_ms: int = 700
    min_speech_ms: int = 250
    pre_roll_ms: int = 400
    post_roll_ms: int = 500


@dataclass(frozen=True)
class SegmentationConfig:
    # GigaAM .transcribe() is documented for audio up to 25 s; stay safely below.
    target_max_s: float = 16.0
    hard_max_s: float = 22.0
    # Where to look for the quietest frame when a monologue exceeds target_max_s.
    split_search_window_s: float = 5.0
    # Audio overlap added only on forced splits (no pause found); merger dedups it.
    forced_split_overlap_ms: int = 600
    min_segment_ms: int = 300


@dataclass(frozen=True)
class LevelConfig:
    low_level_dbfs: float = -45.0
    silent_dbfs: float = -70.0
    low_level_window_s: float = 8.0
    clip_threshold: float = 0.99
    # Segment normaliser: target peak and max gain.
    target_peak_dbfs: float = -3.0
    max_gain_db: float = 20.0


@dataclass(frozen=True)
class STTQueueConfig:
    # Segments kept in RAM; beyond that pending audio is spilled to the session temp dir.
    max_in_memory_segments: int = 24
    lag_warning_s: float = 45.0


@dataclass(frozen=True)
class LLMDefaults:
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 120.0
    overall_timeout_s: float = 300.0
    max_attempts: int = 3
    backoff_base_s: float = 1.5


@dataclass(frozen=True)
class StaticConfig:
    vad: VADConfig = field(default_factory=VADConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    levels: LevelConfig = field(default_factory=LevelConfig)
    stt_queue: STTQueueConfig = field(default_factory=STTQueueConfig)
    llm: LLMDefaults = field(default_factory=LLMDefaults)


DEFAULTS = StaticConfig()
