"""Versioned user settings. Contains no PHI and no secrets (API key is in the vault)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class Theme(StrEnum):
    SYSTEM = "system"
    LIGHT = "light"
    DARK = "dark"


class InferenceDevice(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


class GeneralSettings(BaseModel):
    always_on_top: bool = True
    start_with_windows: bool = False
    theme: Theme = Theme.SYSTEM
    language: str = "ru"
    window_geometry: str | None = None  # base64 of QWidget.saveGeometry()


class AudioSettings(BaseModel):
    # Persist by name + host API: PortAudio indices change when devices are plugged in.
    device_name: str | None = None  # None = system default microphone
    device_hostapi: str | None = None


class VADSettings(BaseModel):
    threshold: float = Field(0.5, ge=0.1, le=0.95)
    min_silence_ms: int = Field(700, ge=200, le=3000)
    pre_roll_ms: int = Field(400, ge=0, le=1000)
    post_roll_ms: int = Field(500, ge=0, le=1500)


class SpeechSettings(BaseModel):
    model_name: str = "v3_e2e_rnnt"
    device: InferenceDevice = InferenceDevice.AUTO
    vad: VADSettings = Field(default_factory=VADSettings)


class AISettings(BaseModel):
    enabled: bool = False
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    timeout_s: float = Field(120.0, ge=5, le=600)
    use_llm_formatter: bool = True


class PrivacySettings(BaseModel):
    # Raw audio is always deleted after transcription; shown read-only in the UI.
    delete_audio_after_transcription: bool = True
    save_history: bool = False


class HotkeySettings(BaseModel):
    start_stop: str = "Ctrl+Alt+R"
    copy_complaints: str = "Ctrl+Alt+1"
    copy_history: str = "Ctrl+Alt+2"
    new_visit: str = "Ctrl+Alt+N"


class DeveloperSettings(BaseModel):
    enabled: bool = False
    save_test_audio: bool = False


class AppSettings(BaseModel):
    schema_version: int = SCHEMA_VERSION
    onboarding_completed: bool = False
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    speech: SpeechSettings = Field(default_factory=SpeechSettings)
    ai: AISettings = Field(default_factory=AISettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    hotkeys: HotkeySettings = Field(default_factory=HotkeySettings)
    developer: DeveloperSettings = Field(default_factory=DeveloperSettings)


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade older settings dicts. Unknown keys are ignored by pydantic."""
    version = int(raw.get("schema_version", 0))
    if version < 1:
        raw.setdefault("general", {})
        raw["schema_version"] = 1
    return raw


class SettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("settings root is not an object")
            return AppSettings.model_validate(migrate(raw))
        except (OSError, ValueError, ValidationError) as exc:
            backup = self.path.with_suffix(".corrupt.json")
            log.warning("Settings unreadable (%s); using defaults", type(exc).__name__)
            try:
                os.replace(self.path, backup)
            except OSError:
                pass
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = settings.model_dump_json(indent=2)
        fd, tmp = tempfile.mkstemp(prefix="settings.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
