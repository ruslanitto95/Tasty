"""Filesystem locations. Nothing here may contain patient identifiers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mva.config import APP_DIR_NAME


def app_data_dir() -> Path:
    override = os.environ.get("MVA_HOME")
    if override:
        base = Path(override)
    elif sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        base = Path(local) / APP_DIR_NAME
    else:
        xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        base = Path(xdg) / APP_DIR_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def models_dir() -> Path:
    override = os.environ.get("MVA_MODELS_DIR")
    path = Path(override) if override else app_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def temp_root() -> Path:
    path = app_data_dir() / "temp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = app_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_file() -> Path:
    return app_data_dir() / "settings.json"


def benchmarks_file() -> Path:
    return app_data_dir() / "benchmarks.jsonl"


def dev_recordings_dir() -> Path:
    path = app_data_dir() / "dev_recordings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resources_dir() -> Path:
    # PyInstaller unpacks data under _MEIPASS.
    frozen_base = getattr(sys, "_MEIPASS", None)
    if frozen_base:
        return Path(frozen_base) / "mva" / "resources"
    return Path(__file__).resolve().parent / "resources"


def self_test_wav() -> Path:
    return resources_dir() / "test_audio" / "self_test_ru.wav"
