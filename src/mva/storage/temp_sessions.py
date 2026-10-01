"""Per-visit temporary directories: %LOCALAPPDATA%/MedicalVisitAssistant/temp/<UUID>.

Directory names are random UUIDs; no patient identifiers ever appear in paths.
Directories left behind by a crashed process are removed on the next start.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

import psutil

log = logging.getLogger(__name__)

_OWNER_FILE = ".owner_pid"


class TempSession:
    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def id(self) -> str:
        return self.path.name

    def file(self, name: str) -> Path:
        safe = Path(name).name  # strip directory components (path traversal guard)
        if safe in ("", ".", ".."):
            raise ValueError("invalid temp file name")
        return self.path / safe

    def destroy(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


class TempSessionManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self) -> TempSession:
        path = self.root / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        (path / _OWNER_FILE).write_text(str(os.getpid()), encoding="ascii")
        return TempSession(path)

    def cleanup_orphans(self) -> int:
        removed = 0
        for entry in self.root.iterdir():
            if not entry.is_dir():
                continue
            owner = self._owner(entry)
            if owner == os.getpid():
                continue
            if owner is not None and psutil.pid_exists(owner) and self._is_mva(owner):
                continue  # belongs to another running instance
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        if removed:
            log.info("Removed %d orphaned temp session(s)", removed)
        return removed

    def count(self) -> int:
        return sum(1 for entry in self.root.iterdir() if entry.is_dir())

    @staticmethod
    def _owner(entry: Path) -> int | None:
        try:
            return int((entry / _OWNER_FILE).read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _is_mva(pid: int) -> bool:
        # PIDs are reused; only trust owners that still look like this application.
        try:
            proc = psutil.Process(pid)
            text = " ".join([proc.name(), *proc.cmdline()]).lower()
        except (psutil.Error, OSError):
            return False
        return "mva" in text or "medicalvisitassistant" in text
