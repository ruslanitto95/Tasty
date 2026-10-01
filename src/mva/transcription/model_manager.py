"""Downloads, verifies and locates the GigaAM checkpoint without user involvement.

* Files are pinned by size + SHA-256 in ``resources/model_manifest.json``.
* Download goes to ``<name>.part`` and resumes with HTTP Range after interruption.
* Only a fully verified file is atomically renamed to its final name, so a
  corrupted/partial download can never be mistaken for a valid model.
* Models live in %LOCALAPPDATA%/MedicalVisitAssistant/models, independent of the
  application install directory, so app updates never re-download the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx
from pydantic import BaseModel

from mva.paths import resources_dir

log = logging.getLogger(__name__)

_CHUNK = 1 << 20
_DISK_MARGIN = 300 * 1024 * 1024
_VERIFIED_SUFFIX = ".verified.json"


class ModelFile(BaseModel):
    name: str
    url: str
    size: int
    sha256: str


class ModelManifest(BaseModel):
    model_name: str
    display_name: str
    gigaam_commit: str
    revision: str
    files: list[ModelFile]

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)


def load_manifest(model_name: str = "v3_e2e_rnnt") -> ModelManifest:
    data = json.loads((resources_dir() / "model_manifest.json").read_text(encoding="utf-8"))
    for entry in data["models"]:
        if entry["model_name"] == model_name:
            return ModelManifest.model_validate(entry)
    raise KeyError(f"model {model_name!r} is not in the manifest")


class ModelStatus(StrEnum):
    MISSING = "missing"
    PARTIAL = "partial"
    READY = "ready"
    CORRUPT = "corrupt"


class ModelError(Exception):
    code = "model_error"


class ModelDownloadError(ModelError):
    code = "model_download_failed"


class NotEnoughDiskSpace(ModelError):
    code = "not_enough_disk"

    def __init__(self, required: int, free: int) -> None:
        super().__init__(f"required={required} free={free}")
        self.required = required
        self.free = free


class ChecksumMismatch(ModelError):
    code = "model_checksum"


class ModelStorageError(ModelError):
    code = "model_storage"


class DownloadCancelled(ModelError):
    code = "cancelled"


@dataclass
class DownloadProgress:
    downloaded: int
    total: int
    file_name: str
    bytes_per_s: float

    @property
    def fraction(self) -> float:
        return self.downloaded / self.total if self.total else 0.0


ProgressCallback = Callable[[DownloadProgress], None]
ClientFactory = Callable[[], httpx.Client]


def _default_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=15.0),
        follow_redirects=True,
        headers={"User-Agent": "MedicalVisitAssistant-ModelManager"},
    )


def sha256_file(path: Path, cancel: threading.Event | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            if cancel is not None and cancel.is_set():
                raise DownloadCancelled()
            digest.update(chunk)
    return digest.hexdigest()


class ModelManager:
    def __init__(
        self,
        manifest: ModelManifest,
        directory: Path,
        client_factory: ClientFactory = _default_client,
        max_attempts: int = 4,
    ) -> None:
        self.manifest = manifest
        self.directory = directory
        self._client_factory = client_factory
        self._max_attempts = max_attempts

    # ---- paths ---------------------------------------------------------------
    def path_for(self, file: ModelFile) -> Path:
        return self.directory / Path(file.name).name

    @property
    def checkpoint_path(self) -> Path:
        return self.path_for(self.manifest.files[0])

    @property
    def tokenizer_path(self) -> Path | None:
        return self.path_for(self.manifest.files[1]) if len(self.manifest.files) > 1 else None

    def _marker(self, file: ModelFile) -> Path:
        return self.path_for(file).with_name(Path(file.name).name + _VERIFIED_SUFFIX)

    # ---- status --------------------------------------------------------------
    def status(self) -> ModelStatus:
        any_partial = False
        for file in self.manifest.files:
            final = self.path_for(file)
            if final.exists():
                if not self._quick_verified(file):
                    return ModelStatus.CORRUPT
                continue
            if final.with_name(final.name + ".part").exists():
                any_partial = True
            return ModelStatus.PARTIAL if any_partial else ModelStatus.MISSING
        return ModelStatus.READY

    def _quick_verified(self, file: ModelFile) -> bool:
        """Size + marker written after a full SHA-256 check; avoids re-hashing on each launch."""
        final = self.path_for(file)
        try:
            stat = final.stat()
            marker = json.loads(self._marker(file).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            stat.st_size == file.size
            and marker.get("sha256") == file.sha256
            and marker.get("size") == stat.st_size
            and int(marker.get("mtime_ns", -1)) == stat.st_mtime_ns
        )

    def verify_full(self, cancel: threading.Event | None = None) -> bool:
        ok = True
        for file in self.manifest.files:
            final = self.path_for(file)
            if not final.exists() or final.stat().st_size != file.size:
                ok = False
                continue
            if sha256_file(final, cancel) != file.sha256:
                log.error("Checksum mismatch for %s; deleting", file.name)
                self._remove(file)
                ok = False
            else:
                self._write_marker(file)
        return ok

    def remaining_bytes(self) -> int:
        remaining = 0
        for file in self.manifest.files:
            final = self.path_for(file)
            if final.exists() and self._quick_verified(file):
                continue
            part = final.with_name(final.name + ".part")
            have = part.stat().st_size if part.exists() else 0
            remaining += max(0, file.size - have)
        return remaining

    def check_disk_space(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.directory).free
        required = self.remaining_bytes() + _DISK_MARGIN
        if free < required:
            raise NotEnoughDiskSpace(required, free)

    # ---- download ------------------------------------------------------------
    def ensure_downloaded(
        self,
        progress: ProgressCallback | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        cancel = cancel or threading.Event()
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            probe = self.directory / ".write_probe"
            probe.write_bytes(b"ok")
            probe.unlink()
        except OSError as exc:
            raise ModelStorageError(f"models directory not writable: {type(exc).__name__}") from exc

        if self.status() == ModelStatus.CORRUPT:
            for file in self.manifest.files:
                if self.path_for(file).exists() and not self._quick_verified(file):
                    # Could be a valid file without marker (e.g. copied in); verify before deleting.
                    self.verify_full(cancel)
                    break
        if self.status() == ModelStatus.READY:
            return
        self.check_disk_space()

        total = self.manifest.total_size
        done_before = 0
        for file in self.manifest.files:
            if self.path_for(file).exists() and self._quick_verified(file):
                done_before += file.size
                continue
            self._download_file(file, total, done_before, progress, cancel)
            done_before += file.size
        if self.status() != ModelStatus.READY:
            raise ModelDownloadError("model not ready after download")

    def _download_file(
        self,
        file: ModelFile,
        total: int,
        done_before: int,
        progress: ProgressCallback | None,
        cancel: threading.Event,
    ) -> None:
        final = self.path_for(file)
        part = final.with_name(final.name + ".part")
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            if cancel.is_set():
                raise DownloadCancelled()
            try:
                self._stream_to_part(file, part, total, done_before, progress, cancel)
                break
            except DownloadCancelled:
                raise
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
                log.warning(
                    "Model download attempt %d/%d failed: %s",
                    attempt,
                    self._max_attempts,
                    type(exc).__name__,
                )
                if isinstance(exc, OSError) and not isinstance(exc, httpx.HTTPError):
                    if getattr(exc, "errno", None) == 28:  # ENOSPC
                        raise NotEnoughDiskSpace(file.size, 0) from exc
                time.sleep(min(2.0 * attempt, 8.0))
        else:
            raise ModelDownloadError(
                f"download failed: {type(last_error).__name__}"
            ) from last_error

        size = part.stat().st_size
        if size != file.size:
            part.unlink(missing_ok=True)
            raise ChecksumMismatch(f"size mismatch for {file.name}: {size} != {file.size}")
        digest = sha256_file(part, cancel)
        if digest != file.sha256:
            part.unlink(missing_ok=True)
            raise ChecksumMismatch(f"sha256 mismatch for {file.name}")
        os.replace(part, final)
        self._write_marker(file)
        log.info("Model file verified: %s (%d bytes)", file.name, file.size)

    def _stream_to_part(
        self,
        file: ModelFile,
        part: Path,
        total: int,
        done_before: int,
        progress: ProgressCallback | None,
        cancel: threading.Event,
    ) -> None:
        have = part.stat().st_size if part.exists() else 0
        if have > file.size:
            part.unlink()
            have = 0
        if have == file.size:
            return
        headers = {"Range": f"bytes={have}-"} if have else {}
        started = time.monotonic()
        received = 0
        with (
            self._client_factory() as client,
            client.stream("GET", file.url, headers=headers) as resp,
        ):
            if resp.status_code == 416:
                part.unlink(missing_ok=True)
                raise httpx.HTTPError("range not satisfiable; restarting")
            if have and resp.status_code == 200:
                have = 0  # server ignored Range: restart from scratch
            elif resp.status_code not in (200, 206):
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
            mode = "ab" if have else "wb"
            with part.open(mode) as out:
                for chunk in resp.iter_bytes(_CHUNK):
                    if cancel.is_set():
                        raise DownloadCancelled()
                    out.write(chunk)
                    received += len(chunk)
                    if have + received > file.size:
                        raise httpx.HTTPError("server sent more data than expected")
                    if progress is not None:
                        elapsed = max(time.monotonic() - started, 1e-3)
                        progress(
                            DownloadProgress(
                                downloaded=done_before + have + received,
                                total=total,
                                file_name=file.name,
                                bytes_per_s=received / elapsed,
                            )
                        )
        if have + received != file.size:
            raise httpx.HTTPError("connection closed before download completed")

    def _write_marker(self, file: ModelFile) -> None:
        stat = self.path_for(file).stat()
        marker = {"sha256": file.sha256, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        self._marker(file).write_text(json.dumps(marker), encoding="utf-8")

    def _remove(self, file: ModelFile) -> None:
        self.path_for(file).unlink(missing_ok=True)
        self._marker(file).unlink(missing_ok=True)
