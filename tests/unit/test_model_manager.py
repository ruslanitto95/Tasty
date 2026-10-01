from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import httpx
import pytest

from mva.transcription.model_manager import (
    ChecksumMismatch,
    DownloadCancelled,
    ModelDownloadError,
    ModelFile,
    ModelManager,
    ModelManifest,
    ModelStatus,
    NotEnoughDiskSpace,
    load_manifest,
)

CKPT = bytes(range(256)) * 4000  # ~1 MB
TOK = b"tokenizer" * 100


def manifest(ckpt: bytes = CKPT, tok: bytes = TOK) -> ModelManifest:
    return ModelManifest(
        model_name="v3_e2e_rnnt",
        display_name="test",
        gigaam_commit="0" * 40,
        revision="test",
        files=[
            ModelFile(
                name="m.ckpt",
                url="https://cdn.test/m.ckpt",
                size=len(ckpt),
                sha256=hashlib.sha256(ckpt).hexdigest(),
            ),
            ModelFile(
                name="m_tok.model",
                url="https://cdn.test/m_tok.model",
                size=len(tok),
                sha256=hashlib.sha256(tok).hexdigest(),
            ),
        ],
    )


class Server:
    def __init__(self, ckpt: bytes = CKPT, tok: bytes = TOK, honour_range: bool = True) -> None:
        self.files = {"/m.ckpt": ckpt, "/m_tok.model": tok}
        self.honour_range = honour_range
        self.requests: list[httpx.Request] = []
        self.cut_after: int | None = None  # simulate a dropped connection once
        self.fail_status: int | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_status:
            return httpx.Response(self.fail_status)
        data = self.files[request.url.path]
        rng = request.headers.get("Range")
        if rng and self.honour_range:
            start = int(rng.split("=")[1].rstrip("-"))
            body, status = data[start:], 206
        else:
            body, status = data, 200
        if self.cut_after is not None and request.url.path == "/m.ckpt":
            body = body[: self.cut_after]
            self.cut_after = None
        return httpx.Response(status, content=body)

    def factory(self):
        return lambda: httpx.Client(transport=httpx.MockTransport(self.handler))


def test_manifest_is_pinned():
    m = load_manifest()
    assert m.model_name == "v3_e2e_rnnt"
    assert len(m.gigaam_commit) == 40
    assert all(len(f.sha256) == 64 and f.size > 0 for f in m.files)
    assert m.files[0].url.startswith("https://")


def test_fresh_download_verifies_and_marks_ready(tmp_path: Path):
    server = Server()
    mm = ModelManager(manifest(), tmp_path, server.factory())
    assert mm.status() == ModelStatus.MISSING
    seen = []
    mm.ensure_downloaded(seen.append)
    assert mm.status() == ModelStatus.READY
    assert (tmp_path / "m.ckpt").read_bytes() == CKPT
    assert seen and seen[-1].downloaded == len(CKPT) + len(TOK)
    assert not list(tmp_path.glob("*.part"))
    # Second call does not touch the network (model is not re-downloaded on update/restart).
    n = len(server.requests)
    mm.ensure_downloaded()
    assert len(server.requests) == n


def test_interrupted_download_resumes_with_range(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    server = Server()
    server.cut_after = 300_000
    mm = ModelManager(manifest(), tmp_path, server.factory())
    mm.ensure_downloaded()
    assert mm.status() == ModelStatus.READY
    ranges = [r.headers.get("Range") for r in server.requests if r.url.path == "/m.ckpt"]
    assert ranges[0] is None and ranges[1] == "bytes=300000-"


def test_server_ignoring_range_restarts_cleanly(tmp_path: Path):
    server = Server(honour_range=False)
    (tmp_path / "m.ckpt.part").write_bytes(CKPT[:1000])
    mm = ModelManager(manifest(), tmp_path, server.factory())
    assert mm.status() == ModelStatus.PARTIAL
    mm.ensure_downloaded()
    assert (tmp_path / "m.ckpt").read_bytes() == CKPT


def test_checksum_mismatch_never_leaves_valid_looking_model(tmp_path: Path):
    bad = bytearray(CKPT)
    bad[10] ^= 0xFF
    server = Server(ckpt=bytes(bad))
    mm = ModelManager(manifest(), tmp_path, server.factory())
    with pytest.raises(ChecksumMismatch):
        mm.ensure_downloaded()
    assert not (tmp_path / "m.ckpt").exists()
    assert not (tmp_path / "m.ckpt.part").exists()
    assert mm.status() == ModelStatus.MISSING


def test_corrupted_file_on_disk_is_detected_and_replaced(tmp_path: Path):
    server = Server()
    mm = ModelManager(manifest(), tmp_path, server.factory())
    mm.ensure_downloaded()
    (tmp_path / "m.ckpt").write_bytes(b"broken" + CKPT[6:])
    assert mm.status() == ModelStatus.CORRUPT
    mm.ensure_downloaded()
    assert mm.status() == ModelStatus.READY
    assert (tmp_path / "m.ckpt").read_bytes() == CKPT


def test_not_enough_disk_space(tmp_path: Path, monkeypatch):
    import collections

    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr("shutil.disk_usage", lambda _p: usage(10, 10, 1000))
    mm = ModelManager(manifest(), tmp_path, Server().factory())
    with pytest.raises(NotEnoughDiskSpace):
        mm.ensure_downloaded()


def test_http_errors_exhaust_retries(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    server = Server()
    server.fail_status = 503
    mm = ModelManager(manifest(), tmp_path, server.factory(), max_attempts=3)
    with pytest.raises(ModelDownloadError):
        mm.ensure_downloaded()
    assert len(server.requests) == 3


def test_cancel(tmp_path: Path):
    cancel = threading.Event()
    cancel.set()
    mm = ModelManager(manifest(), tmp_path, Server().factory())
    with pytest.raises(DownloadCancelled):
        mm.ensure_downloaded(cancel=cancel)


def test_unwritable_directory(tmp_path: Path):
    from mva.transcription.model_manager import ModelStorageError

    target = tmp_path / "file"
    target.write_text("not a dir")
    mm = ModelManager(manifest(), target / "models", Server().factory())
    with pytest.raises(ModelStorageError):
        mm.ensure_downloaded()
