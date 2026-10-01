"""End-to-end self-test of the speech stack, runnable from the GUI and headless.

1. ModelManager downloads + verifies GigaAM if missing (first run).
2. GigaAM is loaded locally.
3. The bundled test WAV goes through the real pipeline (VAD -> GigaAM -> merger).
4. Optionally the microphone is recorded and transcribed through the same pipeline.

The report contains only technical data plus the text of the bundled synthetic test
audio; microphone text is included only when explicitly requested.
"""

from __future__ import annotations

import json
import logging
import platform
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mva import __version__
from mva.paths import self_test_wav
from mva.transcription.base import TranscriptionProvider
from mva.transcription.model_manager import DownloadProgress, ModelManager
from mva.transcription.session import SessionTranscriber, transcribe_file

log = logging.getLogger(__name__)

# Words expected in the transcript of resources/test_audio/self_test_ru.wav.
EXPECTED_KEYWORDS = ("беспокоит", "нос", "справа", "неделю", "температура")
MIN_KEYWORDS = 4


@dataclass
class StepResult:
    name: str
    ok: bool
    seconds: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfTestReport:
    app_version: str
    os: str
    ok: bool = False
    steps: list[StepResult] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def step(self, name: str) -> StepResult | None:
        return next((s for s in self.steps if s.name == name), None)


Reporter = Callable[[str], None]


def keyword_hits(text: str) -> list[str]:
    low = text.lower().replace("ё", "е")
    return [k for k in EXPECTED_KEYWORDS if k in low]


def run_self_test(
    manager: ModelManager,
    provider: TranscriptionProvider,
    mic_seconds: float = 0.0,
    mic_device: Any = None,
    include_mic_text: bool = False,
    report: Reporter | None = None,
    cancel: threading.Event | None = None,
    progress: Callable[[DownloadProgress], None] | None = None,
) -> SelfTestReport:
    say = report or (lambda _msg: None)
    result = SelfTestReport(app_version=__version__, os=f"{platform.system()} {platform.release()}")

    def step(name: str, fn: Callable[[], dict[str, Any]]) -> bool:
        started = time.monotonic()
        try:
            detail = fn()
            ok = bool(detail.pop("_ok", True))
        except Exception as exc:
            log.exception("Self-test step %s failed", name)
            detail = {"error": type(exc).__name__, "code": getattr(exc, "code", "error")}
            ok = False
        result.steps.append(StepResult(name, ok, round(time.monotonic() - started, 2), detail))
        say(f"{name}: {'OK' if ok else 'FAILED'} {detail}")
        return ok

    last_pct = [-1]

    def on_progress(p: DownloadProgress) -> None:
        if progress is not None:
            progress(p)
        pct = int(p.fraction * 100)
        if pct != last_pct[0] and pct % 10 == 0:
            last_pct[0] = pct
            say(f"download {pct}% ({p.downloaded // 1_000_000} MB, {p.bytes_per_s / 1e6:.1f} MB/s)")

    def do_download() -> dict[str, Any]:
        before = manager.status().value
        manager.ensure_downloaded(on_progress, cancel)
        return {
            "status_before": before,
            "status_after": manager.status().value,
            "model": manager.manifest.model_name,
            "revision": manager.manifest.revision,
            "dir": str(manager.directory),
        }

    def do_load() -> dict[str, Any]:
        provider.load()
        info = provider.info()
        return {"model": info.model, "revision": info.revision, "device": info.device}

    def do_wav() -> dict[str, Any]:
        transcript, stats = transcribe_file(provider, self_test_wav())
        text = transcript.text()
        hits = keyword_hits(text)
        return {
            "_ok": len(hits) >= MIN_KEYWORDS and stats.segments_failed == 0,
            "segments": len(transcript.segments),
            "text": text,
            "keywords_found": hits,
            "audio_s": round(stats.audio_seconds, 2),
            "rtf": round(stats.rtf, 3),
        }

    def do_mic() -> dict[str, Any]:
        from mva.audio.capture import AudioCaptureService, MicrophoneError, resolve_device

        device = mic_device or resolve_device(None, None)
        if device is None:
            return {"_ok": False, "error": "no_input_device"}
        session = SessionTranscriber(provider, None)
        session.start()
        errors: list[MicrophoneError] = []
        levels: list[float] = []
        capture = AudioCaptureService()
        capture.start(device, session.feed, lambda r: levels.append(r.peak), errors.append)
        deadline = time.monotonic() + mic_seconds
        while time.monotonic() < deadline and capture.running:
            time.sleep(0.05)
        capture.stop()
        transcript = session.finish()
        text = transcript.text()
        detail: dict[str, Any] = {
            "_ok": not errors and bool(text.strip()),
            "device_name": device.name,
            "device_hostapi": device.hostapi,
            "device_rate": device.default_samplerate,
            "peak": round(float(np.max(levels)) if levels else 0.0, 3),
            "segments": len(transcript.segments),
            "chars": len(text),
            "errors": [e.code for e in errors],
        }
        if include_mic_text:
            detail["text"] = text
            detail["keywords_found"] = keyword_hits(text)
        return detail

    ok = step("model_download", do_download) and step("model_load", do_load)
    ok = ok and step("wav_transcription", do_wav)
    if ok and mic_seconds > 0:
        ok = step("microphone_transcription", do_mic)
    result.ok = ok
    return result


def write_report(path: Path, report: SelfTestReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
