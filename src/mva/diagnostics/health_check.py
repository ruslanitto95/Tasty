"""Settings → Diagnostics checks and the exportable PHI-free diagnostic report."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

from mva import __version__
from mva.transcription.model_manager import ModelManager, ModelStatus


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_temp_storage(root: Path) -> CheckResult:
    try:
        probe = root / ".probe"
        probe.write_bytes(b"x")
        probe.unlink()
        free_gb = shutil.disk_usage(root).free / 1e9
        return CheckResult("temp_storage", True, f"{free_gb:.1f} GB free")
    except OSError as exc:
        return CheckResult("temp_storage", False, type(exc).__name__)


def check_model(manager: ModelManager, loaded: bool, device: str) -> CheckResult:
    status = manager.status()
    ok = status == ModelStatus.READY and loaded
    return CheckResult(
        "gigaam",
        ok,
        f"{manager.manifest.model_name} {status.value} loaded={loaded} device={device}",
    )


def check_microphone() -> CheckResult:
    from mva.audio.capture import list_input_devices

    devices = list_input_devices(refresh=False)
    if not devices:
        return CheckResult("microphone", False, "no input devices")
    default = next((d for d in devices if d.is_default), devices[0])
    return CheckResult(
        "microphone", True, f"{len(devices)} device(s), default via {default.hostapi}"
    )


def check_audio_format() -> CheckResult:
    try:
        import numpy as np

        from mva.audio.resample import resample

        out = resample(np.zeros(48000, dtype=np.float32), 48000)
        return CheckResult("audio_format", len(out) == 16000, "16 kHz mono float32")
    except Exception as exc:
        return CheckResult("audio_format", False, type(exc).__name__)


def dependency_versions() -> dict[str, str]:
    names = [
        "torch",
        "torchaudio",
        "gigaam",
        "silero-vad",
        "PySide6-Essentials",
        "sounddevice",
        "pydantic",
        "httpx",
        "numpy",
    ]
    out = {"python": sys.version.split()[0]}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "missing"
    return out


def diagnostic_report(
    manager: ModelManager, device: str, checks: list[CheckResult], error_codes: list[str]
) -> str:
    """Technical data only: no transcript, no medical text, no settings values like URLs/keys."""
    report = {
        "app_version": __version__,
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "machine": platform.machine(),
        "model": manager.manifest.model_name,
        "model_revision": manager.manifest.revision,
        "gigaam_commit": manager.manifest.gigaam_commit,
        "device": device,
        "dependencies": dependency_versions(),
        "checks": [asdict(c) for c in checks],
        "error_codes": error_codes[-50:],
    }
    return json.dumps(report, ensure_ascii=False, indent=2)
