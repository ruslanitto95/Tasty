"""Application entry point.

GUI by default. Headless diagnostics (used by CI and support):
    MedicalVisitAssistant.exe --self-test [--mic-seconds 6] [--report report.json]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from mva import __version__
from mva.config import APP_NAME

log = logging.getLogger("mva")


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mva", description=APP_NAME)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--self-test", action="store_true", help="download/load GigaAM, transcribe test audio"
    )
    parser.add_argument(
        "--mic-seconds", type=float, default=0.0, help="also record N s from default mic"
    )
    parser.add_argument("--include-mic-text", action="store_true")
    parser.add_argument(
        "--mic-device", help="self-test: input device name (substring); default = system"
    )
    parser.add_argument("--report", type=Path, help="write the self-test JSON report here")
    parser.add_argument("--transcribe", type=Path, help="developer: transcribe an audio file")
    parser.add_argument(
        "--smoke-seconds", type=float, default=0.0, help="GUI: quit after N seconds"
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def quiet_native_warnings() -> None:
    # NNPACK prints an unsupported-hardware warning per conv call on some CPUs.
    try:
        import torch

        torch.backends.nnpack.set_flags(False)
    except Exception:
        log.debug("NNPACK flag not available", exc_info=True)


def _headless(args: argparse.Namespace) -> int:
    from mva.diagnostics.self_test import run_self_test, write_report
    from mva.paths import models_dir, settings_file
    from mva.storage.app_settings import SettingsStore
    from mva.transcription.gigaam_provider import GigaAMTranscriptionProvider
    from mva.transcription.model_manager import ModelManager, load_manifest

    settings = SettingsStore(settings_file()).load()
    manager = ModelManager(load_manifest(settings.speech.model_name), models_dir())
    provider = GigaAMTranscriptionProvider(manager, settings.speech.device)
    quiet_native_warnings()
    if args.transcribe:
        from mva.transcription.session import transcribe_file

        manager.ensure_downloaded()
        provider.load()
        transcript, stats = transcribe_file(provider, args.transcribe)
        for seg in transcript.segments:
            print(f"[{seg.start_ms / 1000:7.2f}-{seg.end_ms / 1000:7.2f}] {seg.text}")
        print(
            f"# segments={len(transcript.segments)} audio={stats.audio_seconds:.1f}s rtf={stats.rtf:.3f}"
        )
        return 0
    report = run_self_test(
        manager,
        provider,
        mic_seconds=args.mic_seconds,
        mic_device=_pick_device(args.mic_device),
        include_mic_text=args.include_mic_text,
        report=lambda msg: print(msg, flush=True),
    )
    if args.report:
        write_report(args.report, report)
    print(report.to_json(), flush=True)
    return 0 if report.ok else 1


def _ensure_std_streams() -> None:
    # Windowed (no console) frozen builds have sys.stdout/stderr = None; libraries still print.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115


def _pick_device(name: str | None):  # type: ignore[no-untyped-def]
    if not name:
        return None
    from mva.audio.capture import list_input_devices

    devices = list_input_devices(refresh=True)
    return next((d for d in devices if name.lower() in d.name.lower()), None)


def main(argv: list[str] | None = None) -> int:
    _ensure_std_streams()
    args = _parse(sys.argv[1:] if argv is None else argv)
    from mva.paths import logs_dir
    from mva.security.privacy import configure_logging

    configure_logging(logs_dir(), logging.DEBUG if args.debug else logging.INFO)
    log.info("Starting %s %s", APP_NAME, __version__)
    if args.self_test or args.transcribe:
        return _headless(args)
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    from mva.ui.application import run_gui

    return run_gui(smoke_seconds=args.smoke_seconds)
