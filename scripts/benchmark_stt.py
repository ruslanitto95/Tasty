"""GigaAM STT benchmark: WAV -> VAD -> GigaAM; WER + clinical error rates + RTF.

uv run python scripts/benchmark_stt.py [--audio-dir tests/benchmark/audio]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mva.clinical.lexicon import load_lexicon
from mva.diagnostics.benchmark import run_benchmark
from mva.diagnostics.metrics import clinical_metrics
from mva.paths import models_dir, self_test_wav
from mva.transcription.gigaam_provider import GigaAMTranscriptionProvider
from mva.transcription.model_manager import ModelManager, load_manifest

ROOT = Path(__file__).resolve().parents[1] / "tests" / "benchmark"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "audio")
    args = parser.parse_args()
    wavs = sorted(args.audio_dir.glob("*.wav")) or [self_test_wav()]
    manager = ModelManager(load_manifest(), models_dir())
    manager.ensure_downloaded()
    provider = GigaAMTranscriptionProvider(manager)
    provider.load()
    lexicon = load_lexicon()
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "metrics").mkdir(exist_ok=True)
    summary = []
    for wav in wavs:
        result, text = run_benchmark(provider, wav)
        (ROOT / "outputs" / f"{wav.stem}.txt").write_text(text, encoding="utf-8")
        ref_path = ROOT / "references" / f"{wav.stem}.txt"
        metrics = None
        if ref_path.exists():
            metrics = asdict(clinical_metrics(ref_path.read_text(encoding="utf-8"), text, lexicon))
        row = {"file": wav.name, **asdict(result), "metrics": metrics}
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False))
    (ROOT / "metrics" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
