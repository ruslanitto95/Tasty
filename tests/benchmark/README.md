# STT benchmark

- `audio/` — WAV files (16 kHz preferred). **Do not commit patient recordings**; the folder is git-ignored.
- `references/` — `<name>.txt` reference transcripts.
- `outputs/` — hypotheses written by the benchmark (git-ignored).
- `metrics/` — JSON metrics (git-ignored).

Run: `uv run python scripts/benchmark_stt.py` (uses the bundled synthetic WAV when `audio/` is empty).
