# Testing

| Suite | Command | Notes |
|---|---|---|
| Unit | `uv run pytest tests/unit` | model manager (resume/checksum/disk/cancel), segmenter, VAD hysteresis, resampling, merger, queue spill, worker, settings, privacy, credentials, LLM client, extraction repair, formatter fallback |
| Integration | `uv run pytest tests/integration` | controller with real Silero VAD, streamed WAV and fake STT: full visit, double start, offline LLM + retry, no AI, mic lost, cancel, **50 consecutive visits** (threads, temp files, RSS) |
| Clinical | `uv run pytest tests/clinical` | golden cases: transcript → scripted LLM facts (incl. adversarial ones) → exact complaints/history, accepted/rejected facts, warnings, fact precision |
| Audio (real model) | `uv run pytest -m model tests/audio` | WAV → VAD → GigaAM → transcript → clinical pipeline; 2-minute visit; pause-less monologue split; corrupt checkpoint; microphone capture via PortAudio (virtual loopback in CI) |
| E2E GUI | `uv run pytest -m model tests/e2e` | Start → speak → Stop → GigaAM → draft → edit → copy; screenshots with `MVA_SCREENSHOT_DIR` |
| Self-test | `mva --self-test [--mic-seconds N] [--mic-device NAME]` | the shipped executable downloads, loads and runs GigaAM on the bundled WAV and the microphone |
| GUI first run | `mva --gui-first-run-check report.json [--mic-seconds N] [--mic-device NAME]` | drives the real first-run wizard unattended on the current profile: the GUI downloads + verifies + loads GigaAM, transcribes the bundled WAV, records the microphone and transcribes it; exit code 0 only if all passed |

Real-model tests download GigaAM once through the application's own `ModelManager` into `MVA_MODELS_DIR` (default: system temp `mva-test-models`).

## STT benchmark
`tests/benchmark/{audio,references,outputs,metrics}`; run `uv run python scripts/benchmark_stt.py`. Metrics: WER, medical-term, medication, negation, number and laterality error rates (`mva.diagnostics.metrics`). Medical recordings must not be committed without consent and rights; only the synthetic `self_test_ru.wav` is bundled.

## Test audio
`src/mva/resources/test_audio/self_test_ru.wav` is synthetic: generated with Piper TTS voices `ru_RU-dmitri-medium` and `ru_RU-denis-medium` (training datasets CC0). No real patient speech is included.

## Clean-machine test
`.github/workflows/windows.yml` builds the installer and installs it silently. It then runs the installed app twice, each time with an empty `LOCALAPPDATA`: once headless (`--self-test`) and once through the GUI first-run wizard (`--gui-first-run-check`, PATH without Python/uv/Git). Each run must download the model itself, load it, transcribe the WAV, and transcribe speech captured through WASAPI from a VB-CABLE virtual microphone. All of these steps are mandatory. Finally the workflow starts the GUI and uninstalls. GitHub's Windows Server runner is not a pristine consumer Windows 10/11 VM — see PACKAGING.md.
