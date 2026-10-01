# Technical decisions (Phase 0 research)

| Topic | Decision | Why / source |
|---|---|---|
| STT | **GigaAM-v3 `v3_e2e_rnnt`** from the official `salute-developers/GigaAM` package, pinned to commit recorded in `uv.lock` and `model_manifest.json` | Russian SOTA family; e2e variants add punctuation + text normalisation; README documents `.transcribe` for ≤25 s audio and pyannote+HF token for long-form — so long-form is done by our own VAD segmentation instead. |
| Model distribution | Official CDN `cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v3_e2e_rnnt.ckpt` + tokenizer, pinned size + SHA-256 | Same files `gigaam.load_model` uses; no HF `trust_remote_code`; our own resumable, verified downloader. |
| Model loading | Re-implemented `load_model` tail: `torch.load(weights_only=True)` with allow-listed omegaconf types, target check, `GigaAMASR(cfg)` | Avoids pickle code execution and GigaAM's in-memory MD5 of the whole file; identical outputs to `model.transcribe` were verified on the official `example.wav`. |
| Inference input | Tensors via `model.forward` + `model.decoding.decode` | Avoids GigaAM's ffmpeg subprocess loader → no FFmpeg dependency. |
| PyTorch / torchaudio | **2.10.0** (CPU wheels) | GigaAM's `longform` extra pins `torch==2.10.*`, `torchaudio==2.10.*`; torchaudio ≥2.9 still provides `MelSpectrogram`. CUDA is optional: device AUTO/CPU/GPU. |
| Python | 3.12 | Supported by all pinned wheels on Windows x64. |
| transformers | not used | Only needed for the HF loading path. |
| VAD | **silero-vad 6.2.3** pip package, bundled JIT model, `load_silero_vad()`, 512-sample frames at 16 kHz | Local, MIT, no download; frame contract checked against package source. Own hysteresis + pre/post-roll. |
| Audio I/O | sounddevice (PortAudio, DLL in wheel) + soxr resampling + soundfile | WASAPI shared mode requires the device rate, so capture at native rate and resample to 16 kHz mono float32. |
| UI | PySide6 6.11 (Fusion, light/dark/system) | Required stack. |
| LLM | OpenAI-compatible Chat Completions with `json_schema` → `json_object` fallback | Works with cloud and local servers; provider abstraction allows a LocalLLMProvider later. |
| Packaging | PyInstaller onedir + Inno Setup | See PACKAGING.md. |
| ONNX | Not used in v0.1 | PyTorch CPU already runs at RTF ≈0.15 on 4 vCPU; ONNX evaluation is on the roadmap and should only be adopted if faster, smaller and equally accurate. |
