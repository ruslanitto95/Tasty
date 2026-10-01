# Security review (v0.1.0)

| Area | Measure |
|---|---|
| API key leakage | Stored via `keyring` (Windows Credential Manager); never in settings, logs (redaction of `Bearer …`/`sk-…`), diagnostics or Git. Without a secure backend the key lives in memory only. |
| Transport | LLM base URL must be `https://`; plain `http://` is accepted only for loopback/private addresses. Connect/read/overall timeouts, bounded retries with backoff. |
| Model download | Pinned URLs + **size + SHA-256** in `resources/model_manifest.json`; download to `.part`, verify, atomic rename. A verified marker (size + mtime + hash) avoids re-hashing on start; any mismatch marks the model CORRUPT. |
| Arbitrary code execution | The checkpoint is loaded with `torch.load(weights_only=True)` and an explicit allow-list of 7 omegaconf/builtin types (no pickle code execution). Before `hydra.instantiate`, every `_target_` in the config must start with `gigaam.`/`torchaudio.`. GigaAM source is pinned to a reviewed commit (`uv.lock`); no `trust_remote_code`. |
| Path traversal | Temp file names are reduced to a basename; temp dirs are random UUIDs under the app data dir. |
| Temporary files | Deleted after transcription, at visit end, on exit and on next start (orphans). Uninstaller removes `temp` and `logs`. |
| Logging | `RedactingFormatter` + `RedactingFilter` redact Cyrillic, e-mails, phone numbers and secrets after formatting, including tracebacks. A CI step fails if Cyrillic appears in the GUI log. |
| Crashes | Global `sys.excepthook`/`threading.excepthook`: the user sees «Произошла техническая ошибка. Текущий текст не был отправлен или сохранён.»; the traceback goes to the redacted log only. |
| Dependencies | Every version is locked in `uv.lock`; CI runs `pip-audit` on the runtime set (torch/torchaudio `+cpu` builds come from the PyTorch index and are not covered by the PyPI advisory DB; track PyTorch security advisories manually). GigaAM's optional `onnx` pin is overridden to a patched release; onnx is not shipped. No floating `latest`. |
| Updates | No self-made updater (none in v0.1). |
| Single instance | `QLockFile` prevents two instances competing for the microphone. |

Report vulnerabilities privately to the maintainers; do not attach patient data.
