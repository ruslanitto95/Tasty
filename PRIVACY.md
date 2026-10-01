# Privacy

```
Microphone → local Silero VAD → local GigaAM → text → (optional) configured LLM → doctor
```

**Audio is never sent to any cloud speech-recognition service.** Speech recognition runs entirely on this computer.

| Data | Where | Lifetime |
|---|---|---|
| Raw audio blocks | RAM (pre-roll ring buffer, current segment) | Released immediately after the segment is transcribed |
| Pending segments under load | `%LOCALAPPDATA%\MedicalVisitAssistant\temp\<random UUID>\sNNNN.wav` | Deleted after transcription; whole dir deleted at visit end; orphaned dirs deleted on next start |
| Transcript, facts, drafts, warnings | RAM only | Destroyed on «Новый приём», cancel or exit |
| Text sent to LLM | Only if the doctor enabled AI; only the transcript text (segment ids, timestamps, text) and validated facts | Governed by the configured provider |
| Settings | `settings.json` | No PHI, no API key, no patient identifiers |
| API key | Windows Credential Manager (keyring) | Until removed |
| Logs | `logs\mva.log` (rotated, 3 × 1 MB) | Technical only; Cyrillic text, e-mails and phone numbers are redacted after formatting |
| Benchmarks | `benchmarks.jsonl` | Timing numbers only |
| Model | `models\` | Public GigaAM weights |

Defaults: `SAVE_AUDIO = false`, history = off, developer mode = off, AI = off. Developer mode can save test-recording audio to `dev_recordings\` only when explicitly enabled.

- File and directory names never contain patient name, card number or date of birth.
- The application does not ask for or extract the patient's name.
- No analytics, telemetry, crash reporting or Sentry; the only network traffic is the one-time model download from `cdn.chatwm.opensmodel.sberdevices.ru` and LLM calls to the URL the doctor configured.
- Cloud processing can be fully disabled («Настройки → ИИ»), or pointed at a local/clinic server (`localhost`/private network addresses are shown as local).
- The «Экспорт диагностики» report contains app/OS/model versions, device and error codes only.
