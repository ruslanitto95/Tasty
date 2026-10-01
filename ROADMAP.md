# Roadmap

- **v0.1 (this release)** — complaints + history of present illness, local GigaAM, clipboard.
- **v0.2** — MediLog AutoFill via Windows UI Automation (FlaUI/pywinauto) behind `MedicalRecordIntegration`; never presses «Подписать», «Закрыть приём», «Сохранить окончательно».
- **v0.3** — ЛОР-статус (structured examination entry by the doctor).
- **v0.4** — Local LLM bundled/managed (current builds already accept a local OpenAI-compatible URL).
- **v0.5** — specialty templates.
- **v0.6** — clinical decision support (separate safety case).

Technical backlog: speaker diarisation (`SpeakerResolver`), PyTorch vs ONNX benchmark for GigaAM (ONNX export exists upstream), signed installer, optional encrypted crash recovery.
