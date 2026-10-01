"""Settings: General, Audio, Speech, AI, Privacy, Hotkeys, Diagnostics, About."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mva import __version__
from mva.audio.capture import list_input_devices, record_seconds, resolve_device
from mva.diagnostics import health_check
from mva.diagnostics.benchmark import append_benchmark, run_benchmark
from mva.llm.base import LLMError, is_local_url
from mva.llm.openai_compatible import OpenAICompatibleProvider
from mva.paths import benchmarks_file, self_test_wav, temp_root
from mva.storage.app_settings import AppSettings, InferenceDevice, Theme
from mva.ui.autostart import set_autostart
from mva.ui.controller import AppController
from mva.ui.strings import error_text
from mva.ui.tasks import BackgroundTask
from mva.ui.theme import apply_theme
from mva.ui.widgets import LevelMeter


class SettingsDialog(QDialog):
    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.c = controller
        self.s: AppSettings = controller.settings.model_copy(deep=True)
        self.setWindowTitle("Настройки")
        self.resize(560, 520)
        self._tasks: list[BackgroundTask] = []
        tabs = QTabWidget()
        tabs.addTab(self._general(), "Общие")
        tabs.addTab(self._audio(), "Аудио")
        tabs.addTab(self._speech(), "Распознавание")
        tabs.addTab(self._ai(), "ИИ")
        tabs.addTab(self._privacy(), "Конфиденциальность")
        tabs.addTab(self._hotkeys(), "Горячие клавиши")
        tabs.addTab(self._diagnostics(), "Диагностика")
        tabs.addTab(self._about(), "О программе")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    def _task(self, fn, done, failed=None) -> BackgroundTask:
        task = BackgroundTask(fn, self)
        task.finished.connect(done)
        task.failed.connect(failed or (lambda exc: done(exc)))
        self._tasks.append(task)
        task.start()
        return task

    # ---- tabs ----------------------------------------------------------------------
    def _general(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.on_top = QCheckBox("Поверх других окон")
        self.on_top.setChecked(self.s.general.always_on_top)
        f.addRow(self.on_top)
        self.autostart = QCheckBox("Запускать вместе с Windows")
        self.autostart.setChecked(self.s.general.start_with_windows)
        f.addRow(self.autostart)
        self.theme = QComboBox()
        for value, label in (
            (Theme.SYSTEM, "Как в системе"),
            (Theme.LIGHT, "Светлая"),
            (Theme.DARK, "Тёмная"),
        ):
            self.theme.addItem(label, value)
        self.theme.setCurrentIndex(self.theme.findData(self.s.general.theme))
        f.addRow("Тема:", self.theme)
        language = QComboBox()
        language.addItem("Русский")
        language.setEnabled(False)
        f.addRow("Язык:", language)
        return w

    def _audio(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.mic = QComboBox()
        self.mic.addItem("Системный по умолчанию", None)
        for dev in list_input_devices(refresh=not self.c.capture.running):
            self.mic.addItem(f"{dev.name} ({dev.hostapi})", (dev.name, dev.hostapi))
            if dev.name == self.s.audio.device_name:
                self.mic.setCurrentIndex(self.mic.count() - 1)
        f.addRow("Микрофон:", self.mic)
        self.mic_meter = LevelMeter()
        f.addRow("Уровень:", self.mic_meter)
        self.mic_result = QLabel("")
        test = QPushButton("Проверить микрофон (3 с)")
        test.clicked.connect(self._test_mic)
        f.addRow(test, self.mic_result)
        return w

    def _speech(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        m = self.c.manager.manifest
        f.addRow("Модель GigaAM:", QLabel(f"{m.display_name}\n{m.model_name}"))
        f.addRow("Ревизия:", QLabel(f"{m.revision}\ngigaam@{m.gigaam_commit[:12]}"))
        self.device = QComboBox()
        for value, label in (
            (InferenceDevice.AUTO, "AUTO"),
            (InferenceDevice.CPU, "CPU"),
            (InferenceDevice.GPU, "GPU"),
        ):
            self.device.addItem(label, value)
        self.device.setCurrentIndex(self.device.findData(self.s.speech.device))
        f.addRow("Устройство:", self.device)
        f.addRow(QLabel("Изменение устройства применяется после перезапуска."))
        vad = self.s.speech.vad
        self.vad_threshold = QDoubleSpinBox()
        self.vad_threshold.setRange(0.1, 0.95)
        self.vad_threshold.setSingleStep(0.05)
        self.vad_threshold.setValue(vad.threshold)
        self.vad_silence = self._spin(200, 3000, vad.min_silence_ms)
        self.vad_pre = self._spin(0, 1000, vad.pre_roll_ms)
        self.vad_post = self._spin(0, 1500, vad.post_roll_ms)
        f.addRow("VAD: порог речи", self.vad_threshold)
        f.addRow("VAD: пауза конца фразы, мс", self.vad_silence)
        f.addRow("VAD: pre-roll, мс", self.vad_pre)
        f.addRow("VAD: post-roll, мс", self.vad_post)
        return w

    @staticmethod
    def _spin(lo: int, hi: int, value: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(lo, hi)
        box.setSingleStep(50)
        box.setValue(value)
        return box

    def _ai(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        ai = self.s.ai
        self.ai_enabled = QCheckBox("Использовать ИИ для формирования документа")
        self.ai_enabled.setChecked(ai.enabled)
        f.addRow(self.ai_enabled)
        self.ai_url = QLineEdit(ai.base_url)
        self.ai_model = QLineEdit(ai.model)
        self.ai_key = QLineEdit()
        self.ai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.ai_key.setPlaceholderText(
            "сохранён в защищённом хранилище" if self.c.credentials.get_api_key() else ""
        )
        self.ai_timeout = QDoubleSpinBox()
        self.ai_timeout.setRange(5, 600)
        self.ai_timeout.setValue(ai.timeout_s)
        self.ai_formatter = QCheckBox("Стилистическая обработка текста ИИ (с проверкой фактов)")
        self.ai_formatter.setChecked(ai.use_llm_formatter)
        f.addRow("Провайдер:", QLabel("OpenAI-совместимый API"))
        f.addRow("Base URL:", self.ai_url)
        f.addRow("Модель:", self.ai_model)
        f.addRow("API-ключ:", self.ai_key)
        f.addRow("Таймаут, с:", self.ai_timeout)
        f.addRow(self.ai_formatter)
        self.cloud_note = QLabel()
        self.cloud_note.setWordWrap(True)
        self.ai_url.textChanged.connect(self._update_cloud_note)
        self._update_cloud_note()
        f.addRow(self.cloud_note)
        self.ai_test_result = QLabel("")
        self.ai_test_result.setWordWrap(True)
        test = QPushButton("Проверить подключение")
        test.clicked.connect(self._test_ai)
        f.addRow(test, self.ai_test_result)
        if not self.c.credentials.secure_backend_available:
            f.addRow(
                QLabel(
                    "⚠ Защищённое хранилище недоступно: ключ хранится только до закрытия программы."
                )
            )
        return w

    def _update_cloud_note(self) -> None:
        url = self.ai_url.text().strip()
        if url and is_local_url(url):
            self.cloud_note.setText(
                "Локальный сервер: текст расшифровки не покидает компьютер/сеть клиники."
            )
        else:
            self.cloud_note.setText(
                "Облачная модель: в неё отправляется ТЕКСТ расшифровки (без аудио). "
                "Аудио никогда не отправляется в облако."
            )

    def _privacy(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        delete_audio = QCheckBox("Удалять аудио после распознавания")
        delete_audio.setChecked(True)
        delete_audio.setEnabled(False)
        f.addRow(delete_audio)
        history = QCheckBox("Сохранять историю приёмов")
        history.setChecked(False)
        history.setEnabled(False)
        f.addRow(history)
        text = QLabel(
            "Микрофон → локальный VAD → локальный GigaAM → текст → (по желанию) настроенный ИИ.\n"
            "Аудио никогда не отправляется в облачное распознавание и удаляется сразу после обработки.\n"
            "Текст приёма не сохраняется на диск; «Новый приём» удаляет все данные визита.\n"
            "Облачную обработку можно полностью отключить на вкладке «ИИ»."
        )
        text.setWordWrap(True)
        f.addRow(text)
        return w

    def _hotkeys(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        hk = self.s.hotkeys
        self.hk_start = QKeySequenceEdit(QKeySequence(hk.start_stop))
        self.hk_c1 = QKeySequenceEdit(QKeySequence(hk.copy_complaints))
        self.hk_c2 = QKeySequenceEdit(QKeySequence(hk.copy_history))
        self.hk_new = QKeySequenceEdit(QKeySequence(hk.new_visit))
        f.addRow("Начать / завершить приём:", self.hk_start)
        f.addRow("Копировать жалобы:", self.hk_c1)
        f.addRow("Копировать анамнез:", self.hk_c2)
        f.addRow("Новый приём:", self.hk_new)
        return w

    def _diagnostics(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self.diag = QPlainTextEdit()
        self.diag.setReadOnly(True)
        layout.addWidget(self.diag)
        run = QPushButton("Проверить систему")
        run.clicked.connect(self._run_health)
        layout.addWidget(run)
        selftest = QPushButton("Проверка распознавания GigaAM (тестовый WAV)")
        selftest.clicked.connect(self._run_selftest)
        layout.addWidget(selftest)
        export = QPushButton("Экспорт диагностики…")
        export.clicked.connect(self._export)
        layout.addWidget(export)
        self.dev_mode = QCheckBox("Режим разработчика")
        self.dev_mode.setChecked(self.s.developer.enabled)
        layout.addWidget(self.dev_mode)
        self.dev_audio = QCheckBox("Разработчик: сохранять аудио тестовых записей")
        self.dev_audio.setChecked(self.s.developer.save_test_audio)
        layout.addWidget(self.dev_audio)
        bench = QPushButton("Разработчик: бенчмарк распознавания (WAV)…")
        bench.clicked.connect(self._benchmark)
        layout.addWidget(bench)
        return w

    def _about(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        f.addRow("Версия:", QLabel(__version__))
        info = self.c.provider.info()
        f.addRow("Модель:", QLabel(info.model))
        f.addRow("Ревизия модели:", QLabel(info.revision))
        f.addRow("Устройство:", QLabel(info.device.upper() if self.c.provider.is_loaded else "—"))
        deps = health_check.dependency_versions()
        f.addRow("Компоненты:", QLabel("\n".join(f"{k} {v}" for k, v in deps.items())))
        f.addRow(QLabel("Лицензии: THIRD_PARTY_LICENSES.md в папке установки."))
        return w

    # ---- actions -------------------------------------------------------------------
    def _test_mic(self) -> None:
        data = self.mic.currentData()
        self.mic_result.setText("Говорите…")

        def work(_task: BackgroundTask) -> float:
            device = resolve_device(data[0] if data else None, data[1] if data else None)
            if device is None:
                raise RuntimeError("no device")
            audio = record_seconds(device, 3.0)
            return float(abs(audio).max()) if len(audio) else 0.0

        def done(result: object) -> None:
            if isinstance(result, Exception):
                self.mic_result.setText(
                    error_text(getattr(result, "code", "microphone_unavailable"))
                )
                return
            peak = float(result)  # type: ignore[arg-type]
            self.mic_meter.set_level(peak)
            if peak < 0.005:
                self.mic_result.setText("Сигнал не обнаружен")
            elif peak < 0.03:
                self.mic_result.setText("Слишком тихо")
            else:
                self.mic_result.setText("Микрофон: OK")

        self._task(work, done)

    def _llm_from_fields(self) -> OpenAICompatibleProvider:
        key = self.ai_key.text().strip() or self.c.credentials.get_api_key()
        return OpenAICompatibleProvider(
            self.ai_url.text().strip(), self.ai_model.text().strip(), key, self.ai_timeout.value()
        )

    def _test_ai(self) -> None:
        self.ai_test_result.setText("Проверяю…")
        try:
            provider = self._llm_from_fields()
        except ValueError as exc:
            self.ai_test_result.setText(f"Ошибка адреса: {exc}")
            return

        def done(result: object) -> None:
            if isinstance(result, LLMError):
                self.ai_test_result.setText(error_text(result.code))
            elif isinstance(result, Exception):
                self.ai_test_result.setText(error_text("llm_unavailable"))
            else:
                self.ai_test_result.setText("ИИ: OK")

        self._task(lambda _t: provider.test_connection(), done)

    def _run_health(self) -> None:
        self.diag.setPlainText("Проверяю…")
        c = self.c

        def work(_task: BackgroundTask) -> list[health_check.CheckResult]:
            checks = [
                health_check.check_microphone(),
                health_check.check_model(c.manager, c.provider.is_loaded, c.provider.info().device),
                health_check.check_temp_storage(temp_root()),
                health_check.check_audio_format(),
            ]
            llm = c.llm()
            if llm is None:
                checks.append(health_check.CheckResult("ai_provider", False, "not configured"))
            else:
                try:
                    llm.test_connection()
                    checks.append(
                        health_check.CheckResult(
                            "ai_provider", True, "cloud" if llm.is_cloud else "local"
                        )
                    )
                except LLMError as exc:
                    checks.append(health_check.CheckResult("ai_provider", False, exc.code))
            return checks

        names = {
            "microphone": "Микрофон",
            "gigaam": "GigaAM",
            "ai_provider": "AI Provider",
            "temp_storage": "Временное хранилище",
            "audio_format": "Формат аудио",
        }

        def done(result: object) -> None:
            if isinstance(result, Exception):
                self.diag.setPlainText(f"Ошибка: {type(result).__name__}")
                return
            self._checks = result
            self.diag.setPlainText(
                "\n".join(
                    f"{names.get(r.name, r.name)}: {'OK' if r.ok else 'ОШИБКА'} — {r.detail}"
                    for r in result
                )  # type: ignore[union-attr]
            )

        self._task(work, done)

    def _run_selftest(self) -> None:
        self.diag.setPlainText("Распознаю тестовый WAV…")
        c = self.c

        def work(_task: BackgroundTask) -> object:
            from mva.diagnostics.self_test import keyword_hits
            from mva.transcription.session import transcribe_file

            if not c.provider.is_loaded:
                raise RuntimeError("model not loaded")
            transcript, stats = transcribe_file(c.provider, self_test_wav())
            return transcript.text(), keyword_hits(transcript.text()), stats.rtf

        def done(result: object) -> None:
            if isinstance(result, Exception):
                self.diag.setPlainText(error_text("model_unavailable"))
                return
            text, hits, rtf = result  # type: ignore[misc]
            ok = len(hits) >= 4
            self.diag.setPlainText(
                f"GigaAM: {'OK' if ok else 'ОШИБКА'} (RTF {rtf:.2f})\n\n«{text}»"
            )

        self._task(work, done)

    def _benchmark(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "WAV для бенчмарка", "", "Audio (*.wav *.flac)")
        if not path or not self.c.provider.is_loaded:
            return
        self.diag.setPlainText("Бенчмарк…")
        provider = self.c.provider

        def work(_task: BackgroundTask) -> object:
            result, _text = run_benchmark(provider, Path(path))
            append_benchmark(benchmarks_file(), result)
            return result

        def done(result: object) -> None:
            if isinstance(result, Exception):
                self.diag.setPlainText(f"Ошибка: {type(result).__name__}")
                return
            r = result
            self.diag.setPlainText(
                f"Длительность аудио: {r.audio_seconds} с\nВремя распознавания: {r.transcription_seconds} с\n"  # type: ignore[attr-defined]
                f"RTF: {r.rtf}\nRAM: ~{r.rss_mb} МБ\nУстройство: {r.device}\nМодель: {r.model}"  # type: ignore[attr-defined]
            )

        self._task(work, done)

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт диагностики", "mva-diagnostics.json", "JSON (*.json)"
        )
        if not path:
            return
        checks = getattr(self, "_checks", [])
        report = health_check.diagnostic_report(
            self.c.manager, self.c.provider.info().device, checks, self.c.error_codes
        )
        Path(path).write_text(report, encoding="utf-8")
        self.diag.setPlainText(f"Сохранено: {path}\n(без расшифровок и медицинского текста)")

    def _save(self) -> None:
        s = self.s
        s.general.always_on_top = self.on_top.isChecked()
        s.general.theme = self.theme.currentData()
        if self.autostart.isChecked() != s.general.start_with_windows:
            set_autostart(self.autostart.isChecked())
        s.general.start_with_windows = self.autostart.isChecked()
        data = self.mic.currentData()
        s.audio.device_name = data[0] if data else None
        s.audio.device_hostapi = data[1] if data else None
        s.speech.device = self.device.currentData()
        s.speech.vad.threshold = self.vad_threshold.value()
        s.speech.vad.min_silence_ms = self.vad_silence.value()
        s.speech.vad.pre_roll_ms = self.vad_pre.value()
        s.speech.vad.post_roll_ms = self.vad_post.value()
        s.ai.enabled = self.ai_enabled.isChecked()
        s.ai.base_url = self.ai_url.text().strip()
        s.ai.model = self.ai_model.text().strip()
        s.ai.timeout_s = self.ai_timeout.value()
        s.ai.use_llm_formatter = self.ai_formatter.isChecked()
        if self.ai_key.text().strip():
            self.c.credentials.set_api_key(self.ai_key.text().strip())
        for edit, attr in (
            (self.hk_start, "start_stop"),
            (self.hk_c1, "copy_complaints"),
            (self.hk_c2, "copy_history"),
            (self.hk_new, "new_visit"),
        ):
            seq = edit.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
            if seq:
                setattr(s.hotkeys, attr, seq)
        s.developer.enabled = self.dev_mode.isChecked()
        s.developer.save_test_audio = self.dev_audio.isChecked() and s.developer.enabled
        self.c.save_settings(s)
        app = QGuiApplication.instance()
        if app is not None:
            apply_theme(app, s.general.theme)  # type: ignore[arg-type]
        self.accept()
