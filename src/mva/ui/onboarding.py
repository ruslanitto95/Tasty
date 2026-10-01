"""First-run wizard: microphone → GigaAM (automatic download + load + WAV test) → AI → test → done."""

from __future__ import annotations

from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from mva.audio.capture import list_input_devices, resolve_device
from mva.core.state_machine import AppState
from mva.diagnostics.self_test import MIN_KEYWORDS, keyword_hits
from mva.paths import self_test_wav
from mva.transcription.session import SessionTranscriber, transcribe_file
from mva.ui.controller import AppController
from mva.ui.strings import error_text, tr
from mva.ui.tasks import BackgroundTask
from mva.ui.widgets import LevelMeter


class MicPage(QWizardPage):
    def __init__(self, c: AppController) -> None:
        super().__init__()
        self.c = c
        self.setTitle("Шаг 1. Микрофон")
        layout = QVBoxLayout(self)
        self.combo = QComboBox()
        self.combo.addItem(tr("mic_default"), None)
        for dev in list_input_devices(refresh=True):
            self.combo.addItem(dev.name, (dev.name, dev.hostapi))
        layout.addWidget(QLabel("Выберите микрофон, который будет слушать приём:"))
        layout.addWidget(self.combo)
        if self.combo.count() == 1:
            layout.addWidget(
                QLabel("⚠ Микрофоны не найдены. Подключите микрофон — его можно выбрать позже.")
            )

    def validatePage(self) -> bool:
        data = self.combo.currentData()
        s = self.c.settings.model_copy(deep=True)
        s.audio.device_name = data[0] if data else None
        s.audio.device_hostapi = data[1] if data else None
        self.c.save_settings(s)
        return True


class GigaAMPage(QWizardPage):
    """Shows the automatic download/load started at app launch, then runs the WAV test."""

    def __init__(self, c: AppController) -> None:
        super().__init__()
        self.c = c
        self.setTitle("Шаг 2. Проверка GigaAM")
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Модель распознавания речи скачивается и проверяется автоматически (около 450 МБ, один раз)."
            )
        )
        self.status = QLabel(tr("model_loading"))
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.bar = QProgressBar()
        layout.addWidget(self.bar)
        self.result = QLabel("")
        self.result.setWordWrap(True)
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.result)
        self.retry = QPushButton(tr("retry"))
        self.retry.setVisible(False)
        self.retry.clicked.connect(self._retry)
        layout.addWidget(self.retry)
        self.ok = False
        self._task: BackgroundTask | None = None
        c.model_progress.connect(self._progress)
        c.model_ready.connect(lambda _d: self._run_test())
        c.error.connect(self._error)

    def initializePage(self) -> None:
        if self.c.provider.is_loaded:
            self._run_test()
        elif self.c.state == AppState.ERROR:
            self.retry.setVisible(True)

    def _progress(self, stage: str, fraction: float) -> None:
        if stage == "download":
            self.bar.setRange(0, 100)
            self.bar.setValue(int(fraction * 100))
            self.status.setText(tr("model_download", pct=int(fraction * 100)))
        else:
            self.bar.setRange(0, 0)
            self.status.setText(tr("model_loading"))

    def _error(self, code: str, _detail: str) -> None:
        if self.c.state == AppState.ERROR:
            self.status.setText(error_text(code))
            self.bar.setRange(0, 1)
            self.bar.setValue(0)
            self.retry.setVisible(True)

    def _retry(self) -> None:
        self.retry.setVisible(False)
        self.c.load_model()

    def _run_test(self) -> None:
        if self._task is not None:
            return
        self.bar.setRange(0, 0)
        self.status.setText("Модель загружена. Распознаю тестовую запись…")
        provider = self.c.provider

        def work(_t: BackgroundTask) -> object:
            transcript, stats = transcribe_file(provider, self_test_wav())
            return transcript.text(), stats.rtf

        def done(result: object) -> None:
            self.bar.setRange(0, 1)
            self.bar.setValue(1)
            self._task = None
            if isinstance(result, Exception):
                self.status.setText(error_text("stt_failed"))
                self.retry.setVisible(True)
                return
            text, rtf = cast(tuple[str, float], result)
            self.ok = len(keyword_hits(text)) >= MIN_KEYWORDS
            device = provider.info().device.upper()
            self.status.setText(
                f"GigaAM: {'OK' if self.ok else 'распознавание неточное'} — устройство {device}, RTF {rtf:.2f}"
            )
            self.result.setText(f"Распознано: «{text}»")
            self.completeChanged.emit()

        task = BackgroundTask(work, self)
        task.finished.connect(done)
        task.failed.connect(done)
        self._task = task
        task.start()

    def isComplete(self) -> bool:
        return self.ok


class AIPage(QWizardPage):
    def __init__(self, c: AppController) -> None:
        super().__init__()
        self.c = c
        self.setTitle("Шаг 3. ИИ для формирования документа")
        f = QFormLayout(self)
        note = QLabel(
            "Жалобы и анамнез формирует ИИ по ТЕКСТУ расшифровки (аудио не отправляется). "
            "Можно указать облачный OpenAI-совместимый сервис или локальный сервер клиники. "
            "Шаг можно пропустить — тогда будет доступна только расшифровка."
        )
        note.setWordWrap(True)
        f.addRow(note)
        self.enabled = QCheckBox("Использовать ИИ")
        self.enabled.setChecked(c.settings.ai.enabled)
        self.url = QLineEdit(c.settings.ai.base_url)
        self.model = QLineEdit(c.settings.ai.model)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        f.addRow(self.enabled)
        f.addRow("Base URL:", self.url)
        f.addRow("Модель:", self.model)
        f.addRow("API-ключ:", self.key)

    def validatePage(self) -> bool:
        s = self.c.settings.model_copy(deep=True)
        s.ai.enabled = self.enabled.isChecked()
        s.ai.base_url = self.url.text().strip()
        s.ai.model = self.model.text().strip()
        if self.key.text().strip():
            self.c.credentials.set_api_key(self.key.text().strip())
        self.c.save_settings(s)
        return True


class TestPage(QWizardPage):
    """Records a few seconds from the selected microphone and transcribes it with GigaAM."""

    SECONDS = 6.0

    def __init__(self, c: AppController) -> None:
        super().__init__()
        self.c = c
        self.setTitle("Шаг 4. Проверка микрофона и распознавания")
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Нажмите «Записать» и скажите фразу, например: «Нос заложен три дня».")
        )
        self.meter = LevelMeter()
        layout.addWidget(self.meter)
        self.button = QPushButton(f"Записать {int(self.SECONDS)} секунд")
        self.button.clicked.connect(self._record)
        layout.addWidget(self.button)
        self.result = QLabel("")
        self.result.setWordWrap(True)
        layout.addWidget(self.result)

    def _record(self) -> None:
        from mva.audio.capture import AudioCaptureService

        self.button.setEnabled(False)
        self.result.setText("Идёт запись…")
        c = self.c

        def work(task: BackgroundTask) -> object:
            import time

            device = resolve_device(c.settings.audio.device_name, c.settings.audio.device_hostapi)
            session = SessionTranscriber(c.provider, None, c.settings.speech.vad)
            session.start()
            errors: list = []
            capture = AudioCaptureService()
            capture.start(device, session.feed, lambda r: task.progress.emit(r.peak), errors.append)
            end = time.monotonic() + self.SECONDS
            while time.monotonic() < end and capture.running:
                time.sleep(0.05)
            capture.stop()
            transcript = session.finish()
            if errors:
                raise errors[0]
            return transcript.text()

        def done(result: object) -> None:
            self.button.setEnabled(True)
            self.meter.reset()
            if isinstance(result, Exception):
                self.result.setText(error_text(getattr(result, "code", "microphone_unavailable")))
            elif not str(result).strip():
                self.result.setText("Речь не распознана. Проверьте микрофон и попробуйте ещё раз.")
            else:
                self.result.setText(f"Распознано: «{result}»")

        task = BackgroundTask(work, self)
        task.progress.connect(lambda peak: self.meter.set_level(float(peak)))
        task.finished.connect(done)
        task.failed.connect(done)
        task.start()


class DonePage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Шаг 5. Готово")
        layout = QVBoxLayout(self)
        label = QLabel(
            "Всё готово.\n\nCtrl+Alt+R — начать и завершить приём.\nCtrl+Alt+1 — копировать жалобы.\n"
            "Ctrl+Alt+2 — копировать анамнез.\n\nРезультат всегда является черновиком: проверьте его перед "
            "внесением в медицинскую карту."
        )
        label.setWordWrap(True)
        layout.addWidget(label)


class OnboardingWizard(QWizard):
    def __init__(self, c: AppController, parent=None) -> None:
        super().__init__(parent)
        self.c = c
        self.setWindowTitle("Первый запуск — " + tr("app_title"))
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.resize(600, 460)
        self.gigaam_page = GigaAMPage(c)
        self.test_page = TestPage(c)
        for page in (MicPage(c), self.gigaam_page, AIPage(c), self.test_page, DonePage()):
            self.addPage(page)

    def accept(self) -> None:
        s = self.c.settings.model_copy(deep=True)
        s.onboarding_completed = True
        self.c.save_settings(s)
        super().accept()
