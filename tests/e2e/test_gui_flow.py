"""GUI end-to-end: Start → speak → Stop → GigaAM → draft → edit → copy.

Uses the real GigaAM model and a real capture device when a virtual loopback is
available (otherwise the bundled WAV is streamed through the same pipeline).
Set MVA_SCREENSHOT_DIR to store screenshots of each stage.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication

from mva.audio.capture import AudioCaptureService
from mva.core.state_machine import AppState
from mva.paths import self_test_wav
from mva.storage.app_settings import AppSettings, SettingsStore
from mva.storage.temp_sessions import TempSessionManager
from mva.ui.controller import AppController
from mva.ui.main_window import MainWindow
from tests.audio.test_microphone_real import _virtual_sink
from tests.helpers import FileCapture
from tests.scripted_llm import scripted_llm

pytestmark = [pytest.mark.model, pytest.mark.gui]


def shot(widget, name: str) -> None:
    target = os.environ.get("MVA_SCREENSHOT_DIR")
    if target:
        Path(target).mkdir(parents=True, exist_ok=True)
        widget.grab().save(str(Path(target) / f"{name}.png"))


def test_doctor_flow(qtbot, tmp_path, gigaam):
    store = SettingsStore(tmp_path / "settings.json")
    settings = AppSettings(onboarding_completed=True)
    settings.ai.enabled = True
    settings.ai.use_llm_formatter = False
    store.save(settings)
    real_mic = _virtual_sink()
    capture = AudioCaptureService() if real_mic else FileCapture(speed=1.0)
    c = AppController(
        store,
        provider=gigaam,
        manager=gigaam._manager,
        llm_factory=lambda _s, _k: scripted_llm(),
        capture=capture,
        temp_manager=TempSessionManager(tmp_path / "temp"),
    )
    window = MainWindow(c)
    qtbot.addWidget(window)
    window.resize(460, 680)
    window.show()
    c.startup()
    qtbot.waitUntil(lambda: c.state == AppState.IDLE, timeout=60000)
    shot(window, "1_idle")

    qtbot.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(window.start_button, Qt.MouseButton.LeftButton)  # double click is harmless
    assert c.state == AppState.RECORDING
    if real_mic:
        qtbot.wait(500)
        subprocess.run(["paplay", str(self_test_wav())], check=True, timeout=60)
        qtbot.wait(1500)
    else:
        qtbot.wait(21000)
    qtbot.waitUntil(lambda: window.segments_label.text().endswith(("5", "6")), timeout=15000)
    shot(window, "2_recording")
    qtbot.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: c.state == AppState.REVIEW, timeout=60000)
    qtbot.waitUntil(lambda: bool(window.complaints.text()), timeout=10000)
    shot(window, "3_review")

    assert window.complaints.text() == "Затруднение носового дыхания, преимущественно справа."
    assert "около 7 дней" in window.history.text()
    assert "Називин" not in window.history.text()

    window.complaints.edit.setPlainText(window.complaints.text() + " Заложенность ушей.")
    assert window.complaints.modified_by_user
    qtbot.mouseClick(window.complaints.copy_button, Qt.MouseButton.LeftButton)
    assert QGuiApplication.clipboard().text().endswith("Заложенность ушей.")
    window.copy_all()
    assert QGuiApplication.clipboard().text().startswith("Жалобы: Затруднение носового дыхания")
    assert "\nАнамнез заболевания: Считает себя больным" in QGuiApplication.clipboard().text()

    # AI may never overwrite the doctor's edit.
    window.complaints.set_ai_text("перезапись")
    assert window.complaints.text().endswith("Заложенность ушей.")
    assert TempSessionManager(tmp_path / "temp").count() == 0

    assert window.request_new_visit()  # copied -> no confirmation needed
    assert c.state == AppState.IDLE and not window.complaints.text()
    window._quitting = True
    c.shutdown()
