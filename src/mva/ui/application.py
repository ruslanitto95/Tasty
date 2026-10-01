"""QApplication bootstrap: single instance, crash handler, theme, first-run wizard."""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from types import TracebackType

from PySide6.QtCore import QLibraryInfo, QLocale, QLockFile, QObject, QTimer, QTranslator, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from mva.app import quiet_native_warnings
from mva.config import APP_NAME, APP_ORG
from mva.paths import app_data_dir, settings_file
from mva.storage.app_settings import SettingsStore
from mva.ui.controller import AppController
from mva.ui.main_window import MainWindow
from mva.ui.strings import tr
from mva.ui.theme import apply_theme

log = logging.getLogger(__name__)


class _CrashBridge(QObject):
    crashed = Signal()


def install_crash_handler(bridge: _CrashBridge) -> None:
    def handle(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        # Technical traceback goes to the PHI-redacting log only; the user sees a plain message.
        log.error(
            "Unhandled exception:\n%s", "".join(traceback.format_exception(exc_type, exc, tb))
        )
        bridge.crashed.emit()

    sys.excepthook = handle
    threading.excepthook = lambda args: handle(
        args.exc_type, args.exc_value or Exception(), args.exc_traceback
    )


def run_gui(smoke_seconds: float = 0.0) -> int:
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORG)
    app.setQuitOnLastWindowClosed(False)
    translator = QTranslator(app)
    if translator.load(
        QLocale("ru"), "qtbase", "_", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    ):
        app.installTranslator(translator)
    lock = QLockFile(str(app_data_dir() / "instance.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(100):
        QMessageBox.information(None, APP_NAME, tr("already_running"))
        return 0
    quiet_native_warnings()
    store = SettingsStore(settings_file())
    controller = AppController(store)
    apply_theme(app, controller.settings.general.theme)
    bridge = _CrashBridge()
    window = MainWindow(controller)
    bridge.crashed.connect(lambda: QMessageBox.critical(window, APP_NAME, tr("crash")))
    install_crash_handler(bridge)
    window.show()
    controller.startup()
    if not controller.settings.onboarding_completed and smoke_seconds <= 0:
        from mva.ui.onboarding import OnboardingWizard

        QTimer.singleShot(200, lambda: OnboardingWizard(controller, window).exec())
    if smoke_seconds > 0:
        QTimer.singleShot(int(smoke_seconds * 1000), window.quit)
    code = app.exec()
    lock.unlock()
    return int(code)
