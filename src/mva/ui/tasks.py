"""Run a blocking callable on a worker thread and deliver the result in the GUI thread."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Signal

log = logging.getLogger(__name__)


class BackgroundTask(QObject):
    finished = Signal(object)
    failed = Signal(object)
    progress = Signal(object)

    def __init__(self, fn: Callable[[BackgroundTask], Any], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._fn = fn
        self.cancel = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="ui-task", daemon=True).start()

    def _run(self) -> None:
        try:
            result = self._fn(self)
        except Exception as exc:
            log.warning("Background task failed: %s", type(exc).__name__)
            self.failed.emit(exc)
            return
        self.finished.emit(result)
