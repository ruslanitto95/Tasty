"""Unattended verification of the real first-run GUI path (used by Windows/Linux CI).

Drives the same OnboardingWizard a doctor sees on a clean profile: the controller
downloads + verifies + loads GigaAM, the «Проверка GigaAM» page transcribes the bundled
WAV, and the «Проверка микрофона» page records from the selected microphone and
transcribes it. Nothing is simulated; only the button clicks are automated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer

from mva.diagnostics.self_test import MIN_KEYWORDS, keyword_hits
from mva.ui.controller import AppController
from mva.ui.onboarding import OnboardingWizard

log = logging.getLogger(__name__)

MIC_MIN_KEYWORDS = 3


class FirstRunCheck(QObject):
    def __init__(
        self,
        controller: AppController,
        wizard: OnboardingWizard,
        report_path: Path,
        mic_seconds: float,
        status_before: str,
        on_done: Any,
        timeout_s: float = 900.0,
    ) -> None:
        super().__init__(wizard)
        self.c = controller
        self.w = wizard
        self.path = report_path
        self.on_done = on_done
        self.finished = False
        self.report: dict[str, Any] = {
            "ok": False,
            "mode": "gui_first_run",
            "model_status_before": status_before,
            "errors": [],
        }
        wizard.test_page.SECONDS = mic_seconds
        wizard.gigaam_page.test_finished.connect(self._wav_done)
        wizard.test_page.recorded.connect(self._mic_done)
        controller.error.connect(lambda code, _d: self._fail(f"controller:{code}"))
        QTimer.singleShot(int(timeout_s * 1000), lambda: self._fail("timeout"))

    def start(self) -> None:
        self.w.show()
        self.w.next()  # Mic page -> GigaAM page (download/load already running)

    def _wav_done(self, ok: bool, text: str) -> None:
        info = self.c.provider.info()
        self.report.update(
            model_status_after=self.c.manager.status().value,
            model=info.model,
            revision=info.revision,
            device=info.device,
            wav_text=text,
            wav_keywords=keyword_hits(text),
            wav_ok=ok and len(keyword_hits(text)) >= MIN_KEYWORDS,
        )
        if not self.report["wav_ok"]:
            self._fail("wav_transcription")
            return
        self.w.next()  # -> AI page (left disabled)
        self.w.next()  # -> microphone test page
        self.report["mic_device"] = self.c.settings.audio.device_name or "system default"
        self.w.test_page._record()

    def _mic_done(self, text: str, error: str) -> None:
        hits = keyword_hits(text)
        self.report.update(mic_text=text, mic_keywords=hits, mic_error=error)
        self.report["mic_ok"] = not error and len(hits) >= MIC_MIN_KEYWORDS
        if not self.report["mic_ok"]:
            self._fail("microphone_transcription")
            return
        self.report["ok"] = True
        self.w.next()
        self.w.accept()
        self._finish()

    def _fail(self, reason: str) -> None:
        if self.finished:
            return
        self.report["errors"].append(reason)
        self._finish()

    def _finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info(
            "First-run check finished: ok=%s errors=%s", self.report["ok"], self.report["errors"]
        )
        if self.w.isVisible():
            self.w.done(0)
        self.on_done(bool(self.report["ok"]))
