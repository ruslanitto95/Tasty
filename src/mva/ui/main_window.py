"""Main window: Idle → Recording → Processing → Review. Minimal, calm, keyboard friendly."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QColor, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QSystemTrayIcon,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mva.audio.capture import list_input_devices
from mva.clinical.schemas import DocumentDraft
from mva.core.state_machine import AppState
from mva.integrations.medialog.base import format_all
from mva.transcription.models import format_ts
from mva.ui.controller import AppController
from mva.ui.hotkeys import HotkeyManager
from mva.ui.strings import error_text, tr
from mva.ui.widgets import LevelMeter, Toast

log = logging.getLogger(__name__)


def dot_icon(colour: str) -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(colour))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(6, 6, 52, 52)
    painter.end()
    return QIcon(pix)


class SectionEditor(QWidget):
    """Editable section with its own copy button; tracks manual edits."""

    def __init__(
        self, title: str, placeholder: str, on_copy, on_source, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        label = QLabel(title)
        label.setObjectName("section")
        header.addWidget(label)
        header.addStretch()
        self.copy_button = QPushButton(tr("copy"))
        self.copy_button.clicked.connect(on_copy)
        header.addWidget(self.copy_button)
        layout.addLayout(header)
        self.edit = QTextEdit()
        self.edit.setAcceptRichText(False)
        self.edit.setPlaceholderText(placeholder)
        self.edit.setTabChangesFocus(True)
        self.edit.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.edit.customContextMenuRequested.connect(self._menu)
        self.edit.textChanged.connect(self._changed)
        label.setBuddy(self.edit)
        layout.addWidget(self.edit)
        self._on_source = on_source
        self._programmatic = False
        self.modified_by_user = False
        self.copied = True

    def set_ai_text(self, text: str) -> None:
        if self.modified_by_user:
            return  # AI may never overwrite the doctor's edits
        self._programmatic = True
        self.edit.setPlainText(text)
        self._programmatic = False
        self.copied = not text

    def text(self) -> str:
        return self.edit.toPlainText().strip()

    def clear(self) -> None:
        self._programmatic = True
        self.edit.clear()
        self._programmatic = False
        self.modified_by_user = False
        self.copied = True

    def _changed(self) -> None:
        if not self._programmatic:
            self.modified_by_user = True
            self.copied = False

    def _menu(self, pos) -> None:
        menu = self.edit.createStandardContextMenu()
        menu.addSeparator()
        action = menu.addAction(tr("show_source"))
        cursor = self.edit.cursorForPosition(pos)
        selected = self.edit.textCursor().selectedText() or ""
        position = cursor.position()
        action.triggered.connect(
            lambda: self._on_source(self.edit.toPlainText(), position, selected)
        )
        menu.exec(self.edit.mapToGlobal(pos))


class MainWindow(QMainWindow):
    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self.c = controller
        self.setWindowTitle(tr("app_title"))
        self.setWindowIcon(dot_icon("#2e7d32"))
        self.setMinimumSize(420, 560)
        self._draft: DocumentDraft | None = None
        self._model_status = tr("model_loading")
        self._quitting = False

        self.stack = QStackedWidget()
        self.idle_page = self._build_idle()
        self.record_page = self._build_recording()
        self.process_page = self._build_processing()
        self.review_page = self._build_review()
        for page in (self.idle_page, self.record_page, self.process_page, self.review_page):
            self.stack.addWidget(page)
        self.setCentralWidget(self.stack)
        self.toast = Toast(self)
        self._status_ai = QLabel()
        self._status_model = QLabel()
        self.statusBar().addWidget(self._status_model)
        self.statusBar().addPermanentWidget(self._status_ai)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)
        self.hotkeys = HotkeyManager(self)
        self.tray = self._build_tray()

        c = controller
        c.state_changed.connect(self._on_state)
        c.model_progress.connect(self._on_model_progress)
        c.model_ready.connect(self._on_model_ready)
        c.error.connect(self._on_error)
        c.notice.connect(self._on_notice)
        c.level.connect(self._on_level)
        c.segments_changed.connect(lambda n: self.segments_label.setText(tr("segments", n=n)))
        c.lagging.connect(lambda lag: self.lag_label.setVisible(lag))
        c.finalize_progress.connect(self._on_finalize_progress)
        c.stage.connect(self._on_stage)
        c.draft_ready.connect(self._on_draft)
        c.analysis_failed.connect(self._on_analysis_failed)
        c.settings_changed.connect(self.apply_settings)
        self.apply_settings()
        self._restore_geometry()
        self._on_state("", c.state.value)

    # ---- pages -----------------------------------------------------------------------
    def _build_idle(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel(tr("app_title"))
        title.setStyleSheet("font-size: 16pt; font-weight: 700;")
        layout.addWidget(title)
        self.model_label = QLabel(tr("model_loading"))
        layout.addWidget(self.model_label)
        self.model_progress_bar = QProgressBar()
        self.model_progress_bar.setVisible(False)
        layout.addWidget(self.model_progress_bar)
        self.model_retry = QPushButton(tr("retry"))
        self.model_retry.setVisible(False)
        self.model_retry.clicked.connect(self.c.load_model)
        layout.addWidget(self.model_retry)
        mic_row = QHBoxLayout()
        mic_label = QLabel(tr("mic_label"))
        mic_row.addWidget(mic_label)
        self.mic_combo = QComboBox()
        self.mic_combo.setAccessibleName(tr("mic_label"))
        mic_label.setBuddy(self.mic_combo)
        self.mic_combo.currentIndexChanged.connect(self._mic_selected)
        mic_row.addWidget(self.mic_combo, 1)
        refresh = QPushButton(tr("refresh"))
        refresh.clicked.connect(lambda: self.refresh_microphones(True))
        mic_row.addWidget(refresh)
        layout.addLayout(mic_row)
        layout.addStretch()
        self.start_button = QPushButton(tr("start"))
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.c.start_visit)
        layout.addWidget(self.start_button)
        self.cloud_label = QLabel()
        self.cloud_label.setObjectName("cloud")
        self.cloud_label.setWordWrap(True)
        layout.addWidget(self.cloud_label)
        dev_row = QHBoxLayout()
        self.dev_wav = QPushButton(tr("dev_open_wav"))
        self.dev_wav.clicked.connect(self._dev_open_wav)
        self.dev_paste = QPushButton(tr("dev_paste"))
        self.dev_paste.clicked.connect(self._dev_paste)
        dev_row.addWidget(self.dev_wav)
        dev_row.addWidget(self.dev_paste)
        layout.addLayout(dev_row)
        bottom = QHBoxLayout()
        self.hotkey_hint = QLabel(tr("hotkeys_hint"))
        self.hotkey_hint.setStyleSheet("color: gray;")
        self.hotkey_hint.setWordWrap(True)
        bottom.addWidget(self.hotkey_hint, 1)
        settings = QPushButton(tr("settings"))
        settings.clicked.connect(self.open_settings)
        bottom.addWidget(settings)
        layout.addLayout(bottom)
        return page

    def _build_recording(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.recording_label = QLabel(tr("recording"))
        self.recording_label.setObjectName("recording")
        layout.addWidget(self.recording_label)
        self.timer_label = QLabel("00:00")
        self.timer_label.setObjectName("timer")
        layout.addWidget(self.timer_label)
        mic = QHBoxLayout()
        mic.addWidget(QLabel("Микрофон"))
        self.meter = LevelMeter()
        mic.addWidget(self.meter, 1)
        layout.addLayout(mic)
        self.segments_label = QLabel(tr("segments", n=0))
        layout.addWidget(self.segments_label)
        self.low_label = QLabel(tr("low_level"))
        self.low_label.setObjectName("warn")
        self.low_label.setWordWrap(True)
        self.low_label.setVisible(False)
        layout.addWidget(self.low_label)
        self.lag_label = QLabel(tr("lagging"))
        self.lag_label.setObjectName("warn")
        self.lag_label.setWordWrap(True)
        self.lag_label.setVisible(False)
        layout.addWidget(self.lag_label)
        layout.addStretch()
        self.stop_button = QPushButton(tr("stop"))
        self.stop_button.setObjectName("primary")
        self.stop_button.clicked.connect(self.c.stop_visit)
        layout.addWidget(self.stop_button)
        cancel = QPushButton(tr("cancel"))
        cancel.clicked.connect(self._confirm_cancel_recording)
        layout.addWidget(cancel)
        return page

    def _build_processing(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch()
        self.stage_label = QLabel(tr("finalizing"))
        self.stage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stage_label.setStyleSheet("font-size: 13pt;")
        layout.addWidget(self.stage_label)
        self.stage_progress = QProgressBar()
        self.stage_progress.setRange(0, 0)
        layout.addWidget(self.stage_progress)
        layout.addStretch()
        self.cancel_processing = QPushButton(tr("cancel"))
        self.cancel_processing.clicked.connect(self._cancel_processing)
        layout.addWidget(self.cancel_processing)
        return page

    def _build_review(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        banner = QLabel(tr("draft_banner"))
        banner.setObjectName("banner")
        banner.setWordWrap(True)
        layout.addWidget(banner)
        self.review_error = QLabel()
        self.review_error.setObjectName("warn")
        self.review_error.setWordWrap(True)
        self.review_error.setVisible(False)
        layout.addWidget(self.review_error)
        self.complaints = SectionEditor(
            tr("complaints"), tr("placeholder_complaints"), self.copy_complaints, self._show_source
        )
        self.history = SectionEditor(
            tr("history"), tr("placeholder_history"), self.copy_history, self._show_source
        )
        self.complaints.edit.setMinimumHeight(70)
        layout.addWidget(self.complaints, 1)
        layout.addWidget(self.history, 2)
        self.warnings_title = QLabel(tr("check"))
        self.warnings_title.setObjectName("section")
        layout.addWidget(self.warnings_title)
        self.warnings_label = QLabel()
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.warnings_label)
        row = QHBoxLayout()
        self.copy_all_button = QPushButton(tr("copy_all"))
        self.copy_all_button.clicked.connect(self.copy_all)
        row.addWidget(self.copy_all_button)
        self.transcript_button = QPushButton(tr("show_transcript"))
        self.transcript_button.clicked.connect(self.show_transcript)
        row.addWidget(self.transcript_button)
        layout.addLayout(row)
        row2 = QHBoxLayout()
        self.retry_button = QPushButton(tr("retry_analysis"))
        self.retry_button.clicked.connect(self.c.retry_analysis)
        row2.addWidget(self.retry_button)
        self.copy_transcript_button = QPushButton(tr("copy_transcript"))
        self.copy_transcript_button.clicked.connect(self.copy_transcript)
        row2.addWidget(self.copy_transcript_button)
        new_visit = QPushButton(tr("new_visit"))
        new_visit.clicked.connect(self.request_new_visit)
        row2.addWidget(new_visit)
        layout.addLayout(row2)
        return page

    def _build_tray(self) -> QSystemTrayIcon | None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return None
        tray = QSystemTrayIcon(dot_icon("#2e7d32"), self)
        menu = QMenu()
        show = QAction(tr("tray_show"), menu)
        show.triggered.connect(self.bring_to_front)
        start = QAction(tr("tray_start"), menu)
        # Starting from the tray always shows the window: no hidden recording.
        start.triggered.connect(self._tray_start)
        quit_action = QAction(tr("tray_quit"), menu)
        quit_action.triggered.connect(self.quit)
        menu.addActions([show, start, quit_action])
        tray.setContextMenu(menu)
        tray.setToolTip(tr("app_title"))
        tray.activated.connect(
            lambda reason: (
                self.bring_to_front()
                if reason == QSystemTrayIcon.ActivationReason.Trigger
                else None
            )
        )
        tray.show()
        self._tray_menu = menu
        return tray

    def _tray_start(self) -> None:
        self.bring_to_front()
        self.on_start_stop_hotkey()

    # ---- settings / devices ------------------------------------------------------------
    def apply_settings(self) -> None:
        s = self.c.settings
        flags = self.windowFlags()
        on_top = bool(flags & Qt.WindowType.WindowStaysOnTopHint)
        if on_top != s.general.always_on_top:
            visible = self.isVisible()
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, s.general.always_on_top)
            if visible:
                self.show()
        dev = s.developer.enabled
        self.dev_wav.setVisible(dev)
        self.dev_paste.setVisible(dev)
        llm = self.c.llm()
        if llm is None:
            self.cloud_label.setText(tr("ai_off"))
            self._status_ai.setText(tr("ai_not_ready"))
        else:
            self.cloud_label.setText(tr("cloud_on") if llm.is_cloud else tr("cloud_local"))
            self._status_ai.setText(tr("ai_ready") + (" ☁" if llm.is_cloud else ""))
        hk = s.hotkeys
        self.hotkeys.register(
            {
                hk.start_stop: self.on_start_stop_hotkey,
                hk.copy_complaints: self.copy_complaints,
                hk.copy_history: self.copy_history,
                hk.new_visit: self._new_visit_hotkey,
            }
        )
        self.hotkey_hint.setText(
            f"{hk.start_stop} — начать/завершить, {hk.copy_complaints}/{hk.copy_history.split('+')[-1]} — копировать"
        )
        self.refresh_microphones(False)

    def refresh_microphones(self, refresh: bool) -> None:
        devices = list_input_devices(refresh=refresh and not self.c.capture.running)
        self.mic_combo.blockSignals(True)
        self.mic_combo.clear()
        self.mic_combo.addItem(tr("mic_default"), None)
        selected = 0
        for dev in devices:
            self.mic_combo.addItem(dev.label, (dev.name, dev.hostapi))
            if dev.name == self.c.settings.audio.device_name:
                selected = self.mic_combo.count() - 1
        self.mic_combo.setCurrentIndex(selected)
        self.mic_combo.blockSignals(False)

    def _mic_selected(self, index: int) -> None:
        data = self.mic_combo.itemData(index)
        settings = self.c.settings.model_copy(deep=True)
        settings.audio.device_name = data[0] if data else None
        settings.audio.device_hostapi = data[1] if data else None
        self.c.save_settings(settings)

    # ---- controller events -------------------------------------------------------------
    def _on_state(self, _prev: str, current: str) -> None:
        state = AppState(current)
        idle_like = state in (
            AppState.INITIALIZING,
            AppState.MODEL_LOADING,
            AppState.IDLE,
            AppState.ERROR,
        )
        if idle_like:
            self.stack.setCurrentWidget(self.idle_page)
        elif state == AppState.RECORDING:
            self.stack.setCurrentWidget(self.record_page)
            self.meter.reset()
            self.low_label.setVisible(False)
            self.lag_label.setVisible(False)
            self._timer.start()
            self._tick()
        elif state in (AppState.STOPPING, AppState.FINALIZING_STT, AppState.ANALYZING):
            self.stack.setCurrentWidget(self.process_page)
            self.stage_label.setText(
                tr("analyzing") if state == AppState.ANALYZING else tr("finalizing")
            )
            self.stage_progress.setRange(0, 0)
        elif state == AppState.REVIEW:
            self.stack.setCurrentWidget(self.review_page)
        if state != AppState.RECORDING:
            self._timer.stop()
        self.start_button.setEnabled(state == AppState.IDLE)
        self.dev_wav.setEnabled(state == AppState.IDLE)
        self.dev_paste.setEnabled(state == AppState.IDLE)
        self.mic_combo.setEnabled(
            state in (AppState.IDLE, AppState.MODEL_LOADING, AppState.INITIALIZING)
        )
        if self.tray is not None:
            recording = state == AppState.RECORDING
            self.tray.setIcon(dot_icon("#d0342c" if recording else "#2e7d32"))
            self.tray.setToolTip(tr("recording") if recording else tr("app_title"))
        self.setWindowIcon(dot_icon("#d0342c" if state == AppState.RECORDING else "#2e7d32"))

    def _on_model_progress(self, stage: str, fraction: float) -> None:
        bar = self.model_progress_bar
        self.model_retry.setVisible(False)
        if stage == "download":
            bar.setVisible(True)
            bar.setRange(0, 100)
            bar.setValue(int(fraction * 100))
            text = (
                tr("model_download", pct=int(fraction * 100))
                if fraction > 0
                else tr("model_download_start")
            )
        else:
            bar.setVisible(True)
            bar.setRange(0, 0)
            text = tr("model_loading")
        self.model_label.setText(text)
        self._status_model.setText(text)

    def _on_model_ready(self, device: str) -> None:
        self.model_progress_bar.setVisible(False)
        self.model_retry.setVisible(False)
        text = tr("model_ready_device", device=device.upper())
        self.model_label.setText(text)
        self._status_model.setText(text)

    def _on_error(self, code: str, _detail: str) -> None:
        message = error_text(code)
        if self.c.state == AppState.ERROR:
            self.model_label.setText(f"{tr('model_error')}: {message}")
            self._status_model.setText(tr("model_error"))
            self.model_progress_bar.setVisible(False)
            self.model_retry.setVisible(True)
        else:
            QMessageBox.warning(self, tr("app_title"), message)

    def _on_notice(self, code: str, _detail: str) -> None:
        self.toast.show_message(error_text(code), 4000)

    def _on_level(self, peak: float, low: bool) -> None:
        self.meter.set_level(peak)
        self.low_label.setVisible(low)

    def _on_finalize_progress(self, done: int, total: int) -> None:
        if self.c.state == AppState.FINALIZING_STT and total:
            self.stage_progress.setRange(0, total)
            self.stage_progress.setValue(done)
            self.stage_label.setText(tr("finalizing_n", done=done, total=total))

    def _on_stage(self, stage: str) -> None:
        self.stage_label.setText(tr("analyzing") if stage == "analyzing" else tr("finalizing"))
        self.stage_progress.setRange(0, 0)

    def _on_draft(self, draft: DocumentDraft) -> None:
        self._draft = draft
        self.review_error.setVisible(False)
        self.complaints.set_ai_text(draft.complaints_text)
        self.history.set_ai_text(draft.history_text)
        self._show_warnings(draft)
        self.retry_button.setVisible(False)
        self.copy_transcript_button.setVisible(False)
        self.complaints.copy_button.setFocus()

    def _on_analysis_failed(self, code: str) -> None:
        self.review_error.setText(error_text(code))
        self.review_error.setVisible(True)
        self.retry_button.setVisible(True)
        self.copy_transcript_button.setVisible(True)
        if self._draft is None:
            self.complaints.set_ai_text("")
            self.history.set_ai_text("")
            self.warnings_label.setText("")
            self.warnings_title.setVisible(False)

    def _show_warnings(self, draft: DocumentDraft) -> None:
        lines = [f"⚠ {w.message}" for w in draft.warnings]
        self.warnings_title.setVisible(bool(lines))
        self.warnings_label.setText("\n".join(lines))

    def _tick(self) -> None:
        visit = self.c.visit
        if visit is None:
            return
        seconds = int(visit.elapsed_s())
        hours, rem = divmod(seconds, 3600)
        text = (
            f"{hours:02d}:{rem // 60:02d}:{rem % 60:02d}"
            if hours
            else f"{rem // 60:02d}:{rem % 60:02d}"
        )
        self.timer_label.setText(text)

    # ---- actions -----------------------------------------------------------------------
    def on_start_stop_hotkey(self) -> None:
        state = self.c.state
        if state == AppState.REVIEW:
            if self.request_new_visit():
                self.c.start_visit()
            return
        self.c.toggle_recording()

    def _clipboard(self, text: str, message: str) -> None:
        if not text:
            self.toast.show_message(tr("nothing_to_copy"))
            return
        QGuiApplication.clipboard().setText(text)
        self.toast.show_message(message)

    def copy_complaints(self) -> None:
        self._clipboard(self.complaints.text(), tr("copied_complaints"))
        self.complaints.copied = True

    def copy_history(self) -> None:
        self._clipboard(self.history.text(), tr("copied_history"))
        self.history.copied = True

    def copy_all(self) -> None:
        if not self.complaints.text() and not self.history.text():
            self.toast.show_message(tr("nothing_to_copy"))
            return
        self._clipboard(format_all(self.complaints.text(), self.history.text()), tr("copied_all"))
        self.complaints.copied = self.history.copied = True

    def _transcript_text(self, with_time: bool = True) -> str:
        visit = self.c.visit
        if visit is None or visit.transcript is None:
            return ""
        return "\n".join(
            f"[{format_ts(s.start_ms)}] {s.text}" if with_time else s.text
            for s in visit.transcript.segments
            if s.text
        )

    def copy_transcript(self) -> None:
        self._clipboard(self._transcript_text(False), tr("copied_transcript"))

    def show_transcript(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("transcript_title"))
        dialog.resize(560, 480)
        layout = QVBoxLayout(dialog)
        view = QPlainTextEdit(self._transcript_text() or "—")
        view.setReadOnly(True)
        layout.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy = buttons.addButton(tr("copy_transcript"), QDialogButtonBox.ButtonRole.ActionRole)
        copy.clicked.connect(self.copy_transcript)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def _show_source(self, full_text: str, position: int, selected: str) -> None:
        draft = self._draft
        visit = self.c.visit
        message = tr("no_source")
        if draft is not None and visit is not None and visit.transcript is not None:
            target = selected.strip()
            if not target:
                start = max(full_text.rfind(".", 0, max(0, position - 1)) + 1, 0)
                end = full_text.find(".", position)
                target = full_text[start : end + 1 if end >= 0 else len(full_text)].strip()
            fact_ids: list[str] = []
            for item in draft.trace:
                if target and (target in item.sentence or item.sentence in target):
                    fact_ids.extend(item.fact_ids)
            facts = [f for f in draft.accepted_facts if f.id in fact_ids]
            by_id = visit.transcript.by_id()
            lines = []
            for fact in facts:
                for sid in fact.evidence_segment_ids:
                    seg = by_id.get(sid)
                    if seg is not None:
                        line = f"{format_ts(seg.start_ms)}: «{seg.text}»"
                        if line not in lines:
                            lines.append(line)
            if lines:
                message = f"«{target}»\n\n" + "\n".join(lines)
        QMessageBox.information(self, tr("source_title"), message)

    def has_unsaved_changes(self) -> bool:
        edited = self.complaints.modified_by_user or self.history.modified_by_user
        not_copied = not (self.complaints.copied and self.history.copied)
        return edited and not_copied

    def request_new_visit(self) -> bool:
        if self.c.state == AppState.RECORDING:
            return False
        if self.c.state == AppState.REVIEW and self.has_unsaved_changes():
            answer = QMessageBox.question(self, tr("new_visit"), tr("confirm_new_visit"))
            if answer != QMessageBox.StandardButton.Yes:
                return False
        self._draft = None
        self.complaints.clear()
        self.history.clear()
        self.warnings_label.clear()
        self.review_error.setVisible(False)
        self.c.new_visit()
        return True

    def _new_visit_hotkey(self) -> None:
        self.request_new_visit()

    def _confirm_cancel_recording(self) -> None:
        answer = QMessageBox.question(
            self, tr("cancel"), "Отменить приём? Распознанный текст будет удалён."
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.c.cancel_recording()

    def _cancel_processing(self) -> None:
        if self.c.state == AppState.ANALYZING:
            self.c.cancel_analysis()
        else:
            self.c.cancel_recording()

    def _dev_open_wav(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("dev_open_wav"), "", "Audio (*.wav *.flac *.ogg)"
        )
        if path:
            self.c.analyze_file(Path(path))

    def _dev_paste(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("dev_paste_title"))
        dialog.resize(560, 420)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(tr("dev_paste_hint")))
        edit = QPlainTextEdit()
        layout.addWidget(edit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        analyze = buttons.addButton(tr("analyze"), QDialogButtonBox.ButtonRole.AcceptRole)
        analyze.clicked.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted and edit.toPlainText().strip():
            self.c.analyze_text(edit.toPlainText())

    def open_settings(self) -> None:
        from mva.ui.settings_dialog import SettingsDialog

        SettingsDialog(self.c, self).exec()

    def bring_to_front(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    # ---- window lifecycle ------------------------------------------------------------
    def _restore_geometry(self) -> None:
        geometry = self.c.settings.general.window_geometry
        if geometry:
            self.restoreGeometry(QByteArray.fromBase64(geometry.encode("ascii")))
        else:
            self.resize(460, 640)

    def _save_geometry(self) -> None:
        settings = self.c.settings.model_copy(deep=True)
        settings.general.window_geometry = bytes(self.saveGeometry().toBase64().data()).decode(
            "ascii"
        )
        self.c.settings = settings
        self.c.settings_store.save(settings)

    def quit(self) -> None:
        if self.c.state == AppState.REVIEW and self.has_unsaved_changes():
            answer = QMessageBox.question(self, tr("tray_quit"), tr("confirm_new_visit"))
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._quitting = True
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._quitting and self.c.state == AppState.RECORDING:
            answer = QMessageBox.question(
                self, tr("tray_quit"), "Идёт приём. Завершить запись и выйти?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._save_geometry()
        self.hotkeys.unregister()
        self.c.shutdown()
        if self.tray is not None:
            self.tray.hide()
        event.accept()
        app = QGuiApplication.instance()
        if app is not None:
            app.quit()
