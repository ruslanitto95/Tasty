"""Small reusable widgets."""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QLabel, QWidget


class LevelMeter(QWidget):
    """Thin microphone activity meter (no waveform animation)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(8)
        self.setMinimumWidth(120)
        self._level = 0.0
        self.setAccessibleName("Уровень сигнала микрофона")

    def set_level(self, peak: float) -> None:
        # Smooth decay so the meter is calm.
        self._level = max(min(1.0, peak), self._level * 0.7)
        self.update()

    def reset(self) -> None:
        self._level = 0.0
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor(128, 128, 128, 60))
        width = int(rect.width() * min(1.0, self._level**0.5))
        colour = QColor(76, 175, 80) if self._level < 0.9 else QColor(230, 126, 34)
        painter.fillRect(QRect(0, 0, width, rect.height()), colour)
        painter.end()


class Toast(QLabel):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            "background: rgba(40,40,40,220); color: white; padding: 8px 14px; border-radius: 6px;"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, text: str, ms: int = 1800) -> None:
        self.setText(text)
        self.adjustSize()
        parent = self.parentWidget()
        if parent is not None:
            self.move((parent.width() - self.width()) // 2, parent.height() - self.height() - 24)
        self.show()
        self.raise_()
        self._timer.start(ms)
