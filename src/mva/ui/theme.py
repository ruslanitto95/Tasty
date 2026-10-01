"""Light / dark / system palettes on the Fusion style; calm colours, readable fonts."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

from mva.storage.app_settings import Theme


def _dark_palette() -> QPalette:
    p = QPalette()
    base, alt, text = QColor(32, 34, 37), QColor(43, 45, 49), QColor(230, 230, 230)
    p.setColor(QPalette.ColorRole.Window, alt)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, alt)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, QColor(55, 58, 63))
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.Highlight, QColor(64, 128, 200))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.ToolTipBase, base)
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(150, 150, 150))
    return p


def is_dark(theme: Theme) -> bool:
    if theme == Theme.DARK:
        return True
    if theme == Theme.LIGHT:
        return False
    return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark


def apply_theme(app: QApplication, theme: Theme) -> None:
    app.setStyle("Fusion")
    font = QFont(app.font())
    font.setPointSize(max(font.pointSize(), 11))
    app.setFont(font)
    app.setPalette(_dark_palette() if is_dark(theme) else app.style().standardPalette())
    app.setStyleSheet(
        """
        QPushButton { padding: 6px 12px; }
        QPushButton#primary { font-size: 15pt; padding: 12px; font-weight: 600; }
        QPushButton:focus, QTextEdit:focus, QComboBox:focus, QLineEdit:focus { outline: 2px solid #4a90d9; }
        QLabel#section { font-weight: 700; letter-spacing: 1px; margin-top: 6px; }
        QLabel#recording { color: #d0342c; font-size: 15pt; font-weight: 700; }
        QLabel#timer { font-size: 22pt; font-family: monospace; }
        QLabel#banner { background: #fff4ce; color: #5c4400; padding: 6px; border-radius: 4px; }
        QLabel#cloud { color: #8a5a00; }
        QLabel#warn { color: #8a5a00; }
        """
    )
