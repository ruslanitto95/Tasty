"""Global hotkeys. Windows: RegisterHotKey on the main window (works while MediLog has
focus). Other platforms: application-wide Qt shortcuts (window must be focused)."""

from __future__ import annotations

import ctypes
import logging
import sys
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QAbstractNativeEventFilter
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QWidget

log = logging.getLogger(__name__)

WM_HOTKEY = 0x0312
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000


def parse_hotkey(sequence: str) -> tuple[int, int] | None:
    """'Ctrl+Alt+R' -> (modifiers, virtual key). Returns None if unsupported."""
    parts = [p.strip().lower() for p in sequence.split("+") if p.strip()]
    if not parts:
        return None
    mods = MOD_NOREPEAT
    key = parts[-1]
    for part in parts[:-1]:
        mods |= {
            "ctrl": MOD_CONTROL,
            "alt": MOD_ALT,
            "shift": MOD_SHIFT,
            "win": MOD_WIN,
            "meta": MOD_WIN,
        }.get(part, 0)
    if len(key) == 1 and (key.isalnum()):
        vk = ord(key.upper())
    elif key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        vk = 0x70 + int(key[1:]) - 1
    else:
        return None
    if mods == MOD_NOREPEAT:
        return None  # refuse bare keys as global hotkeys
    return mods, vk


class _WinFilter(QAbstractNativeEventFilter):
    def __init__(self, callbacks: dict[int, Callable[[], None]]) -> None:
        super().__init__()
        self.callbacks = callbacks

    def nativeEventFilter(self, event_type: Any, message: Any) -> tuple[bool, int]:
        if bytes(event_type) != b"windows_generic_MSG":
            return False, 0
        from ctypes import wintypes

        msg = wintypes.MSG.from_address(int(message))
        if msg.message == WM_HOTKEY and msg.wParam in self.callbacks:
            self.callbacks[msg.wParam]()
            return True, 0
        return False, 0


class HotkeyManager:
    def __init__(self, window: QWidget) -> None:
        self.window = window
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._shortcuts: list[QShortcut] = []
        self._filter: _WinFilter | None = None
        self.failed: list[str] = []
        self.global_mode = sys.platform == "win32"

    def register(self, bindings: dict[str, Callable[[], None]]) -> None:
        self.unregister()
        self.failed = []
        if self.global_mode:
            self._register_windows(bindings)
        else:
            for seq, callback in bindings.items():
                shortcut = QShortcut(QKeySequence(seq), self.window)
                shortcut.setContext(shortcut.context().ApplicationShortcut)
                shortcut.activated.connect(callback)
                self._shortcuts.append(shortcut)

    def _register_windows(self, bindings: dict[str, Callable[[], None]]) -> None:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        hwnd = int(self.window.winId())
        for index, (seq, callback) in enumerate(bindings.items(), start=1):
            parsed = parse_hotkey(seq)
            if parsed is None or not user32.RegisterHotKey(hwnd, index, parsed[0], parsed[1]):
                log.warning("Hotkey registration failed for binding #%d", index)
                self.failed.append(seq)
                continue
            self._callbacks[index] = callback
        if self._filter is None:
            self._filter = _WinFilter(self._callbacks)
            app = QApplication.instance()
            if app is not None:
                app.installNativeEventFilter(self._filter)
        else:
            self._filter.callbacks = self._callbacks

    def unregister(self) -> None:
        if self.global_mode and self._callbacks:
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            hwnd = int(self.window.winId())
            for index in list(self._callbacks):
                user32.UnregisterHotKey(hwnd, index)
        self._callbacks.clear()
        for shortcut in self._shortcuts:
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        self._shortcuts.clear()
