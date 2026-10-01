"""«Запускать вместе с Windows» via the per-user Run registry key."""

from __future__ import annotations

import contextlib
import sys

_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_NAME = "MedicalVisitAssistant"


def set_autostart(enabled: bool) -> bool:
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return False
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _KEY, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            winreg.SetValueEx(key, _NAME, 0, winreg.REG_SZ, f'"{sys.executable}"')
        else:
            with contextlib.suppress(FileNotFoundError):
                winreg.DeleteValue(key, _NAME)
    return True
