"""API key storage via the OS credential vault (Windows Credential Manager on Windows).

The key is never written to settings.json or any plaintext file. If no secure backend
exists (e.g. headless Linux CI) the key is kept only in memory for the current process.
"""

from __future__ import annotations

import logging

import keyring
import keyring.errors

log = logging.getLogger(__name__)

SERVICE = "MedicalVisitAssistant"
_ACCOUNT = "llm_api_key"
_INSECURE_BACKENDS = ("fail", "null", "plaintext", "encryptedfile")


class CredentialStore:
    def __init__(self, service: str = SERVICE) -> None:
        self._service = service
        self._memory: dict[str, str] = {}
        self.secure_backend_available = self._probe()

    @staticmethod
    def _probe() -> bool:
        backend = keyring.get_keyring()
        name = f"{type(backend).__module__}.{type(backend).__name__}".lower()
        if any(part in name for part in _INSECURE_BACKENDS):
            return False
        try:
            return float(getattr(backend, "priority", 1)) > 0
        except Exception:
            return True

    def get_api_key(self) -> str | None:
        if _ACCOUNT in self._memory:
            return self._memory[_ACCOUNT]
        if not self.secure_backend_available:
            return None
        try:
            return keyring.get_password(self._service, _ACCOUNT)
        except keyring.errors.KeyringError as exc:
            log.warning("Credential read failed: %s", type(exc).__name__)
            return None

    def set_api_key(self, value: str | None) -> bool:
        """Returns True if persisted in the secure vault, False if kept in memory only."""
        if not value:
            self.delete_api_key()
            return True
        if self.secure_backend_available:
            try:
                keyring.set_password(self._service, _ACCOUNT, value)
                self._memory.pop(_ACCOUNT, None)
                return True
            except keyring.errors.KeyringError as exc:
                log.warning("Credential write failed: %s", type(exc).__name__)
        self._memory[_ACCOUNT] = value
        return False

    def delete_api_key(self) -> None:
        self._memory.pop(_ACCOUNT, None)
        if not self.secure_backend_available:
            return
        try:
            keyring.delete_password(self._service, _ACCOUNT)
        except keyring.errors.PasswordDeleteError:
            pass
        except keyring.errors.KeyringError as exc:
            log.warning("Credential delete failed: %s", type(exc).__name__)
