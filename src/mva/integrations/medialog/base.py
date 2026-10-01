"""Medical-record integration boundary.

MVP uses the clipboard. A future MediLog adapter (UI Automation / official API — never
mouse coordinates) implements the same interface and may only *fill* fields; it must
never press «Подписать», «Закрыть приём» or «Сохранить окончательно».
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum


class RecordSection(StrEnum):
    COMPLAINTS = "complaints"
    HISTORY = "history"


class MedicalRecordIntegration(ABC):
    name: str = "integration"

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def put(self, section: RecordSection | None, text: str) -> None:
        """Deliver text for one section (None = whole document)."""


def format_all(complaints: str, history: str) -> str:
    return f"Жалобы: {complaints.strip()}\nАнамнез заболевания: {history.strip()}"
