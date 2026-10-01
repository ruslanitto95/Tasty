"""MediLog adapter placeholder: not available in the MVP and not exposed in the UI."""

from __future__ import annotations

from mva.integrations.medialog.base import MedicalRecordIntegration, RecordSection


class MediLogAdapter(MedicalRecordIntegration):
    name = "medialog"

    @property
    def available(self) -> bool:
        return False

    def put(self, section: RecordSection | None, text: str) -> None:
        raise NotImplementedError("MediLog AutoFill is planned for v0.2 (see ROADMAP.md)")
