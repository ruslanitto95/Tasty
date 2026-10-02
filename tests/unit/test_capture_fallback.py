from __future__ import annotations

import numpy as np
import pytest

from mva.audio import capture
from mva.audio.capture import AudioCaptureService, InputDevice, MicrophoneUnavailable


class FakeStream:
    def __init__(self, callback):
        self.callback = callback
        self.active = True

    def start(self):
        self.callback(np.zeros((512, 1), np.float32), 512, None, None)

    def stop(self):
        self.active = False

    def close(self):
        pass


class FakeSD:
    def __init__(self, bad: set[tuple[int, int]]):
        self.bad = bad
        self.opened: list[tuple[int, int]] = []

    def query_devices(self):
        return [{}] * 12

    def InputStream(self, device, channels, samplerate, dtype, blocksize, callback):
        if (device, channels) in self.bad:
            raise RuntimeError("Error opening InputStream: Invalid device [PaErrorCode -9996]")
        self.opened.append((device, channels))
        return FakeStream(callback)


WASAPI = InputDevice(
    9, "CABLE Output (VB-Audio Virtual Cable)", "Windows WASAPI", 48000.0, 2, False
)
FRESH = InputDevice(7, "CABLE Output (VB-Audio Virtual Cable)", "Windows WASAPI", 48000.0, 2, False)
MME = InputDevice(1, "CABLE Output (VB-Audio Virtual ", "MME", 44100.0, 8, False)


def start(monkeypatch, sd, candidates):
    monkeypatch.setattr(capture, "_sd", lambda: sd)
    monkeypatch.setattr(capture, "open_candidates", lambda _d: list(candidates))
    service = AudioCaptureService()
    used = service.start(WASAPI, lambda _b: None)
    service.stop()
    return used


def test_stale_index_reopens_by_name(monkeypatch):
    sd = FakeSD(bad={(9, 1), (9, 2)})
    assert start(monkeypatch, sd, [FRESH, MME]) == FRESH
    assert sd.opened == [(7, 1)]


def test_mono_rejected_uses_stereo_downmix(monkeypatch):
    sd = FakeSD(bad={(9, 1)})
    assert start(monkeypatch, sd, []) == WASAPI
    assert sd.opened == [(9, 2)]


def test_falls_back_to_other_host_api(monkeypatch):
    sd = FakeSD(bad={(9, 1), (9, 2), (7, 1), (7, 2)})
    assert start(monkeypatch, sd, [FRESH, MME]).hostapi == "MME"


def test_all_fail_raises_clean_error(monkeypatch):
    sd = FakeSD(bad={(9, 1), (9, 2), (1, 1), (1, 2)})
    monkeypatch.setattr(capture, "_sd", lambda: sd)
    monkeypatch.setattr(capture, "open_candidates", lambda _d: [MME])
    with pytest.raises(MicrophoneUnavailable):
        AudioCaptureService().start(WASAPI, lambda _b: None)


def test_same_endpoint_matches_truncated_mme_names():
    assert capture._same_endpoint(WASAPI.name, MME.name)
    assert not capture._same_endpoint(WASAPI.name, "Microphone (Realtek)")


def test_blocked_privacy_setting_reports_permission(monkeypatch):
    from mva.audio.capture import MicrophonePermissionDenied

    sd = FakeSD(bad={(9, 1), (9, 2)})
    monkeypatch.setattr(capture, "_sd", lambda: sd)
    monkeypatch.setattr(capture, "open_candidates", lambda _d: [])
    monkeypatch.setattr(capture, "windows_microphone_blocked", lambda: True)
    with pytest.raises(MicrophonePermissionDenied):
        AudioCaptureService().start(WASAPI, lambda _b: None)
