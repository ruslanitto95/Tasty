"""Real capture through PortAudio from an input device.

Developer machines: any microphone (speak the dialogue). CI/sandbox: a PulseAudio null
sink monitor is the default source and the test plays the WAV into it with `paplay`.
"""

from __future__ import annotations

import shutil
import subprocess
import time

import pytest

from mva.audio.capture import AudioCaptureService, resolve_device
from mva.diagnostics.self_test import keyword_hits
from mva.paths import self_test_wav
from mva.transcription.session import SessionTranscriber

pytestmark = [pytest.mark.model, pytest.mark.audio_device]


def _virtual_sink() -> bool:
    if not shutil.which("pactl") or not shutil.which("paplay"):
        return False
    out = subprocess.run(["pactl", "get-default-source"], capture_output=True, text=True, check=False)
    return out.returncode == 0 and out.stdout.strip().endswith(".monitor")


@pytest.fixture
def device():
    dev = resolve_device(None, None)
    if dev is None:
        pytest.skip("no input device")
    if not _virtual_sink():
        pytest.skip("no virtual loopback source; run manually with a real microphone")
    return dev


def test_microphone_capture_is_transcribed(gigaam, device):
    session = SessionTranscriber(gigaam, None)
    session.start()
    capture = AudioCaptureService()
    peaks = []
    errors = []
    capture.start(device, session.feed, lambda r: peaks.append(r.peak), errors.append)
    time.sleep(0.5)
    player = subprocess.Popen(["paplay", str(self_test_wav())])
    player.wait(timeout=60)
    time.sleep(1.5)
    capture.stop()
    transcript = session.finish()
    assert not errors
    assert max(peaks) > 0.05
    assert len(transcript.segments) >= 5
    assert len(keyword_hits(transcript.text())) >= 4, transcript.text()
    assert not capture.running


def test_device_released_after_stop(device):
    capture = AudioCaptureService()
    for _ in range(3):
        capture.start(device, lambda _b: None)
        time.sleep(0.3)
        capture.stop()
    assert not capture.running
