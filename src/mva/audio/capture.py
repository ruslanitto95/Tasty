"""Microphone capture via PortAudio (sounddevice).

The PortAudio callback only copies blocks into a queue; resampling, VAD and STT run on
other threads so neither the audio callback nor the UI thread is ever blocked.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from mva.audio.levels import LevelMeter, LevelReading
from mva.audio.resample import StreamResampler

log = logging.getLogger(__name__)


class MicrophoneError(Exception):
    code = "microphone_unavailable"


class MicrophoneUnavailable(MicrophoneError):
    code = "microphone_unavailable"


class MicrophonePermissionDenied(MicrophoneError):
    code = "microphone_permission"


class MicrophoneLost(MicrophoneError):
    code = "microphone_lost"


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    hostapi: str
    default_samplerate: float
    channels: int
    is_default: bool

    @property
    def label(self) -> str:
        return self.name


def _sd() -> Any:
    import sounddevice

    return sounddevice


_PREFERRED_HOSTAPIS = {
    "win32": ("Windows WASAPI", "MME", "Windows DirectSound"),
    "linux": ("PulseAudio", "ALSA"),
    "darwin": ("Core Audio",),
}


def refresh_portaudio() -> None:
    """PortAudio caches the device list at init; re-init to see hot-plugged devices."""
    sd = _sd()
    try:
        sd._terminate()
        sd._initialize()
    except Exception as exc:
        log.warning("PortAudio refresh failed: %s", type(exc).__name__)


def list_input_devices(refresh: bool = False) -> list[InputDevice]:
    try:
        sd = _sd()
    except OSError:
        log.warning("PortAudio library not available")
        return []
    if refresh:
        refresh_portaudio()
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_in = sd.default.device[0]
    except Exception as exc:
        log.warning("Device query failed: %s", type(exc).__name__)
        return []
    preferred = _PREFERRED_HOSTAPIS.get(sys.platform, ())
    available = {h["name"] for h in hostapis}
    chosen_api = next((name for name in preferred if name in available), None)
    result: list[InputDevice] = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        api_name = hostapis[dev["hostapi"]]["name"]
        if chosen_api and api_name != chosen_api:
            continue
        result.append(
            InputDevice(
                index=idx,
                name=str(dev["name"]),
                hostapi=api_name,
                default_samplerate=float(dev["default_samplerate"]),
                channels=int(dev["max_input_channels"]),
                is_default=idx == default_in,
            )
        )
    if chosen_api and not result:
        # Preferred host API exposes nothing (unusual drivers): fall back to all APIs.
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                result.append(
                    InputDevice(
                        idx,
                        str(dev["name"]),
                        hostapis[dev["hostapi"]]["name"],
                        float(dev["default_samplerate"]),
                        int(dev["max_input_channels"]),
                        idx == default_in,
                    )
                )
    return result


def resolve_device(name: str | None, hostapi: str | None) -> InputDevice | None:
    """Find the saved device by name; None/missing -> current system default."""
    devices = list_input_devices(refresh=True)
    if not devices:
        return None
    if name:
        for dev in devices:
            if dev.name == name and (hostapi is None or dev.hostapi == hostapi):
                return dev
        log.warning("Saved microphone not found; using system default")
    return next((d for d in devices if d.is_default), devices[0])


BlockCallback = Callable[[np.ndarray], None]
LevelCallback = Callable[[LevelReading], None]
ErrorCallback = Callable[[MicrophoneError], None]


class AudioCaptureService:
    """Opens one input stream; delivers 16 kHz mono float32 blocks on a worker thread."""

    WATCHDOG_S = 2.5
    NO_SIGNAL_S = 4.0

    def __init__(self) -> None:
        self._stream: Any = None
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=2000)
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._last_callback = 0.0
        self._lock = threading.Lock()
        self.device: InputDevice | None = None
        self.overflows = 0
        self.meter = LevelMeter()

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def start(
        self,
        device: InputDevice | None,
        on_block: BlockCallback,
        on_level: LevelCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> InputDevice:
        with self._lock:
            if self._running.is_set():
                raise RuntimeError("capture already running")
            if device is None:
                raise MicrophoneUnavailable("no input device")
            sd = _sd()
            rate = int(device.default_samplerate) or 48000
            resampler = StreamResampler(rate)
            self.meter.reset()
            self.overflows = 0
            self._queue = queue.Queue(maxsize=2000)

            def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
                self._last_callback = time.monotonic()
                if status and status.input_overflow:
                    self.overflows += 1
                try:
                    self._queue.put_nowait(indata[:, 0].copy())
                except queue.Full:
                    self.overflows += 1

            try:
                stream = sd.InputStream(
                    device=device.index,
                    channels=1,
                    samplerate=rate,
                    dtype="float32",
                    blocksize=int(rate * 0.032),
                    callback=callback,
                )
                stream.start()
            except Exception as exc:
                raise _map_portaudio_error(exc) from exc
            self._stream = stream
            self.device = device
            self._last_callback = time.monotonic()
            self._running.set()
            self._thread = threading.Thread(
                target=self._pump,
                args=(resampler, on_block, on_level, on_error),
                name="audio-pump",
                daemon=True,
            )
            self._thread.start()
            log.info("Capture started: hostapi=%s rate=%d", device.hostapi, rate)
            return device

    def _pump(
        self,
        resampler: StreamResampler,
        on_block: BlockCallback,
        on_level: LevelCallback | None,
        on_error: ErrorCallback | None,
    ) -> None:
        signal_seen = False
        started = time.monotonic()
        while self._running.is_set():
            try:
                block = self._queue.get(timeout=0.2)
            except queue.Empty:
                block = None
            now = time.monotonic()
            if block is None:
                stream = self._stream
                lost = stream is not None and not stream.active
                if lost or now - self._last_callback > self.WATCHDOG_S:
                    self._fail(on_error, MicrophoneLost("audio callbacks stopped"))
                    return
                continue
            try:
                out = resampler.process(block)
                reading = self.meter.update(block, now)
                if reading.peak > 1e-4:
                    signal_seen = True
                elif not signal_seen and now - started > self.NO_SIGNAL_S:
                    # Windows delivers digital zeros when mic access is blocked in Privacy settings.
                    self._fail(on_error, MicrophonePermissionDenied("digital silence"))
                    return
                if on_level is not None:
                    on_level(reading)
                if len(out):
                    on_block(out)
            except Exception as exc:
                log.exception("Audio pump failed")
                self._fail(on_error, MicrophoneError(type(exc).__name__))
                return
        # Drain resampler tail after stop.
        try:
            tail = resampler.process(np.zeros(0, dtype=np.float32), last=True)
            if len(tail):
                on_block(tail)
        except Exception:
            log.debug("Resampler flush failed", exc_info=True)

    def _fail(self, on_error: ErrorCallback | None, error: MicrophoneError) -> None:
        log.warning("Capture error: %s", error.code)
        self._running.clear()
        self._close_stream()
        if on_error is not None:
            on_error(error)

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            log.debug("stream.stop failed", exc_info=True)
        finally:
            try:
                stream.close()
            except Exception:
                log.debug("stream.close failed", exc_info=True)

    def stop(self) -> None:
        """Stops capture and releases the device; delivers all buffered audio first."""
        with self._lock:
            self._close_stream()
            thread = self._thread
            if thread is not None and thread.is_alive():
                # Let the pump consume what PortAudio already delivered.
                deadline = time.monotonic() + 2.0
                while not self._queue.empty() and time.monotonic() < deadline:
                    time.sleep(0.02)
            self._running.clear()
            if thread is not None:
                thread.join(3.0)
            self._thread = None
            log.info("Capture stopped (overflows=%d)", self.overflows)


def _map_portaudio_error(exc: Exception) -> MicrophoneError:
    text = str(exc).lower()
    if "permission" in text or "access" in text or ("-9999" in text and "unanticipated" in text):
        return MicrophonePermissionDenied(type(exc).__name__)
    return MicrophoneUnavailable(type(exc).__name__)


def record_seconds(device: InputDevice, seconds: float) -> np.ndarray:
    """Blocking helper for microphone tests (runs on a worker thread)."""
    chunks: list[np.ndarray] = []
    errors: list[MicrophoneError] = []
    service = AudioCaptureService()
    service.start(device, chunks.append, on_error=errors.append)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and service.running:
        time.sleep(0.05)
    service.stop()
    if errors:
        raise errors[0]
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
