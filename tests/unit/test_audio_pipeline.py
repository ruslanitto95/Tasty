from __future__ import annotations

import numpy as np
import pytest

from mva.audio.buffers import RingBuffer
from mva.audio.levels import LevelMeter, normalize_segment
from mva.audio.resample import StreamResampler, resample
from mva.audio.segmenter import SpeechSegmenter
from mva.audio.vad import VADEvent, VoiceActivityDetector
from mva.config import SegmentationConfig
from tests.helpers import ScriptedProb

F = 512


def frames(n: int, value: float = 0.1) -> np.ndarray:
    return np.full(n * F, value, dtype=np.float32)


def run(probs: list[float], config: SegmentationConfig | None = None, **kw):
    segs = []
    vad = VoiceActivityDetector(
        ScriptedProb(probs), threshold=0.5, neg_threshold=0.35, min_silence_ms=320
    )
    seg = SpeechSegmenter(vad, segs.append, config=config or SegmentationConfig(), **kw)
    audio = (
        np.arange(len(probs) * F, dtype=np.float32) / 1e6
    )  # unique sample values for alignment checks
    for i in range(0, len(audio), 1000):
        seg.feed(audio[i : i + 1000])
    seg.flush()
    return segs, audio


def test_ring_buffer_wraps():
    rb = RingBuffer(5)
    rb.write(np.arange(3, dtype=np.float32))
    rb.write(np.arange(3, 7, dtype=np.float32))
    assert rb.read_all().tolist() == [2, 3, 4, 5, 6]
    rb.clear()
    assert len(rb) == 0


def test_vad_hysteresis_events():
    vad = VoiceActivityDetector(ScriptedProb([0.1, 0.9, 0.4, 0.2, 0.2]), min_silence_ms=64)
    events = [vad.process(np.zeros(F, np.float32)) for _ in range(5)]
    assert events == [
        VADEvent.SILENCE,
        VADEvent.SPEECH_STARTED,
        VADEvent.SPEECH_CONTINUES,
        VADEvent.SPEECH_CONTINUES,
        VADEvent.SPEECH_ENDED,
    ]


def test_segment_with_pre_and_post_roll():
    probs = [0.0] * 20 + [0.9] * 30 + [0.0] * 30
    segs, audio = run(probs, pre_roll_ms=128, post_roll_ms=96)
    assert len(segs) == 1
    s = segs[0]
    pre = int(128 * 16)  # 128 ms -> 2048 samples (4 frames)
    assert s.samples[0] == audio[20 * F - pre]
    # speech end = frame 50; post-roll 96 ms = 1536 samples
    assert len(s.samples) == pre + 30 * F + 96 * 16
    assert s.start_ms == (20 * F - pre) * 1000 // 16000
    assert s.overlap_prev_ms == 0


def test_short_noise_is_ignored():
    segs, _ = run([0.0] * 10 + [0.9] * 3 + [0.0] * 30, pre_roll_ms=0, min_speech_ms=250)
    assert segs == []


def test_long_speech_split_at_pause_without_overlap():
    cfg = SegmentationConfig(
        target_max_s=3.0, hard_max_s=6.0, split_search_window_s=2.0, min_segment_ms=100
    )
    speech = [0.9] * 80 + [0.3] + [0.9] * 80 + [0.0] * 30  # one short dip (no VAD end)
    segs, audio = run(speech, config=cfg, pre_roll_ms=0, post_roll_ms=0)
    assert len(segs) >= 2
    assert all(len(s.samples) <= 6.0 * 16000 for s in segs)
    joined = np.concatenate([s.samples for s in segs])
    # Non-forced split: audio is contiguous, nothing duplicated or lost.
    assert segs[1].overlap_prev_ms == 0 and not segs[1].forced_split
    assert np.array_equal(joined[: 161 * F], audio[: 161 * F])


def test_forced_split_adds_overlap_and_respects_hard_max():
    cfg = SegmentationConfig(
        target_max_s=2.0,
        hard_max_s=4.0,
        split_search_window_s=1.0,
        forced_split_overlap_ms=200,
        min_segment_ms=100,
    )
    segs, _ = run([0.9] * 400 + [0.0] * 20, config=cfg, pre_roll_ms=0, post_roll_ms=0)
    assert len(segs) >= 3
    assert all(len(s.samples) <= 4.0 * 16000 + F for s in segs)
    assert segs[1].forced_split and segs[1].overlap_prev_ms == 200
    assert segs[1].start_ms < segs[0].end_ms  # overlapping audio


def test_provider_limit_caps_segments():
    cfg = SegmentationConfig(target_max_s=30, hard_max_s=30)
    segs, _ = run([0.9] * 1000 + [0.0] * 20, config=cfg, pre_roll_ms=0, max_segment_s=10.0)
    assert all(len(s.samples) <= 9.0 * 16000 + F for s in segs)


def test_level_meter_low_level_and_clipping():
    meter = LevelMeter()
    reading = None
    for i in range(100):
        reading = meter.update(np.full(160, 0.001, np.float32), now=i * 0.1)
    assert reading is not None and reading.low_level
    loud = meter.update(np.ones(160, np.float32), now=10.1)
    assert loud.clipped and not loud.low_level


def test_normalize_segment_bounds():
    quiet = (np.sin(np.linspace(0, 100, 16000)) * 0.05).astype(np.float32)
    out = normalize_segment(quiet)
    assert 0.45 < np.max(np.abs(out)) <= 0.51  # gain capped at +20 dB
    silent = np.zeros(1000, np.float32)
    assert np.max(np.abs(normalize_segment(silent))) == 0
    hot = np.full(1000, 3.0, np.float32)
    assert np.max(np.abs(normalize_segment(hot))) <= 1.0


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000])
def test_resample_to_16k(rate: int):
    x = np.random.default_rng(0).standard_normal(rate).astype(np.float32) * 0.1
    assert abs(len(resample(x, rate)) - 16000) <= 2
    stream = StreamResampler(rate)
    out = np.concatenate(
        [stream.process(x[i : i + rate // 10]) for i in range(0, rate, rate // 10)]
        + [stream.process(np.zeros(0, np.float32), last=True)]
    )
    assert abs(len(out) - 16000) <= 50
    stereo = np.stack([x, x], axis=1)
    assert resample(stereo, rate).ndim == 1
