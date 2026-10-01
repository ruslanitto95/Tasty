from __future__ import annotations

import numpy as np

from mva.storage.temp_sessions import TempSessionManager
from mva.transcription.fake import FakeTranscriptionProvider
from mva.transcription.merger import TranscriptMerger, dedupe_overlap
from mva.transcription.models import AudioSegment, TranscriptSegment
from mva.transcription.queue import SegmentQueue
from mva.transcription.worker import STTWorker


def seg(seq: int, text: str, overlap: int = 0) -> TranscriptSegment:
    return TranscriptSegment(
        id=f"s{seq:04d}",
        seq=seq,
        start_ms=seq * 1000,
        end_ms=seq * 1000 + 900,
        text=text,
        overlap_prev_ms=overlap,
    )


def test_overlap_dedup_removes_duplicated_words():
    assert (
        dedupe_overlap(
            "затем появилась заложенность носа", "заложенность носа стала сильнее справа"
        )
        == "стала сильнее справа"
    )
    assert (
        dedupe_overlap("болит горло.", "Горло болит три дня") == "болит три дня"
    )  # only called for real audio overlap
    assert dedupe_overlap("ухо болит", "болит справа") == "справа"
    assert dedupe_overlap("сказал да", "да конечно") == "да конечно"  # weak 1-word match is kept


def test_merger_dedups_only_real_audio_overlap():
    m = TranscriptMerger()
    m.add(seg(2, "заложенность носа стала сильнее справа", overlap=600))
    m.add(seg(1, "затем появилась заложенность носа"))
    # seq 2 arrived first (no predecessor yet) -> kept as is; ordering still by seq
    assert [s.seq for s in m.transcript().segments] == [1, 2]
    m2 = TranscriptMerger()
    m2.add(seg(1, "затем появилась заложенность носа"))
    m2.add(seg(2, "заложенность носа стала сильнее справа", overlap=600))
    assert m2.transcript().text() == "затем появилась заложенность носа стала сильнее справа"


def test_merger_keeps_genuine_repetition_without_overlap():
    m = TranscriptMerger()
    m.add(seg(1, "болит правое ухо"))
    m.add(seg(2, "правое ухо болит сильно"))
    assert m.transcript().text() == "болит правое ухо правое ухо болит сильно"


def test_queue_spills_to_disk_and_deletes_after_read(tmp_path):
    session = TempSessionManager(tmp_path).create()
    q = SegmentQueue(session, max_in_memory=2)
    for i in range(1, 6):
        q.put(
            AudioSegment(
                seq=i, start_ms=0, end_ms=1000, samples=np.full(16000, 0.1 * i, np.float32)
            )
        )
    assert q.spilled_total == 3
    assert len(list(session.path.glob("*.wav"))) == 3
    got = [q.get(0.1) for _ in range(5)]
    assert [g.seq for g in got] == [1, 2, 3, 4, 5]
    assert all(g.samples is not None and len(g.samples) == 16000 for g in got)
    assert abs(float(got[4].samples[0]) - 0.5) < 1e-3
    assert not list(session.path.glob("*.wav"))


def test_worker_preserves_order_and_flags_failures():
    provider = FakeTranscriptionProvider(texts=lambda s: f"t{s.seq}", fail_on={3})
    q = SegmentQueue(None)
    merger = TranscriptMerger()
    w = STTWorker(provider, q, merger)
    w.start()
    for i in range(1, 6):
        q.put(AudioSegment(seq=i, start_ms=i, end_ms=i + 1, samples=np.zeros(1600, np.float32)))
    assert w.finish(5)
    segments = merger.transcript().segments
    assert [s.seq for s in segments] == [1, 2, 3, 4, 5]
    assert segments[2].failed and segments[2].text == ""
    assert w.failed == 1 and w.processed == 5


def test_worker_cancel_discards_pending(tmp_path):
    session = TempSessionManager(tmp_path).create()
    provider = FakeTranscriptionProvider(delay_s=0.05)
    q = SegmentQueue(session, max_in_memory=1)
    w = STTWorker(provider, q, TranscriptMerger())
    for i in range(1, 20):
        q.put(AudioSegment(seq=i, start_ms=0, end_ms=1, samples=np.zeros(1600, np.float32)))
    w.start()
    w.cancel()
    assert not w.alive
    assert not list(session.path.glob("*.wav"))
