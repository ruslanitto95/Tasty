"""WAV -> VAD -> GigaAM -> transcript -> clinical pipeline, with the real model."""

from __future__ import annotations

import numpy as np
import pytest

from mva.audio.wav import read_audio_16k, write_wav_16k
from mva.clinical.pipeline import ClinicalPipeline
from mva.diagnostics.self_test import keyword_hits, run_self_test
from mva.paths import self_test_wav
from mva.transcription.models import AudioSegment
from mva.transcription.session import transcribe_file
from tests.scripted_llm import scripted_llm

pytestmark = pytest.mark.model


def test_self_test_wav_transcribed_through_pipeline(gigaam):
    transcript, stats = transcribe_file(gigaam, self_test_wav())
    assert len(transcript.segments) == 6
    assert len(keyword_hits(transcript.text())) >= 4
    assert "справа" in transcript.text().lower()
    assert stats.segments_failed == 0
    assert stats.rtf < 1.0


def test_wav_to_clinical_draft(gigaam):
    transcript, _ = transcribe_file(gigaam, self_test_wav())
    draft = ClinicalPipeline(scripted_llm(), use_llm_formatter=False).run(transcript)
    assert draft.complaints_text == "Затруднение носового дыхания, преимущественно справа."
    assert "около 7 дней" in draft.history_text
    assert "37,5" in draft.history_text
    assert "Називин" not in draft.history_text
    for fact in draft.accepted_facts:
        assert fact.evidence_segment_ids and fact.evidence_quote


def test_long_visit_is_processed_in_segments(gigaam, tmp_path):
    audio = read_audio_16k(self_test_wav())
    pause = np.zeros(16000 * 2, dtype=np.float32)
    long_audio = np.concatenate([np.concatenate([audio, pause]) for _ in range(6)])  # ~2 min
    path = tmp_path / "long.wav"
    write_wav_16k(path, long_audio)
    transcript, stats = transcribe_file(gigaam, path)
    assert len(transcript.segments) == 36
    assert all(s.end_ms - s.start_ms <= 22_500 for s in transcript.segments)
    assert [s.seq for s in transcript.segments] == sorted(s.seq for s in transcript.segments)
    assert transcript.text().lower().count("справа") == 6
    assert stats.segments_failed == 0


def test_monologue_without_pauses_is_split_safely(gigaam, tmp_path):
    audio = read_audio_16k(self_test_wav())
    # Remove pauses: keep only voiced regions to build ~35 s of continuous speech.
    voiced = audio[np.convolve(np.abs(audio) > 0.01, np.ones(800), "same") > 0]
    mono = np.concatenate([voiced, voiced, voiced])
    path = tmp_path / "mono.wav"
    write_wav_16k(path, mono)
    transcript, _ = transcribe_file(gigaam, path)
    assert len(transcript.segments) >= 2
    assert all(s.end_ms - s.start_ms <= 22_500 for s in transcript.segments)
    assert transcript.text().lower().count("справа") >= 2


def test_too_long_segment_is_refused(gigaam):
    from mva.transcription.base import TranscriptionError

    with pytest.raises(TranscriptionError):
        gigaam.transcribe(
            AudioSegment(seq=1, start_ms=0, end_ms=30000, samples=np.zeros(16000 * 30, np.float32))
        )


def test_self_test_report(gigaam):
    report = run_self_test(gigaam._manager, gigaam)
    assert report.ok
    assert [s.name for s in report.steps] == ["model_download", "model_load", "wav_transcription"]


def test_broken_checkpoint_load_failure(tmp_path):
    """A corrupted model file must surface as a clean error, never as a crash or valid model."""
    import shutil

    from mva.transcription.base import ModelLoadError
    from mva.transcription.gigaam_provider import GigaAMTranscriptionProvider
    from mva.transcription.model_manager import ModelManager, ModelStatus, load_manifest
    from tests.helpers import models_dir_for_tests

    src = models_dir_for_tests()
    manifest = load_manifest()
    for f in manifest.files:
        shutil.copy(src / f.name, tmp_path / f.name)
    manager = ModelManager(manifest, tmp_path)
    assert manager.status() != ModelStatus.READY  # no verification marker yet
    assert manager.verify_full()
    with (tmp_path / manifest.files[0].name).open("r+b") as handle:
        handle.seek(1000)
        handle.write(b"\x00" * 64)
    assert manager.status() == ModelStatus.CORRUPT
    with pytest.raises(ModelLoadError):
        GigaAMTranscriptionProvider(manager).load()
