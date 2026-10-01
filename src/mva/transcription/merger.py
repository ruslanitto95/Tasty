"""Deterministic transcript assembly: ordering + removal of audio-overlap duplicates.

Deduplication is applied only where the segmenter actually duplicated audio
(``overlap_prev_ms > 0``), so a patient genuinely repeating a phrase in separate
utterances is never collapsed. The merger never rewrites medical content.
"""

from __future__ import annotations

import re
import threading

from mva.transcription.models import Transcript, TranscriptSegment

_WORD = re.compile(r"\S+")
_STRIP = re.compile(r"[^\wёЁ]+", re.UNICODE)


def _norm(word: str) -> str:
    return _STRIP.sub("", word.lower().replace("ё", "е"))


def dedupe_overlap(prev_text: str, text: str, max_words: int = 8) -> str:
    """Remove the longest prefix of ``text`` that repeats the suffix of ``prev_text``."""
    prev = [_norm(w) for w in _WORD.findall(prev_text)]
    words = _WORD.findall(text)
    cur = [_norm(w) for w in words]
    best = 0
    for k in range(min(max_words, len(prev), len(cur)), 0, -1):
        if prev[-k:] == cur[:k] and all(cur[:k]):
            # A single short word match is too weak evidence of duplication.
            if k == 1 and len(cur[0]) < 4:
                continue
            best = k
            break
    if not best:
        return text
    rest = words[best:]
    return " ".join(rest)


class TranscriptMerger:
    def __init__(self) -> None:
        self._segments: dict[int, TranscriptSegment] = {}
        self._lock = threading.Lock()

    def add(self, segment: TranscriptSegment) -> TranscriptSegment:
        with self._lock:
            text = segment.text.strip()
            if segment.overlap_prev_ms > 0:
                prev = self._segments.get(segment.seq - 1)
                if prev is not None:
                    text = dedupe_overlap(prev.text, text)
            stored = segment.model_copy(update={"text": text})
            self._segments[segment.seq] = stored
            return stored

    def transcript(self) -> Transcript:
        with self._lock:
            ordered = [self._segments[k] for k in sorted(self._segments)]
        return Transcript(segments=[s for s in ordered if s.text or s.failed])

    def count(self) -> int:
        with self._lock:
            return sum(1 for s in self._segments.values() if s.text)

    def clear(self) -> None:
        with self._lock:
            self._segments.clear()
