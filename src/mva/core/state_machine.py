"""Explicit application state machine (no ad-hoc boolean flags)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from enum import StrEnum

log = logging.getLogger(__name__)


class AppState(StrEnum):
    INITIALIZING = "initializing"
    MODEL_LOADING = "model_loading"
    IDLE = "idle"
    RECORDING = "recording"
    STOPPING = "stopping"
    FINALIZING_STT = "finalizing_stt"
    ANALYZING = "analyzing"
    REVIEW = "review"
    ERROR = "error"


S = AppState
TRANSITIONS: dict[AppState, set[AppState]] = {
    S.INITIALIZING: {S.MODEL_LOADING, S.IDLE, S.ERROR},
    S.MODEL_LOADING: {S.IDLE, S.ERROR},
    S.IDLE: {S.RECORDING, S.FINALIZING_STT, S.ANALYZING, S.MODEL_LOADING, S.ERROR},
    S.RECORDING: {S.STOPPING, S.IDLE, S.ERROR},
    S.STOPPING: {S.FINALIZING_STT, S.IDLE, S.ERROR},
    S.FINALIZING_STT: {S.ANALYZING, S.REVIEW, S.IDLE, S.ERROR},
    S.ANALYZING: {S.REVIEW, S.IDLE, S.ERROR},
    S.REVIEW: {S.ANALYZING, S.IDLE, S.ERROR},
    S.ERROR: {S.MODEL_LOADING, S.IDLE, S.REVIEW},
}


class InvalidTransition(Exception):
    pass


Listener = Callable[[AppState, AppState], None]


class StateMachine:
    def __init__(self, initial: AppState = AppState.INITIALIZING) -> None:
        self._state = initial
        self._lock = threading.RLock()
        self._listeners: list[Listener] = []

    @property
    def state(self) -> AppState:
        return self._state

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def can(self, target: AppState) -> bool:
        return target in TRANSITIONS[self._state]

    def transition(self, target: AppState) -> AppState:
        with self._lock:
            previous = self._state
            if target == previous:
                return previous
            if target not in TRANSITIONS[previous]:
                raise InvalidTransition(f"{previous} -> {target}")
            self._state = target
        log.info("State %s -> %s", previous.value, target.value)
        for listener in list(self._listeners):
            listener(previous, target)
        return previous

    def try_transition(self, expected: set[AppState], target: AppState) -> bool:
        """Atomic compare-and-set; used to make double clicks / hotkey repeats harmless."""
        with self._lock:
            if self._state not in expected or target not in TRANSITIONS[self._state]:
                return False
            self.transition(target)
            return True
