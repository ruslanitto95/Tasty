from __future__ import annotations

import json
import logging
import os

import pytest

from mva.core.state_machine import AppState, InvalidTransition, StateMachine
from mva.security.credentials import CredentialStore
from mva.security.privacy import configure_logging, redact
from mva.storage.app_settings import AppSettings, SettingsStore
from mva.storage.temp_sessions import TempSessionManager
from mva.ui.hotkeys import MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, parse_hotkey


def test_state_machine_transitions_and_double_start():
    sm = StateMachine()
    sm.transition(AppState.MODEL_LOADING)
    sm.transition(AppState.IDLE)
    assert sm.try_transition({AppState.IDLE}, AppState.RECORDING)
    assert not sm.try_transition({AppState.IDLE}, AppState.RECORDING)  # second click ignored
    with pytest.raises(InvalidTransition):
        sm.transition(AppState.REVIEW)
    for target in (
        AppState.STOPPING,
        AppState.FINALIZING_STT,
        AppState.ANALYZING,
        AppState.REVIEW,
        AppState.IDLE,
    ):
        sm.transition(target)
    assert sm.state == AppState.IDLE


def test_redaction_covers_phi():
    text = redact(
        "transcript='Нос не дышит' phone +7 (912) 345-67-89 mail ivan@mail.ru key sk-abcdef123456"
    )
    assert (
        "Нос" not in text
        and "912" not in text
        and "ivan" not in text
        and "abcdef123456" not in text
    )
    assert redact("size=448929252 rtf=0.15") == "size=448929252 rtf=0.15"


def test_log_file_never_contains_cyrillic(tmp_path):
    configure_logging(tmp_path)
    log = logging.getLogger("phi-test")
    log.info("segment text %s", "Принимал Називин 3 дня")
    try:
        raise ValueError("Пациент Иванов")
    except ValueError:
        log.exception("failure")
    for handler in logging.getLogger().handlers:
        handler.flush()
    content = (tmp_path / "mva.log").read_text(encoding="utf-8")
    assert "segment text" in content and "Traceback" in content
    assert not any("\u0400" <= ch <= "\u04ff" for ch in content)
    configure_logging(None)


def test_settings_roundtrip_corrupt_and_migration(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    s = AppSettings()
    s.ai.model = "local-model"
    store.save(s)
    assert store.load().ai.model == "local-model"
    assert "api_key" not in (tmp_path / "settings.json").read_text()
    (tmp_path / "settings.json").write_text("{not json")
    assert store.load() == AppSettings()
    assert (tmp_path / "settings.corrupt.json").exists()
    (tmp_path / "settings.json").write_text(
        json.dumps({"ai": {"model": "x"}, "unknown_future_key": 1})
    )
    loaded = store.load()
    assert loaded.schema_version == 1 and loaded.ai.model == "x"


def test_defaults_are_privacy_preserving():
    s = AppSettings()
    assert s.privacy.delete_audio_after_transcription is True
    assert s.privacy.save_history is False
    assert s.developer.enabled is False and s.developer.save_test_audio is False
    assert s.ai.enabled is False
    assert s.speech.model_name == "v3_e2e_rnnt" and s.speech.device == "auto"


def test_temp_sessions_orphan_cleanup_and_traversal(tmp_path):
    mgr = TempSessionManager(tmp_path)
    mine = mgr.create()
    orphan = tmp_path / "deadbeef"
    orphan.mkdir()
    (orphan / ".owner_pid").write_text("999999")
    (orphan / "s0001.wav").write_bytes(b"x")
    assert mgr.cleanup_orphans() == 1
    assert mine.path.exists() and not orphan.exists()
    assert mine.file("../../etc/passwd").parent == mine.path
    with pytest.raises(ValueError):
        mine.file("..")
    mine.destroy()
    assert mgr.count() == 0
    assert all(len(p.name) == 32 for p in tmp_path.iterdir()) or not list(tmp_path.iterdir())
    assert str(os.getpid()) not in str(mine.path)


def test_credentials_use_vault_not_files(tmp_path):
    store = CredentialStore(service="mva-test")
    assert store.secure_backend_available
    assert store.set_api_key("sk-secret") is True
    assert store.get_api_key() == "sk-secret"
    store.delete_api_key()
    assert store.get_api_key() is None


def test_hotkey_parsing():
    assert parse_hotkey("Ctrl+Alt+R") == (MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, ord("R"))
    assert parse_hotkey("Ctrl+Alt+1") == (MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, ord("1"))
    assert parse_hotkey("Ctrl+F5")[1] == 0x74
    assert parse_hotkey("R") is None
    assert parse_hotkey("Ctrl+Alt+PgUp") is None
