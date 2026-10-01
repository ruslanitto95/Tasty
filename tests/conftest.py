from __future__ import annotations

import os
import tempfile

import pytest

# Isolate every test run from the real user profile before mva.paths is used.
_HOME = tempfile.mkdtemp(prefix="mva-test-home-")
os.environ.setdefault("MVA_HOME", _HOME)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def mva_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MVA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _memory_keyring():
    import keyring
    from keyring.backend import KeyringBackend

    class MemoryKeyring(KeyringBackend):
        priority = 1  # type: ignore[assignment]

        def __init__(self):
            super().__init__()
            self.store = {}

        def get_password(self, service, username):
            return self.store.get((service, username))

        def set_password(self, service, username, password):
            self.store[(service, username)] = password

        def delete_password(self, service, username):
            self.store.pop((service, username), None)

    previous = keyring.get_keyring()
    keyring.set_keyring(MemoryKeyring())
    yield
    keyring.set_keyring(previous)


@pytest.fixture(scope="session")
def gigaam():
    """Real GigaAM via the app's own ModelManager (downloads once on first use)."""
    from mva.transcription.gigaam_provider import GigaAMTranscriptionProvider
    from mva.transcription.model_manager import ModelManager, load_manifest
    from tests.helpers import models_dir_for_tests

    manager = ModelManager(load_manifest(), models_dir_for_tests())
    manager.ensure_downloaded()
    provider = GigaAMTranscriptionProvider(manager)
    provider.load()
    return provider
