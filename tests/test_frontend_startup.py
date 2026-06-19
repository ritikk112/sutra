import os
from pathlib import Path

from frontend.api.main import _artifacts_dir, _pyright_available


def test_artifacts_dir_defaults_under_home(monkeypatch):
    monkeypatch.delenv("SUTRA_ARTIFACTS_DIR", raising=False)
    assert _artifacts_dir() == Path.home() / ".sutra" / "artifacts"


def test_artifacts_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SUTRA_ARTIFACTS_DIR", str(tmp_path / "art"))
    assert _artifacts_dir() == tmp_path / "art"


def test_pyright_probe_returns_bool():
    assert isinstance(_pyright_available(), bool)
