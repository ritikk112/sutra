"""Tests for `sutra.cli.detect` — read-only checks against the real environment.

No mocks: every check runs against the actual interpreter / filesystem / PATH.
Assertions are loose where the environment is variable (docker, GPU) and exact
where the interpreter guarantees an answer (Python version, venv).
"""
from __future__ import annotations

from pathlib import Path

from sutra.cli import detect


class TestSimpleChecks:
    def test_python_version_is_311_plus(self) -> None:
        # The whole repo requires >=3.11; the running interpreter must satisfy it.
        result = detect.python_version()
        assert result.ok is True
        assert result.critical is True
        assert "." in result.detail

    def test_in_virtualenv_detects_the_test_venv(self) -> None:
        # These tests run under .venv, so this must report active.
        result = detect.in_virtualenv()
        assert result.ok is True

    def test_docker_present_returns_bool_result(self) -> None:
        result = detect.docker_present()
        assert isinstance(result.ok, bool)
        assert result.critical is False  # docker is optional

    def test_gpu_present_is_non_critical(self) -> None:
        result = detect.gpu_present()
        assert isinstance(result.ok, bool)
        assert result.critical is False

    def test_pyright_check_shape(self) -> None:
        result = detect.pyright_langserver_present()
        assert isinstance(result.ok, bool)
        assert result.name == "pyright-langserver"

    def test_sentence_transformers_check_matches_find_spec(self) -> None:
        import importlib.util

        expected = importlib.util.find_spec("sentence_transformers") is not None
        assert detect.sentence_transformers_importable().ok is expected

    def test_command_present_finds_python_executable(self) -> None:
        import sys

        # sys.executable's directory is on PATH inside a venv; `python` resolves.
        assert detect.command_present("python").ok or detect.command_present("python3").ok
        assert sys.executable  # sanity


class TestEnvVar:
    def test_env_var_set_true(self, monkeypatch) -> None:
        monkeypatch.setenv("SUTRA_TEST_VAR", "yes")
        result = detect.env_var_set("SUTRA_TEST_VAR")
        assert result.ok is True

    def test_env_var_unset_false(self, monkeypatch) -> None:
        monkeypatch.delenv("SUTRA_TEST_VAR", raising=False)
        result = detect.env_var_set("SUTRA_TEST_VAR")
        assert result.ok is False

    def test_env_var_empty_is_false(self, monkeypatch) -> None:
        monkeypatch.setenv("SUTRA_TEST_VAR", "   ")
        assert detect.env_var_set("SUTRA_TEST_VAR").ok is False


class TestArtifactsDir:
    def test_missing_dir_reports_not_ok(self, tmp_path: Path) -> None:
        result = detect.artifacts_dir_exists(tmp_path / "nope")
        assert result.ok is False

    def test_existing_dir_reports_ok_with_count(self, tmp_path: Path) -> None:
        (tmp_path / "repo-a").mkdir()
        (tmp_path / "repo-b").mkdir()
        result = detect.artifacts_dir_exists(tmp_path)
        assert result.ok is True
        assert "2" in result.detail


class TestLoaders:
    def test_load_config_missing_returns_empty(self, tmp_path: Path) -> None:
        assert detect.load_config(tmp_path / "sutra.yaml") == {}

    def test_load_config_reads_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "sutra.yaml"
        p.write_text("embedder:\n  provider: local\n  dimensions: 384\n")
        cfg = detect.load_config(p)
        assert cfg["embedder"]["provider"] == "local"

    def test_load_config_malformed_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "sutra.yaml"
        p.write_text("embedder: [unterminated\n")
        assert detect.load_config(p) == {}

    def test_load_env_parses_pairs_and_quotes(self, tmp_path: Path) -> None:
        p = tmp_path / ".env"
        p.write_text(
            "# comment\n"
            "OPENAI_API_KEY=sk-abc\n"
            'export SUTRA_ARTIFACTS_DIR="/home/x/.sutra/artifacts"\n'
            "\n"
            "BLANKLINE\n"
        )
        env = detect.load_env(p)
        assert env["OPENAI_API_KEY"] == "sk-abc"
        assert env["SUTRA_ARTIFACTS_DIR"] == "/home/x/.sutra/artifacts"
        assert "BLANKLINE" not in env  # no '=' -> ignored

    def test_load_env_missing_returns_empty(self, tmp_path: Path) -> None:
        assert detect.load_env(tmp_path / ".env") == {}
