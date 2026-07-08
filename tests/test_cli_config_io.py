"""Tests for `sutra.cli.config_io` — real YAML/.env round-trips in a tmp dir.

Covers the three guarantees the wizard relies on:
* atomic write (temp file + rename, real files),
* diff-on-overwrite (current vs new text),
* secrets land in .env and NEVER in YAML.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sutra.cli import config_io


class TestSanitizeEmbedderBlock:
    def test_drops_blank_optionals_keeps_provider(self) -> None:
        block = config_io.sanitize_embedder_block(
            {"provider": "openai", "model": "text-embedding-3-small", "base_url": "", "api_key_env": None}
        )
        assert block["provider"] == "openai"
        assert block["model"] == "text-embedding-3-small"
        assert "base_url" not in block
        assert "api_key_env" not in block

    def test_keeps_base_url_and_dims_for_compatible(self) -> None:
        block = config_io.sanitize_embedder_block(
            {"provider": "openai", "base_url": "http://localhost:11434/v1", "dimensions": 768}
        )
        assert block["base_url"] == "http://localhost:11434/v1"
        assert block["dimensions"] == 768

    def test_refuses_raw_secret_key(self) -> None:
        with pytest.raises(ValueError, match="secret"):
            config_io.sanitize_embedder_block({"provider": "openai", "api_key": "sk-leak"})


class TestSecretsNeverInYaml:
    def test_api_key_value_only_in_env(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config" / "sutra.yaml"
        env_path = tmp_path / ".env"
        plan = config_io.plan_write(
            cfg_path,
            env_path,
            embedder_cfg={
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimensions": 1536,
                "api_key_env": "OPENAI_API_KEY",
            },
            env_updates={"OPENAI_API_KEY": "sk-super-secret-value"},
        )
        config_io.commit_write(plan)

        yaml_text = cfg_path.read_text()
        env_text = env_path.read_text()

        # The env var NAME may appear in YAML; the secret VALUE must not.
        assert "sk-super-secret-value" not in yaml_text
        assert "api_key_env: OPENAI_API_KEY" in yaml_text
        assert "sk-super-secret-value" in env_text

    def test_env_diff_redacts_secret_value(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "sutra.yaml"
        env_path = tmp_path / ".env"
        plan = config_io.plan_write(
            cfg_path,
            env_path,
            embedder_cfg={"provider": "openai", "dimensions": 1536, "api_key_env": "OPENAI_API_KEY"},
            env_updates={"OPENAI_API_KEY": "sk-do-not-show"},
        )
        diff = plan.env_diff()
        assert "sk-do-not-show" not in diff
        assert "redacted" in diff


class TestRoundTrip:
    def test_yaml_written_is_valid_and_parses_back(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "sutra.yaml"
        env_path = tmp_path / ".env"
        plan = config_io.plan_write(
            cfg_path,
            env_path,
            embedder_cfg={"provider": "local", "model": "all-MiniLM-L6-v2", "dimensions": 384, "batch_size": 32},
            env_updates={"SUTRA_ARTIFACTS_DIR": "/tmp/sutra-art"},
        )
        config_io.commit_write(plan)

        parsed = yaml.safe_load(cfg_path.read_text())
        assert parsed["embedder"]["provider"] == "local"
        assert parsed["embedder"]["dimensions"] == 384
        assert config_io.read_env(env_path)["SUTRA_ARTIFACTS_DIR"] == "/tmp/sutra-art"

    def test_merge_preserves_other_top_level_keys(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "sutra.yaml"
        cfg_path.write_text("embedder:\n  provider: fixture\nother:\n  keep: yes\n")
        env_path = tmp_path / ".env"
        plan = config_io.plan_write(
            cfg_path, env_path,
            embedder_cfg={"provider": "openai", "dimensions": 1536, "api_key_env": "OPENAI_API_KEY"},
            env_updates={},
        )
        config_io.commit_write(plan)
        parsed = yaml.safe_load(cfg_path.read_text())
        assert parsed["embedder"]["provider"] == "openai"
        assert parsed["other"]["keep"] is True  # untouched

    def test_env_merge_keeps_existing_unrelated_keys(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        env_path.write_text("EXISTING_TOKEN=keepme\n")
        cfg_path = tmp_path / "sutra.yaml"
        plan = config_io.plan_write(
            cfg_path, env_path,
            embedder_cfg={"provider": "fixture"},
            env_updates={"SUTRA_ARTIFACTS_DIR": "/tmp/a"},
        )
        config_io.commit_write(plan)
        env = config_io.read_env(env_path)
        assert env["EXISTING_TOKEN"] == "keepme"
        assert env["SUTRA_ARTIFACTS_DIR"] == "/tmp/a"


class TestDiffAndIdempotence:
    def test_diff_shown_when_overwriting(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "sutra.yaml"
        cfg_path.write_text("embedder:\n  provider: fixture\n")
        env_path = tmp_path / ".env"
        plan = config_io.plan_write(
            cfg_path, env_path,
            embedder_cfg={"provider": "local", "dimensions": 384},
            env_updates={},
        )
        assert plan.config_changed is True
        diff = plan.config_diff()
        assert "fixture" in diff and "local" in diff

    def test_no_change_is_detected(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "sutra.yaml"
        env_path = tmp_path / ".env"
        embedder = {"provider": "local", "model": "all-MiniLM-L6-v2", "dimensions": 384, "batch_size": 32}
        first = config_io.plan_write(cfg_path, env_path, embedder, {})
        config_io.commit_write(first)
        # Re-planning with identical inputs must report nothing to do (idempotent).
        second = config_io.plan_write(cfg_path, env_path, embedder, {})
        assert second.config_changed is False
        assert second.anything_changed is False
        assert config_io.commit_write(second) == []


class TestAtomicWrite:
    def test_atomic_write_creates_parents_and_no_temp_left(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deep" / "sutra.yaml"
        config_io.atomic_write(target, "hello: world\n")
        assert target.read_text() == "hello: world\n"
        leftovers = list((tmp_path / "nested" / "deep").glob(".sutra-tmp-*"))
        assert leftovers == []

    def test_atomic_write_overwrites_in_place(self, tmp_path: Path) -> None:
        target = tmp_path / "f.yaml"
        config_io.atomic_write(target, "a: 1\n")
        config_io.atomic_write(target, "a: 2\n")
        assert target.read_text() == "a: 2\n"
