"""Interactive `sutra init` flow test with scripted questionary answers.

No mocks: real detect calls, real config_io writes, real questionary prompts fed
through a prompt_toolkit pipe input (the documented test hook).  Keystroke rules:
* select / text prompts submit on Enter        -> send "\r"
* confirm prompts auto-submit on the y/n char  -> send "y" or "n" (no Enter)

The embedder step drives the Local branch with validation *skipped*, so the flow
needs no network and no live embedder.  Branch-dependent prompts (install
sentence-transformers, register MCP) are included in the script only when the
REAL environment would show them, keeping the test deterministic anywhere.
"""
from __future__ import annotations

import io as _io
from pathlib import Path

import pytest
import yaml
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from sutra.cli import detect
from sutra.cli import init as init_module
from sutra.cli.embed_setup import PromptIO


def _console() -> Console:
    return Console(file=_io.StringIO(), width=100, force_terminal=False)


def _build_local_script() -> str:
    """Assemble the keystrokes for the Local-embedder, skip-validation path."""
    st_installed = detect.sentence_transformers_importable().ok
    claude_present = detect.claude_cli_present().ok

    keys = ""
    keys += "\r"                       # Step2: embedder select -> Local (default)
    keys += "\r"                       # Step2: model text -> accept default
    if not st_installed:
        keys += "n"                    # Step2: install sentence-transformers? -> no
    keys += "n"                        # Step2: validate now? -> no (skip)
    keys += "\r"                       # Step3: artifacts dir -> accept default
    keys += "n"                        # Step4: configure Postgres? -> no
    keys += "\r"                       # Step5: resolver select -> heuristic (first)
    if claude_present:
        keys += "n"                    # Step6: register MCP with Claude Code? -> no
    keys += "y"                        # Step7: write files? -> yes
    keys += "n"                        # Step8: index a repo now? -> no
    return keys


def test_init_local_skip_validation_writes_config(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config" / "sutra.yaml"
    env_path = tmp_path / ".env"
    artifacts = tmp_path / "artifacts"
    # Make the artifacts-dir default deterministic (accepted with a bare Enter).
    monkeypatch.setenv("SUTRA_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.delenv("SUTRA_PG_URL", raising=False)

    with create_pipe_input() as inp:
        inp.send_text(_build_local_script())
        rc = init_module.run(
            io=PromptIO(input=inp, output=DummyOutput()),
            console=_console(),
            config_path=config_path,
            env_path=env_path,
            repo_root=tmp_path,
        )

    assert rc == 0
    assert config_path.exists()
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["embedder"]["provider"] == "local"
    assert cfg["embedder"]["dimensions"] == 384

    env = detect.load_env(env_path)
    assert env["SUTRA_ARTIFACTS_DIR"] == str(artifacts)
    # No secret was entered, so no API key leaked anywhere.
    assert "sk-" not in config_path.read_text()


def test_init_decline_write_leaves_no_config(tmp_path: Path, monkeypatch) -> None:
    """Declining the final write must leave zero config on disk."""
    config_path = tmp_path / "config" / "sutra.yaml"
    env_path = tmp_path / ".env"
    monkeypatch.setenv("SUTRA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("SUTRA_PG_URL", raising=False)

    st_installed = detect.sentence_transformers_importable().ok
    claude_present = detect.claude_cli_present().ok
    keys = "\r" "\r"
    if not st_installed:
        keys += "n"
    keys += "n"          # skip validation
    keys += "\r"         # artifacts default
    keys += "n"          # postgres no
    keys += "\r"         # resolver heuristic
    if claude_present:
        keys += "n"      # mcp no
    keys += "n"          # write? -> NO

    with create_pipe_input() as inp:
        inp.send_text(keys)
        rc = init_module.run(
            io=PromptIO(input=inp, output=DummyOutput()),
            console=_console(),
            config_path=config_path,
            env_path=env_path,
            repo_root=tmp_path,
        )

    assert rc == 1
    assert not config_path.exists()
    assert not env_path.exists()


def test_init_ctrl_c_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    """Ctrl-C at the very first prompt aborts cleanly with no partial config."""
    config_path = tmp_path / "config" / "sutra.yaml"
    env_path = tmp_path / ".env"
    monkeypatch.setenv("SUTRA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    with create_pipe_input() as inp:
        inp.send_text("\x03")  # Ctrl-C on the embedder select
        rc = init_module.run(
            io=PromptIO(input=inp, output=DummyOutput()),
            console=_console(),
            config_path=config_path,
            env_path=env_path,
            repo_root=tmp_path,
        )

    assert rc == 1
    assert not config_path.exists()
    assert not env_path.exists()
