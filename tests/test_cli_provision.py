"""Tests for `sutra.cli.provision`.

Command builders are pure and asserted exactly.  `run_command` is exercised with
a *real* subprocess (the running interpreter) — no mocks — to confirm streaming,
return-code capture, and the not-found path.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sutra.cli import provision


class TestCommandBuilders:
    def test_cpu_torch_two_step(self) -> None:
        cmds = provision.cpu_torch_commands()
        assert len(cmds) == 2
        assert cmds[0][:3] == [sys.executable, "-m", "pip"]
        assert "--index-url" in cmds[0]
        assert "https://download.pytorch.org/whl/cpu" in cmds[0]
        assert cmds[1][-1] == "sentence-transformers"

    def test_predownload_model_command(self) -> None:
        cmd = provision.predownload_model_command("all-MiniLM-L6-v2")
        assert cmd[:2] == [sys.executable, "-c"]
        assert "all-MiniLM-L6-v2" in cmd[2]
        assert "SentenceTransformer" in cmd[2]

    def test_install_pyright_command(self) -> None:
        assert provision.install_pyright_command() == [
            sys.executable, "-m", "pip", "install", "pyright"
        ]

    def test_pgvector_build_and_run(self) -> None:
        build = provision.pgvector_build_command("sutra-pgvector", Path("/repo"))
        assert build == ["docker", "build", "-t", "sutra-pgvector", "/repo"]
        run = provision.pgvector_run_command("sutra-pg", "secretpw", port=5433)
        assert "POSTGRES_PASSWORD=secretpw" in run
        assert "5433:5432" in run

    def test_pg_url_for(self) -> None:
        url = provision.pg_url_for("sutra", "pw", 5433, "sutra")
        assert url == "postgresql://sutra:pw@127.0.0.1:5433/sutra"

    def test_claude_mcp_add_command(self) -> None:
        cmd = provision.claude_mcp_add_command(Path("/home/x/.sutra/artifacts"))
        assert cmd[:5] == ["claude", "mcp", "add", "sutra", "--"]
        assert cmd[-2:] == ["--artifacts-dir", "/home/x/.sutra/artifacts"]

    def test_mcp_json_snippet_is_valid_json(self) -> None:
        import json

        snippet = provision.mcp_json_snippet(Path("/art"))
        parsed = json.loads(snippet)
        assert parsed["mcpServers"]["sutra"]["command"] == "sutra"
        assert "/art" in parsed["mcpServers"]["sutra"]["args"]


class TestRunCommand:
    def test_success_streams_and_reports_zero(self) -> None:
        lines: list[str] = []
        res = provision.run_command(
            [sys.executable, "-c", "print('hello-from-child')"],
            on_output=lines.append,
        )
        assert res.ok is True
        assert res.returncode == 0
        assert any("hello-from-child" in ln for ln in lines)
        # The echoed command line is prefixed with '$ '.
        assert any(ln.startswith("$ ") for ln in lines)

    def test_nonzero_exit_is_data_not_exception(self) -> None:
        res = provision.run_command([sys.executable, "-c", "import sys; sys.exit(3)"])
        assert res.ok is False
        assert res.returncode == 3
        assert "3" in res.message

    def test_missing_binary_returns_127(self) -> None:
        res = provision.run_command(["this-binary-does-not-exist-xyz"])
        assert res.ok is False
        assert res.returncode == 127
        assert "not found" in res.message

    def test_output_captured_on_result(self) -> None:
        res = provision.run_command([sys.executable, "-c", "print('captured-line')"])
        assert "captured-line" in res.output
