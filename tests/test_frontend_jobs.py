from pathlib import Path

from frontend.api.jobs import _full_index_cmd, _child_env


def test_child_env_strips_pg_url(monkeypatch):
    monkeypatch.setenv("SUTRA_PG_URL", "postgresql://x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    env = _child_env()
    assert "SUTRA_PG_URL" not in env
    assert env.get("OPENAI_API_KEY") == "sk-test"   # unrelated vars preserved


def test_index_cmd_has_lsp_and_no_pg_url():
    cmd = _full_index_cmd(
        python="/usr/bin/python", root=Path("/clone"),
        repo_url="https://github.com/team-a/api",
        output_dir=Path("/art/team-a__api"), config=Path("config/sutra.yaml"),
        replace=True,
    )
    assert "--resolver" in cmd and "lsp" in cmd
    assert "--pg-url" not in cmd
    assert "--replace" in cmd
    assert cmd[cmd.index("--output-dir") + 1] == "/art/team-a__api"


def test_index_cmd_without_replace():
    cmd = _full_index_cmd(
        python="/usr/bin/python", root=Path("/clone"),
        repo_url="https://github.com/team-a/api",
        output_dir=Path("/art/team-a__api"), config=Path("config/sutra.yaml"),
        replace=False,
    )
    assert "--replace" not in cmd
