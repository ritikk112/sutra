"""Consent-gated provisioning actions.

Every action here is destructive/expensive (pip installs, model downloads,
docker build/run, MCP registration) and is therefore **only** ever invoked after
the caller has obtained explicit consent.  This module does not prompt; it owns
two things:

* pure **command builders** (``*_command`` / ``*_commands``) that return the exact
  argv a step will run — printable *before* execution and unit-testable, and
* :func:`run_command`, which prints the command, streams combined output, and
  returns a :class:`ProvisionResult` the caller can branch on (retry/skip/abort).

The wizard never dies on a step failure: a non-zero ``ProvisionResult`` is data,
not an exception.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class ProvisionResult:
    ok: bool
    returncode: int
    command: list[str]
    message: str = ""
    output: str = field(default="", repr=False)


def format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def run_command(
    cmd: list[str],
    *,
    on_output: Callable[[str], None] | None = None,
    echo: bool = True,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> ProvisionResult:
    """Run ``cmd``, streaming combined stdout+stderr line-by-line.

    ``on_output`` (when given) receives each line (without trailing newline) as it
    arrives — the wizard wires this to a Rich console so users watch progress.
    The full captured text is also returned on the result for post-mortem hints.
    """
    cmd = [str(c) for c in cmd]
    if echo and on_output is not None:
        on_output(f"$ {format_command(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
    except FileNotFoundError as exc:
        return ProvisionResult(
            ok=False,
            returncode=127,
            command=cmd,
            message=f"command not found: {cmd[0]} ({exc})",
        )

    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        captured.append(line)
        if on_output is not None:
            on_output(line)
    rc = proc.wait()

    return ProvisionResult(
        ok=(rc == 0),
        returncode=rc,
        command=cmd,
        message="" if rc == 0 else f"exited with code {rc}",
        output="\n".join(captured),
    )


# ---------------------------------------------------------------------------
# Command builders (pure)
# ---------------------------------------------------------------------------

def cpu_torch_commands() -> list[list[str]]:
    """Two-step CPU-only torch install that avoids the multi-GB CUDA wheel trap."""
    pip = [sys.executable, "-m", "pip", "install"]
    return [
        pip + ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
        pip + ["sentence-transformers"],
    ]


def predownload_model_command(model_name: str) -> list[str]:
    code = (
        "from sentence_transformers import SentenceTransformer; "
        f"SentenceTransformer({model_name!r})"
    )
    return [sys.executable, "-c", code]


def install_pyright_command() -> list[str]:
    return [sys.executable, "-m", "pip", "install", "pyright"]


def pgvector_build_command(image_tag: str, context_dir: Path) -> list[str]:
    return ["docker", "build", "-t", image_tag, str(context_dir)]


def pgvector_run_command(
    container_name: str,
    password: str,
    *,
    port: int = 5433,
    image_tag: str = "sutra-pgvector",
    db: str = "sutra",
    user: str = "sutra",
) -> list[str]:
    return [
        "docker", "run", "-d",
        "--name", container_name,
        "-e", f"POSTGRES_USER={user}",
        "-e", f"POSTGRES_PASSWORD={password}",
        "-e", f"POSTGRES_DB={db}",
        "-p", f"{port}:5432",
        image_tag,
    ]


def pg_url_for(user: str, password: str, port: int, db: str) -> str:
    return f"postgresql://{user}:{password}@127.0.0.1:{port}/{db}"


def claude_mcp_add_command(artifacts_dir: Path) -> list[str]:
    return [
        "claude", "mcp", "add", "sutra", "--",
        "sutra", "serve", "--artifacts-dir", str(artifacts_dir),
    ]


def mcp_json_snippet(artifacts_dir: Path) -> str:
    """MCP server config snippet for clients other than Claude Code."""
    import json

    return json.dumps(
        {
            "mcpServers": {
                "sutra": {
                    "command": "sutra",
                    "args": ["serve", "--artifacts-dir", str(artifacts_dir)],
                }
            }
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Actions (build a command, then run it)
# ---------------------------------------------------------------------------

def install_cpu_torch(*, on_output: Callable[[str], None] | None = None) -> ProvisionResult:
    """Run the two-step CPU-torch + sentence-transformers install; stop on failure."""
    last: ProvisionResult | None = None
    for cmd in cpu_torch_commands():
        last = run_command(cmd, on_output=on_output)
        if not last.ok:
            return last
    assert last is not None
    return ProvisionResult(True, 0, last.command, "sentence-transformers installed")


def predownload_model(
    model_name: str, *, on_output: Callable[[str], None] | None = None
) -> ProvisionResult:
    return run_command(predownload_model_command(model_name), on_output=on_output)


def install_pyright(*, on_output: Callable[[str], None] | None = None) -> ProvisionResult:
    return run_command(install_pyright_command(), on_output=on_output)


def build_and_run_pgvector(
    context_dir: Path,
    password: str,
    *,
    container_name: str = "sutra-pg",
    port: int = 5433,
    image_tag: str = "sutra-pgvector",
    on_output: Callable[[str], None] | None = None,
) -> ProvisionResult:
    build = run_command(pgvector_build_command(image_tag, context_dir), on_output=on_output)
    if not build.ok:
        return build
    return run_command(
        pgvector_run_command(container_name, password, port=port, image_tag=image_tag),
        on_output=on_output,
    )


def claude_mcp_add(
    artifacts_dir: Path, *, on_output: Callable[[str], None] | None = None
) -> ProvisionResult:
    return run_command(claude_mcp_add_command(artifacts_dir), on_output=on_output)
