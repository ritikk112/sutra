"""The ``sutra`` console command (entry point ``sutra.cli.main:app``).

A thin Typer command tree over the existing entrypoints — no indexing / MCP /
frontend logic is duplicated here.  Each subcommand translates flags and calls an
existing ``main(argv)`` (or launches uvicorn / the wizard):

    sutra init     -> cli.init.run()
    sutra index    -> pipelines.full_index.main([...])   (clones a git URL first)
    sutra serve    -> sutra.mcp.__main__.main([...])
    sutra ui       -> uvicorn frontend.api.main:app
    sutra doctor   -> cli.doctor.run()
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import typer

from sutra.cli import config_io

app = typer.Typer(
    name="sutra",
    add_completion=False,
    no_args_is_help=True,
    help="Sutra — code-aware retrieval. Index repos, serve them over MCP, and browse the UI.",
)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@app.command()
def init() -> None:
    """Run the interactive setup wizard (embedder, artifacts, Postgres, MCP)."""
    from sutra.cli import init as init_module

    raise typer.Exit(init_module.run())


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

@app.command()
def index(
    target: str = typer.Argument(..., help="Local repo path OR a git URL to clone and index."),
    resolver: str = typer.Option("heuristic", help="CALLS resolver: heuristic | lsp | none."),
    replace: bool = typer.Option(
        False, "--replace", help="Delete existing symbols for this repo before writing."
    ),
    repo_url: str = typer.Option(
        None, "--repo-url", help="Override the canonical repo identity (local paths only)."
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Artifacts root (default: $SUTRA_ARTIFACTS_DIR or ~/.sutra/artifacts)."
    ),
    config: Path = typer.Option(
        None, "--config", help="Path to sutra.yaml (default: config/sutra.yaml)."
    ),
) -> None:
    """Index a local path or a git URL into the artifacts directory."""
    from pipelines import full_index
    from sutra.core.extractor.moniker import repo_dir_slug, repo_name_from_url

    artifacts_root = Path(output_dir).expanduser() if output_dir else config_io.default_artifacts_dir()
    config_path = Path(config).expanduser() if config else config_io.default_config_path()

    def _run(root: Path, identity: str) -> int:
        out = artifacts_root / repo_dir_slug(repo_name_from_url(identity))
        out.mkdir(parents=True, exist_ok=True)
        argv = [
            "--root", str(root),
            "--repo-url", identity,
            "--output-dir", str(out),
            "--config", str(config_path),
            "--resolver", resolver,
        ]
        if replace:
            argv.append("--replace")
        typer.echo(f"Indexing {identity} -> {out}")
        return full_index.main(argv)

    if _looks_like_git_url(target):
        tmp = Path(tempfile.mkdtemp(prefix="sutra-index-"))
        clone_path = tmp / "repo"
        try:
            typer.echo(f"Cloning {target} ...")
            clone = subprocess.run(["git", "clone", "--depth", "1", target, str(clone_path)])
            if clone.returncode != 0:
                typer.secho(f"git clone failed (exit {clone.returncode}).", err=True, fg="red")
                raise typer.Exit(1)
            raise typer.Exit(_run(clone_path, target))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        root = Path(target).expanduser().resolve()
        if not root.is_dir():
            typer.secho(f"Not a directory: {root}", err=True, fg="red")
            raise typer.Exit(2)
        identity = repo_url or f"local/{root.name}"
        raise typer.Exit(_run(root, identity))


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@app.command()
def serve(
    artifacts_dir: Path = typer.Option(
        None, "--artifacts-dir", help="Artifacts root (default: $SUTRA_ARTIFACTS_DIR or ~/.sutra/artifacts)."
    ),
    http: bool = typer.Option(False, "--http", help="Serve streamable HTTP instead of stdio."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
) -> None:
    """Start the Sutra MCP server over stdio (default) or HTTP."""
    from sutra.mcp.__main__ import main as mcp_main

    root = Path(artifacts_dir).expanduser() if artifacts_dir else config_io.default_artifacts_dir()
    argv = ["--artifacts-dir", str(root), "--host", host, "--port", str(port)]
    if http:
        argv.append("--http")
    raise typer.Exit(mcp_main(argv))


# ---------------------------------------------------------------------------
# ui
# ---------------------------------------------------------------------------

@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Launch the Sutra web UI (FastAPI frontend) via uvicorn."""
    import uvicorn

    typer.echo(f"Serving Sutra UI at http://{host}:{port}")
    uvicorn.run("frontend.api.main:app", host=host, port=port)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

@app.command()
def doctor(
    check_embedder: bool = typer.Option(
        False, "--check-embedder", help="Also run a live embedder probe (may hit the network / API)."
    ),
    config: Path = typer.Option(None, "--config", help="Path to sutra.yaml."),
) -> None:
    """Run non-interactive environment diagnostics with ✓/✗ and fix hints."""
    from sutra.cli import doctor as doctor_module

    config_path = Path(config).expanduser() if config else None
    raise typer.Exit(doctor_module.run(config_path=config_path, probe=check_embedder))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _looks_like_git_url(target: str) -> bool:
    """Reuse the frontend's git-URL heuristic; fall back to a local check."""
    try:
        from frontend.api.utils import looks_like_git_url

        return looks_like_git_url(target)
    except Exception:
        val = target.strip()
        return val.startswith(("http://", "https://", "git://", "ssh://")) or (
            val.startswith("git@") and val.endswith(".git")
        )


if __name__ == "__main__":
    app()
