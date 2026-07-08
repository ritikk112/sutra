"""``sutra init`` — the guided setup wizard.

Orchestrates the 8-step flow from the design spec.  Every step follows
*detect → explain → ask → act-on-consent → verify*, every step is skippable, and
the wizard is idempotent: it loads any existing ``config/sutra.yaml`` + ``.env``
and offers current values as defaults.

Critical invariant: **config is written only at the end (step 7), atomically.**
A Ctrl-C / ESC anywhere before that raises :class:`WizardAborted` and leaves the
filesystem untouched.

All prompting is delegated to the questionary helpers in :mod:`embed_setup`; all
logic (detect / config_io / provision) lives in the other modules.  ``run`` takes
an injectable :class:`PromptIO` so the whole flow is scriptable in tests.
"""
from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from sutra.cli import config_io, detect, embed_setup, provision
from sutra.cli.embed_setup import (
    PromptIO,
    WizardAborted,
    ask_confirm,
    ask_select,
    ask_text,
    choose_embedder,
)


def run(
    *,
    io: PromptIO | None = None,
    console: Any = None,
    config_path: Path | None = None,
    env_path: Path | None = None,
    repo_root: Path | None = None,
) -> int:
    """Drive the wizard.  Returns a process exit code (0 success, 1 aborted)."""
    from rich.console import Console

    io = io or PromptIO()
    console = console or Console()
    config_path = Path(config_path) if config_path else config_io.default_config_path()
    env_path = Path(env_path) if env_path else config_io.default_env_path()
    repo_root = Path(repo_root) if repo_root else Path.cwd()

    def emit(msg: str) -> None:
        console.print(f"  [dim]{msg}[/dim]")

    try:
        return _run(io, console, emit, config_path, env_path, repo_root)
    except WizardAborted:
        console.print("\n[yellow]Setup cancelled. No configuration was written.[/yellow]")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. No configuration was written.[/yellow]")
        return 1


def _run(
    io: PromptIO,
    console: Any,
    emit: Any,
    config_path: Path,
    env_path: Path,
    repo_root: Path,
) -> int:
    existing_config = detect.load_config(config_path)
    existing_env = detect.load_env(env_path)
    embedder_defaults = (existing_config or {}).get("embedder") or {}

    env_updates: dict[str, str] = {}
    summary: list[tuple[str, str]] = []

    # ---- Step 1: welcome + environment check -----------------------------
    console.print("\n[bold cyan]Sutra setup[/bold cyan] — let's get you indexed.\n")
    for check in (
        detect.python_version(),
        detect.in_virtualenv(),
        detect.docker_present(),
        detect.gpu_present(),
    ):
        mark = "[green]✓[/green]" if check.ok else "[yellow]•[/yellow]"
        console.print(f"  {mark} {check.name}: {check.detail}")
        if not check.ok and check.hint:
            emit(check.hint)

    # ---- Step 2: choose an embedder (centerpiece) ------------------------
    console.print("\n[bold]Step 2 — Embedder[/bold]")
    embed_result = choose_embedder(io, defaults=embedder_defaults, emit=emit)
    embedder_cfg = embed_result.embedder_cfg
    env_updates.update(embed_result.env_vars)
    validated = "validated" if embed_result.validated else "unvalidated"
    summary.append(
        ("Embedder", f"{embedder_cfg.get('provider')} / {embedder_cfg.get('model', '?')} "
                     f"({embed_result.dimensions or '?'} dims, {validated})")
    )

    # ---- Step 3: artifacts directory -------------------------------------
    console.print("\n[bold]Step 3 — Artifacts directory[/bold]")
    default_artifacts = str(existing_env.get("SUTRA_ARTIFACTS_DIR") or config_io.default_artifacts_dir())
    artifacts_dir = ask_text(io, "Where should indexed artifacts be stored?", default=default_artifacts).strip()
    if artifacts_dir:
        env_updates["SUTRA_ARTIFACTS_DIR"] = artifacts_dir
        summary.append(("Artifacts dir", artifacts_dir))

    # ---- Step 4: optional Postgres ---------------------------------------
    console.print("\n[bold]Step 4 — PostgreSQL (optional)[/bold]")
    emit("Sutra works fully without a database; Postgres only enables incremental re-indexing.")
    if ask_confirm(io, "Configure PostgreSQL now?", default=bool(existing_env.get("SUTRA_PG_URL"))):
        _setup_postgres(io, emit, env_updates, summary, repo_root)
    else:
        summary.append(("PostgreSQL", "skipped (JSON-only)"))

    # ---- Step 5: resolver / pyright --------------------------------------
    console.print("\n[bold]Step 5 — Call resolver[/bold]")
    emit("heuristic: fast, no deps. lsp: pyright type inference for ambiguous calls (Python only).")
    resolver = ask_select(
        io,
        "Which CALLS resolver should be the default?",
        choices=["heuristic", "lsp"],
    )
    summary.append(("Resolver", resolver))
    if resolver == "lsp" and not detect.pyright_langserver_present().ok:
        emit("pyright-langserver not found (required for the lsp resolver and the web UI).")
        if ask_confirm(io, "Install pyright now?", default=True):
            res = provision.install_pyright(on_output=emit)
            if not res.ok:
                emit(f"pyright install failed: {res.message}. You can `pip install pyright` later.")

    # ---- Step 6: optional MCP registration -------------------------------
    console.print("\n[bold]Step 6 — MCP registration (optional)[/bold]")
    artifacts_path = Path(artifacts_dir).expanduser() if artifacts_dir else config_io.default_artifacts_dir()
    claude = detect.claude_cli_present()
    if claude.ok:
        if ask_confirm(io, "Register the Sutra MCP server with Claude Code now?", default=False):
            res = provision.claude_mcp_add(artifacts_path, on_output=emit)
            summary.append(("MCP", "registered with Claude Code" if res.ok else f"failed ({res.message})"))
        else:
            summary.append(("MCP", "skipped"))
    else:
        emit("Claude Code CLI not found. Add this to your MCP client config:")
        console.print(provision.mcp_json_snippet(artifacts_path))
        summary.append(("MCP", "manual snippet printed"))

    # ---- Step 7: write config (atomic, diff-confirmed) -------------------
    console.print("\n[bold]Step 7 — Write configuration[/bold]")
    plan = config_io.plan_write(config_path, env_path, embedder_cfg, env_updates)
    if not plan.anything_changed:
        console.print("  Configuration already up to date — nothing to write.")
    else:
        if plan.config_changed:
            console.print(f"[bold]{config_path}[/bold]:")
            console.print(plan.config_diff() or "  (new file)")
        if plan.env_changed:
            console.print(f"[bold]{env_path}[/bold] (secret values redacted):")
            console.print(plan.env_diff() or "  (new file)")
        if not ask_confirm(io, "Write these files now?", default=True):
            console.print("[yellow]Skipped writing configuration.[/yellow]")
            return 1
        written = config_io.commit_write(plan)
        for p in written:
            console.print(f"  [green]wrote[/green] {p}")

    # ---- Step 8: finale ---------------------------------------------------
    console.print("\n[bold green]Setup complete![/bold green]")
    _print_summary(console, summary)
    if ask_confirm(io, "Index a repository now to verify the setup?", default=False):
        target = ask_text(io, "Local path or git URL to index:", default=str(repo_root)).strip()
        console.print(f"  Run: [bold]sutra index {target}[/bold]")
        emit("(re-run this exact command anytime; skipping the live index here.)")

    console.print("\nNext commands:")
    console.print("  [bold]sutra index <path-or-url>[/bold]   index a repo")
    console.print("  [bold]sutra serve[/bold]                  start the MCP server")
    console.print("  [bold]sutra ui[/bold]                     open the web UI")
    console.print("  [bold]sutra doctor[/bold]                 re-check your environment")
    return 0


def _setup_postgres(
    io: PromptIO,
    emit: Any,
    env_updates: dict[str, str],
    summary: list[tuple[str, str]],
    repo_root: Path,
) -> None:
    mode = ask_select(
        io,
        "PostgreSQL setup:",
        choices=[
            "Use an existing connection URL",
            "Build & run the bundled pgvector container (needs Docker)",
        ],
    )
    if mode.startswith("Use an existing"):
        url = ask_text(io, "SUTRA_PG_URL (postgresql://…):").strip()
        if url:
            env_updates["SUTRA_PG_URL"] = url
            summary.append(("PostgreSQL", "existing URL"))
        return

    if not detect.docker_present().ok:
        emit("Docker not found — skipping container build.")
        summary.append(("PostgreSQL", "skipped (no Docker)"))
        return

    password = secrets.token_urlsafe(16)
    port = 5433
    res = provision.build_and_run_pgvector(repo_root, password, port=port, on_output=emit)
    if res.ok:
        url = provision.pg_url_for("sutra", password, port, "sutra")
        env_updates["SUTRA_PG_URL"] = url
        summary.append(("PostgreSQL", f"container on port {port}"))
        emit(f"Postgres running; SUTRA_PG_URL set (password generated, stored in .env).")
    else:
        emit(f"Container setup failed: {res.message}")
        summary.append(("PostgreSQL", f"failed ({res.message})"))


def _print_summary(console: Any, summary: list[tuple[str, str]]) -> None:
    from rich.table import Table

    table = Table(title="Configuration summary", show_header=False)
    table.add_column("Item", style="bold")
    table.add_column("Value")
    for name, value in summary:
        table.add_row(name, value)
    console.print(table)
