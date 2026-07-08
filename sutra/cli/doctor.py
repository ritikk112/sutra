"""``sutra doctor`` — non-interactive health check.

Runs every read-only :mod:`detect` check plus an optional embedder validation
and prints a Rich ✓/✗ table with fix hints.  Returns a process exit code: 0 when
all *critical* checks pass, 1 otherwise.  It never prompts and never writes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from sutra.cli import config_io, detect
from sutra.cli.embed_setup import probe_embedder


def collect_checks(
    *,
    config_path: Path | None = None,
    env_path: Path | None = None,
    artifacts_dir: Path | None = None,
    probe: bool = False,
) -> list[detect.CheckResult]:
    """Gather all detection results (pure — returns data, prints nothing)."""
    config_path = config_path or config_io.default_config_path()
    env_path = env_path or config_io.default_env_path()
    artifacts_dir = artifacts_dir or config_io.default_artifacts_dir()

    checks: list[detect.CheckResult] = [
        detect.python_version(),
        detect.in_virtualenv(),
        detect.docker_present(),
        detect.gpu_present(),
        detect.pyright_langserver_present(),
        detect.sentence_transformers_importable(),
        detect.claude_cli_present(),
        detect.artifacts_dir_exists(artifacts_dir),
    ]

    cfg = detect.load_config(config_path)
    embedder_cfg = (cfg or {}).get("embedder") or {}
    provider = embedder_cfg.get("provider", "fixture")
    api_key_env = str(embedder_cfg.get("api_key_env") or "").strip()
    base_url = str(embedder_cfg.get("base_url") or "").strip()

    if config_path.exists():
        checks.append(
            detect.CheckResult(
                name="config/sutra.yaml",
                ok=True,
                detail=f"{config_path} (provider={provider})",
            )
        )
    else:
        checks.append(
            detect.CheckResult(
                name="config/sutra.yaml",
                ok=False,
                detail=f"{config_path} not found",
                hint="Run `sutra init` to create it.",
                critical=False,
            )
        )

    # An OpenAI provider needs a key UNLESS it is a keyless compatible endpoint.
    if provider == "openai" and not (base_url and not api_key_env):
        checks.append(detect.env_var_set(api_key_env or "OPENAI_API_KEY"))

    if probe and embedder_cfg:
        result = probe_embedder(embedder_cfg)
        checks.append(
            detect.CheckResult(
                name="Embedder probe",
                ok=result.ok,
                detail=(
                    f"{result.model_id} -> {result.dimensions} dims"
                    if result.ok
                    else (result.error or "failed")
                ),
                hint="" if result.ok else "Check the endpoint/API key, or re-run `sutra init`.",
                critical=False,
            )
        )

    return checks


def run(
    *,
    config_path: Path | None = None,
    env_path: Path | None = None,
    artifacts_dir: Path | None = None,
    probe: bool = False,
    console: Any = None,
) -> int:
    """Print the diagnostic table and return an exit code (0 ok / 1 problems)."""
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    checks = collect_checks(
        config_path=config_path,
        env_path=env_path,
        artifacts_dir=artifacts_dir,
        probe=probe,
    )

    table = Table(title="sutra doctor", show_lines=False, expand=False)
    table.add_column("", justify="center", width=3)
    table.add_column("Check", style="bold")
    table.add_column("Detail")
    table.add_column("Fix hint", style="dim")

    critical_failures = 0
    for c in checks:
        mark = "[green]✓[/green]" if c.ok else ("[red]✗[/red]" if c.critical else "[yellow]•[/yellow]")
        if not c.ok and c.critical:
            critical_failures += 1
        table.add_row(mark, c.name, c.detail, c.hint)

    console.print(table)

    if critical_failures:
        console.print(
            f"[red]{critical_failures} critical check(s) failed.[/red] "
            "Sutra may not run until these are fixed."
        )
        return 1
    console.print("[green]All critical checks passed.[/green]")
    return 0
