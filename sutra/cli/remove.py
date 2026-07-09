"""`sutra remove` — delete indexed repos from the artifacts directory.

Destructive by design.  Removes the per-repo JSON artifact directory under the
artifacts root — the only store the MCP server reads.  A running `sutra serve`
hot-unloads the repo via its watcher within one poll; a stopped server just
won't load it next boot.

Postgres is intentionally NOT touched: the default deployment is JSON-only and
serving never reads a DB (a DB is written only when SUTRA_PG_URL is set at
index time).

Resolution is deliberately conservative — a destructive command must never
delete a *different* repo than the one named:
  * the git-URL parser (which strips a leading host segment) is used ONLY for
    URL-shaped input, so a bare `owner/repo` name is never mis-stripped to
    `repo`;
  * candidates are case-folded to match the always-lowercase on-disk slugs;
  * a match must be a direct child of the artifacts root (no `..` escape).
A local checkout PATH resolves by basename (the same lossy identity `sutra
index` assigns via `local/<name>`), so removing by a path shares that tool's
basename-collision caveat — the confirmation prompt shows the resolved slug.

The module never prompts on its own — `run()` takes injectable confirm/echo
callables (defaulting to typer) so the flow is testable without a TTY.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from sutra.core.extractor.moniker import repo_dir_slug, repo_name_from_url

# Any of these in a child dir marks it as a (possibly partial) artifact dir —
# including one left half-written by an interrupted index, so it can be reclaimed.
_ARTIFACT_MARKERS = ("graph.json", "embeddings.npy", "embeddings_index.json", ".ready")

_URL_PREFIXES = ("http://", "https://", "ssh://", "git://", "git@")


def _looks_like_url(target: str) -> bool:
    return target.startswith(_URL_PREFIXES)


def _artifact_dirs(artifacts_root: Path) -> list[Path]:
    """Child dirs that hold at least one artifact marker (partial dirs count)."""
    if not artifacts_root.is_dir():
        return []
    return sorted(
        c for c in artifacts_root.iterdir()
        if c.is_dir() and any((c / m).exists() for m in _ARTIFACT_MARKERS)
    )


def _candidate_slugs(target: str) -> list[str]:
    """Ordered, case-folded slug guesses for a target that may be a git URL, an
    ``owner/repo`` name, an on-disk slug, or a local checkout path."""
    cands: list[str] = []

    def add(slug: str) -> None:
        folded = slug.strip().lower()   # on-disk slugs are always lowercase
        if folded and folded not in cands:
            cands.append(folded)

    # 1. A git URL / ssh shorthand — the host-stripping parser is correct here.
    if _looks_like_url(target):
        add(repo_dir_slug(repo_name_from_url(target)))
    # 2. The canonical repo name sutra_list_repos shows (owner/repo, or bare).
    add(repo_dir_slug(target))
    # 3. The target may already BE the on-disk slug (e.g. `frappe__frappe`).
    add(target)
    # 4. Lowest priority: a local checkout path indexes by basename as
    #    `local/<name>` — the same lossy identity `sutra index` assigns.
    path = Path(target).expanduser()
    if path.is_dir():
        add(repo_dir_slug(repo_name_from_url(f"local/{path.name}")))
    return cands


def _safe_child(artifacts_root: Path, slug: str) -> Optional[Path]:
    """The `slug` dir under root, only if it is a single-component direct child
    that exists — never a `..`/`/`-bearing path that could escape the root."""
    if not slug or "/" in slug or "\\" in slug or slug in ("..", "."):
        return None
    candidate = artifacts_root / slug
    try:
        if candidate.is_dir() and candidate.resolve().parent == artifacts_root.resolve():
            return candidate
    except OSError:
        return None
    return None


@dataclass
class RemovePlan:
    """What a remove would delete — resolved before any confirmation."""

    targets: list[Path] = field(default_factory=list)
    error: Optional[str] = None


def plan_remove(artifacts_root: Path, target: Optional[str]) -> RemovePlan:
    """Resolve `target` to the artifact dir(s) to delete.  `None` → every
    indexed repo.  An unresolvable target is reported (never a silent no-op,
    never a match to a different repo)."""
    artifacts_root = Path(artifacts_root)
    indexed = _artifact_dirs(artifacts_root)
    if target is None:
        return RemovePlan(targets=indexed)

    by_slug = {d.name: d for d in indexed}
    candidates = _candidate_slugs(target)
    for slug in candidates:
        if slug in by_slug:
            return RemovePlan(targets=[by_slug[slug]])
    # Orphan reclaim: a bare artifact dir with no markers left (an index that
    # died right after `mkdir`).  Only an exact, containment-checked slug match.
    for slug in candidates:
        child = _safe_child(artifacts_root, slug)
        if child is not None:
            return RemovePlan(targets=[child])

    listing = ", ".join(d.name for d in indexed) or "(none indexed)"
    return RemovePlan(error=f"No indexed repo matches {target!r}. Indexed: {listing}")


def execute_remove(plan: RemovePlan) -> list[tuple[Path, str]]:
    """Delete each target.  Returns (dir, error) for any that failed, so one
    bad entry never aborts the batch or leaves an uncaught traceback."""
    failures: list[tuple[Path, str]] = []
    for directory in plan.targets:
        try:
            if directory.is_symlink():
                directory.unlink()          # drop the link, never rmtree its target
            else:
                shutil.rmtree(directory)
        except FileNotFoundError:
            continue                        # already gone — treat as done
        except OSError as exc:
            failures.append((directory, str(exc)))
    return failures


def run(
    target: Optional[str],
    artifacts_dir: Path,
    *,
    assume_yes: bool = False,
    confirm: Optional[Callable[[str], bool]] = None,
    echo: Optional[Callable[[str], None]] = None,
    err: Optional[Callable[[str], None]] = None,
) -> int:
    """Resolve → confirm → delete.  0 ok · 1 declined · 2 bad target · 3 partial."""
    import typer

    _echo = echo or typer.echo
    _err = err or (lambda m: typer.secho(m, err=True, fg="red"))
    _confirm = confirm or typer.confirm

    plan = plan_remove(Path(artifacts_dir), target)
    if plan.error:
        _err(plan.error)
        return 2
    if not plan.targets:
        _echo("Nothing to remove — no repos are indexed.")
        return 0

    names = ", ".join(d.name for d in plan.targets)
    if target is None:
        prompt = f"Remove ALL {len(plan.targets)} indexed repos ({names})?"
    else:
        prompt = f"Remove indexed repo {names}?"
    if not assume_yes and not _confirm(prompt):
        _echo("Aborted.")
        return 1

    failures = execute_remove(plan)
    failed = {d for d, _ in failures}
    removed = [d for d in plan.targets if d not in failed]
    if removed:
        _echo(f"Removed {len(removed)} repo(s): {', '.join(d.name for d in removed)}")
        _echo(
            "A running `sutra serve` drops them within a few seconds; a stopped "
            "server won't load them next start."
        )
    for directory, error in failures:
        _err(f"Failed to remove {directory.name}: {error}")
    return 3 if failures else 0
