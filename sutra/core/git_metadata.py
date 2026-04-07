from __future__ import annotations

import subprocess
from pathlib import Path


def resolve_commit_hash(root: Path) -> str:
    """
    Return the HEAD commit SHA for the git repository at `root`.
    Returns "unknown" if the directory is not a git repo or git is unavailable.
    """
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def resolve_remote_url(root: Path) -> str | None:
    """
    Return the URL of the 'origin' remote for the git repository at `root`.
    Returns None if there is no origin remote or the directory is not a git repo.
    """
    result = subprocess.run(
        ["git", "-C", str(root), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def repo_name_from_path(root: Path) -> str:
    """
    Fallback repo name when no remote URL is available: use the directory name.
    """
    return root.name
