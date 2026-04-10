"""
Git differ for Sutra incremental updates.

Computes the set of files that changed between two commit SHAs using
`git diff --name-only --diff-filter=ACDM`.

Rename detection (`-M`) is intentionally NOT used.  Renamed files appear as
a delete of the old path plus an add of the new path.  This matches the
incremental updater's locked design (rename = delete old moniker + add new
moniker).  The wasted re-embedding of unchanged content in a renamed file is
acceptable in Phase 1.

`--diff-filter=ACDM` covers:
    A — Added
    C — Copied (treated as added; new path gets symbols)
    D — Deleted
    M — Modified

Commit-scoped only
------------------
Incremental update operates on committed changes only.  If the working tree
has uncommitted modifications, `git rev-parse HEAD` returns the same SHA on
successive calls and the updater will report "nothing to do."
This is by design — "commit your changes first."  The updater does NOT attempt
to detect or handle dirty working trees.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ChangedFiles:
    """
    Result of diffing two commits.

    All paths are repo-relative POSIX strings (forward slashes).
    """
    added: frozenset[str]
    modified: frozenset[str]
    deleted: frozenset[str]

    @property
    def all_to_index(self) -> frozenset[str]:
        """Files that need to be (re-)extracted: added ∪ modified."""
        return self.added | self.modified


def changed_files(root: Path, from_sha: str, to_sha: str) -> ChangedFiles:
    """
    Return the files that changed between `from_sha` and `to_sha`.

    Parameters
    ----------
    root : Path
        Absolute path to the root of the git repository.
    from_sha : str
        The earlier commit SHA (e.g. last indexed commit).
    to_sha : str
        The later commit SHA (e.g. current HEAD).

    Returns
    -------
    ChangedFiles
        Three disjoint sets of repo-relative paths.

    Raises
    ------
    RuntimeError
        If git exits with a non-zero return code (invalid SHAs, not a git repo,
        git not on PATH).
    """
    # --no-renames: prevent git from merging a delete+add pair into a rename event.
    # Without this flag, `git mv old.py new.py` produces a 'R100 old new' entry
    # which is filtered out by --diff-filter=ACDM (R is not included).
    # With --no-renames, git reports the delete and add separately — consistent
    # with our locked design: renames = delete old moniker + add new moniker.
    status_result = subprocess.run(
        [
            "git", "-C", str(root),
            "diff", "--name-status", "--diff-filter=ACDM", "--no-renames",
            f"{from_sha}..{to_sha}",
        ],
        capture_output=True,
        text=True,
    )
    if status_result.returncode != 0:
        raise RuntimeError(
            f"git diff --name-status failed (exit {status_result.returncode}):\n"
            f"{status_result.stderr.strip()}"
        )

    added: set[str] = set()
    modified: set[str] = set()
    deleted: set[str] = set()

    for line in status_result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: "<status>\t<path>"  e.g. "M\tsrc/foo.py" or "A\tsrc/bar.py"
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        status, path = parts[0].strip(), parts[1].strip()
        # Normalize to POSIX separators
        path = path.replace("\\", "/")
        if status.startswith("A") or status.startswith("C"):
            added.add(path)
        elif status.startswith("M"):
            modified.add(path)
        elif status.startswith("D"):
            deleted.add(path)

    return ChangedFiles(
        added=frozenset(added),
        modified=frozenset(modified),
        deleted=frozenset(deleted),
    )
