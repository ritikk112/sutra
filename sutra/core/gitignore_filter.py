"""
.gitignore filtering for Sutra.

GitignoreFilter reads all .gitignore files found under `root` and exposes a
single `should_ignore(rel_path) -> bool` predicate.

Usage
-----
    filt = GitignoreFilter(root)
    if not filt.should_ignore("src/vendor/generated.py"):
        # index this file
        ...

Integration points
------------------
1. Indexer._walk_files() — constructed once per indexing run; each candidate
   file is checked before being yielded.
2. IncrementalUpdater — applied to the changed-files list from git_differ before
   processing.  Deleted files bypass the filter (we must remove their symbols
   from the DB even if the file was gitignored).

Dependency
----------
Requires `pathspec` (pip install pathspec).  If pathspec is not installed, a
warning is printed to stderr and the filter becomes a no-op (nothing ignored).
This graceful degradation means existing tests never break if pathspec is absent.

Pattern loading
---------------
- Root-level `.gitignore` is always loaded.
- All `.gitignore` files found recursively under `root` are loaded.
- pathspec handles pattern anchoring (patterns in `a/.gitignore` are relative
  to `a/`).  We normalise all patterns to be root-relative before matching.

Note: `.git/` and other Indexer-level excluded dirs are handled in `Indexer`
itself; GitignoreFilter is only for user-authored .gitignore rules.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


class GitignoreFilter:
    """
    Wraps pathspec to test whether a repo-relative path should be ignored.

    Parameters
    ----------
    root : Path
        Absolute path to the repository root.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._spec = self._load(root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_ignore(self, rel_path: str) -> bool:
        """
        Return True if `rel_path` (repo-relative, POSIX) matches any .gitignore rule.

        Returns False when pathspec is not installed (no-op mode).
        """
        if self._spec is None:
            return False
        return bool(self._spec.match_file(rel_path))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load(root: Path):
        """
        Load all .gitignore files under `root` into a single pathspec Spec.

        Returns None if pathspec is not installed.
        """
        try:
            import pathspec  # noqa: PLC0415
        except ImportError:
            print(
                "[GitignoreFilter] pathspec not installed — .gitignore rules will not "
                "be applied.  Install with: pip install pathspec",
                file=sys.stderr,
            )
            return None

        patterns: list[str] = []

        for dirpath, _dirnames, filenames in os.walk(root):
            if ".gitignore" not in filenames:
                continue
            gitignore_path = Path(dirpath) / ".gitignore"
            try:
                raw = gitignore_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            # Make patterns relative to repo root so they match rel_path correctly.
            # For a .gitignore in `a/b/.gitignore`, prefix each non-empty,
            # non-comment pattern with `a/b/`.
            dir_rel = Path(dirpath).relative_to(root)
            prefix = str(dir_rel).replace("\\", "/")
            if prefix == ".":
                prefix = ""
            else:
                prefix = prefix + "/"

            for line in raw.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                # Negation patterns (!foo) are kept as-is — pathspec handles them.
                if prefix and not stripped.startswith("!"):
                    patterns.append(prefix + stripped)
                elif prefix and stripped.startswith("!"):
                    # "!foo" in sub-dir → "!prefix/foo"
                    patterns.append("!" + prefix + stripped[1:])
                else:
                    patterns.append(stripped)

        return pathspec.PathSpec.from_lines("gitignore", patterns)
