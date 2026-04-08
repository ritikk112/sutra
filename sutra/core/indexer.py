from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sutra.core.extractor.base import (
    ClassSymbol,
    IndexResult,
    MethodSymbol,
    RelationKind,
    Relationship,
    Repository,
    Symbol,
    File,
)
from sutra.core.extractor.moniker import repo_name_from_url
from sutra.core.git_metadata import resolve_commit_hash
from sutra.core.output.json_graph_exporter import JsonGraphExporter


# ---------------------------------------------------------------------------
# Directory names that are always excluded from the file walk.
# .gitignore support is deferred to the incremental update pipeline (Priority 11).
# ---------------------------------------------------------------------------
_EXCLUDED_DIRS = frozenset({
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    "testdata",   # Go conventional test-fixture directory; ignored by go toolchain
})

# File extension → language string.
_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
}

# Filename suffixes that are never indexed (before extension check).
# _test.go files are excluded at the Indexer level, not the GoAdapter level,
# to keep the adapter pure (adapters never see test files).
_EXCLUDED_SUFFIXES = frozenset({"_test.go"})


class Indexer:
    """
    Orchestrates a full repository indexing run.

    Design: plug-and-play adapters and exporter.

        indexer = Indexer(
            adapters={"python": PythonAdapter()},
            exporter=JsonGraphExporter(),
        )
        result = indexer.index(
            root=Path("/path/to/repo"),
            repo_url="https://github.com/org/repo",
            output_dir=Path("/path/to/output"),
        )

    Adding TypeScript in Priority 7 is one line:
        adapters={"python": PythonAdapter(), "typescript": TypeScriptAdapter()}

    Contract:
    - Files that cannot be read (I/O error) are skipped; path + error go to
      IndexResult.failed_files.
    - Files where the adapter raises are also skipped and recorded.
    - A duplicate moniker (same id produced by two different symbols) is a bug
      in the moniker generator — this is asserted loudly here, not swallowed.
    - File walk order is sorted globally, so two runs on the same repo produce
      identical IndexResults (modulo timestamps and commit SHA).
    """

    def __init__(
        self,
        adapters: dict[str, Any],   # lang_string -> adapter with .extract()
        exporter: JsonGraphExporter,
    ) -> None:
        self.adapters = adapters
        self.exporter = exporter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(
        self,
        root: Path,
        repo_url: str,
        output_dir: Path,
    ) -> IndexResult:
        """
        Index the repository at `root` and write output to `output_dir`.

        `repo_url` is the canonical remote URL used to derive the repo name
        and populate the repository block in graph.json.  It is not fetched —
        the indexer only reads local files.

        Returns the assembled IndexResult (also written to output_dir).
        """
        repo_name = repo_name_from_url(repo_url)
        commit_hash = resolve_commit_hash(root)

        # Collect all candidate files, sorted for deterministic output
        all_files = sorted(self._walk_files(root))

        file_records: list[File] = []
        symbols: list[Symbol] = []
        relationships: list[Relationship] = []
        language_counts: dict[str, int] = {}
        failed_files: list[tuple[str, str]] = []
        seen_monikers: set[str] = set()

        for abs_path in all_files:
            rel_path = str(abs_path.relative_to(root)).replace("\\", "/")
            lang = _EXTENSION_MAP.get(abs_path.suffix.lower())
            if lang is None or lang not in self.adapters:
                continue

            # Read file bytes
            try:
                source_bytes = abs_path.read_bytes()
            except OSError as exc:
                failed_files.append((rel_path, str(exc)))
                continue

            # Run the language adapter
            try:
                extraction = self.adapters[lang].extract(
                    rel_path, source_bytes, repo_name
                )
            except Exception as exc:  # noqa: BLE001
                failed_files.append((rel_path, f"{type(exc).__name__}: {exc}"))
                continue

            # Moniker uniqueness — duplicate means a generator bug, fail loudly
            for sym in extraction.symbols:
                if sym.id in seen_monikers:
                    raise AssertionError(
                        f"Duplicate moniker produced during indexing: {sym.id!r}\n"
                        f"  file: {rel_path}\n"
                        "This is a bug in the moniker generator."
                    )
                seen_monikers.add(sym.id)

            file_records.append(extraction.file)
            symbols.extend(extraction.symbols)
            relationships.extend(extraction.relationships)
            language_counts[lang] = language_counts.get(lang, 0) + 1

        # Cross-file Go method → struct linking (O(n) in-memory dict lookup).
        # Must run after all files are processed so all ClassSymbols are known.
        if "go" in language_counts:
            self._resolve_go_methods(symbols, relationships)

        result = IndexResult(
            repository=Repository(url=repo_url, name=repo_name),
            files=file_records,
            symbols=symbols,
            relationships=relationships,
            indexed_at=datetime.now(timezone.utc),
            commit_hash=commit_hash,
            languages=language_counts,
            failed_files=failed_files,
        )

        self.exporter.export(result, output_dir)
        return result

    # ------------------------------------------------------------------
    # Post-aggregation linking
    # ------------------------------------------------------------------

    def _resolve_go_methods(
        self,
        symbols: list[Symbol],
        relationships: list[Relationship],
    ) -> None:
        """
        Post-aggregation pass: link Go methods to their receiver type when the
        type is defined in a different file from the method.

        The GoAdapter sets enclosing_class_id=None for cross-file methods and
        emits no CONTAINS relationship for them.  This pass:
          1. Builds a {(language, qualified_name) → class_id} index from all
             ClassSymbol instances.
          2. For each MethodSymbol with enclosing_class_id=None and language="go",
             derives the expected class qualified_name from the method's own
             qualified_name ("pkg.TypeName.MethodName" → "pkg.TypeName"), looks
             it up, and — if found — sets enclosing_class_id and emits a resolved
             CONTAINS relationship.
        """
        # Build class lookup: (language, qualified_name) → symbol_id
        class_index: dict[tuple[str, str], str] = {
            (sym.language, sym.qualified_name): sym.id
            for sym in symbols
            if isinstance(sym, ClassSymbol)
        }

        new_rels: list[Relationship] = []
        for sym in symbols:
            if not isinstance(sym, MethodSymbol):
                continue
            if sym.language != "go" or sym.enclosing_class_id is not None:
                continue

            # qualified_name for Go method: "package.TypeName.MethodName"
            parts = sym.qualified_name.rsplit(".", 2)
            if len(parts) < 3:
                continue  # malformed — skip

            package_part, type_name, _ = parts
            class_qualified = f"{package_part}.{type_name}"
            class_id = class_index.get(("go", class_qualified))
            if class_id is None:
                continue  # type in an external package or truly unresolvable

            sym.enclosing_class_id = class_id
            new_rels.append(Relationship(
                source_id=class_id,
                kind=RelationKind.CONTAINS,
                is_resolved=True,
                target_id=sym.id,
                target_name=None,
            ))

        relationships.extend(new_rels)

    # ------------------------------------------------------------------
    # File walk
    # ------------------------------------------------------------------

    def _walk_files(self, root: Path) -> Iterator[Path]:
        """
        Yield all files under `root`, skipping excluded directories and
        excluded suffixes (_test.go).
        Uses os.walk with in-place pruning of excluded dir names so we never
        descend into .git/, __pycache__/, node_modules/, testdata/, etc.
        """
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune in-place — os.walk respects this and won't descend
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in _EXCLUDED_DIRS and not d.endswith(".egg-info")
            )
            for filename in sorted(filenames):
                if any(filename.endswith(suf) for suf in _EXCLUDED_SUFFIXES):
                    continue
                yield Path(dirpath) / filename
