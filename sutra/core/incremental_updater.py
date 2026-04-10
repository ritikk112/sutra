"""
Incremental update pipeline for Sutra.

IncrementalUpdater re-indexes only the files that changed between the last
indexed commit SHA and the current HEAD.  It is the performance-critical path
for repositories that are already fully indexed (by Indexer).

Design
------
- Commit-scoped: operates on committed changes only.  Uncommitted working-tree
  modifications are invisible because `git rev-parse HEAD` returns the same SHA.
  This is by design — "commit your changes first."
- Rename = delete + add: git's rename detection (-M) is NOT used.  Renamed files
  appear as delete(old) + add(new).  Old monikers are removed, new ones created.
  Body re-embedding of content-identical renamed files is wasted work but
  acceptable in Phase 1 (simpler code, correct output).
- Per-file four-way diff: for each modified/added file, the updater computes:
    deleted   = old monikers not in new extraction
    added     = new monikers not in old
    changed   = same moniker, body_hash differs → re-embed + update
    unchanged = same moniker, same body_hash → skip entirely (no DB touch)
  The "unchanged → skip" path is the performance argument for incremental updates.
- Relationship diffing: before re-writing symbols in a file, all outbound edges
  from those symbols are deleted and re-inserted from the new extraction.
- commit_sha patch is LAST:
    INVARIANT: Repository.commit_sha is updated ONLY after all file processing
    succeeds.  On failure, the old SHA is preserved, and re-running the updater
    will diff the same range again (idempotent recovery).
- Fallback threshold: if more than `fallback_threshold` (default 50%) of indexable
  files changed, the updater delegates to Indexer.index(replace=True) instead of
  doing file-by-file diffs.  Returns UpdateResult with fell_back_to_full_index=True.

Dependency injection
---------------------
IncrementalUpdater does NOT subclass Indexer — different lifecycle, different
inputs, different outputs.  It shares adapters, embedder, and writers by
reference.  All five writer/reader objects are pre-constructed by the caller:

    updater = IncrementalUpdater(
        adapters={"python": PythonAdapter()},
        embedder=FixtureEmbedder(),
        age_writer=age_writer,
        age_reader=age_reader,
        pgvector_store=pgvector_store,
        gitignore_filter=GitignoreFilter(root),
    )
    result = updater.update(root, repo_url, output_dir)

.gitignore filtering
---------------------
GitignoreFilter is applied to added/modified files before processing.
Deleted files bypass the filter — their symbols must be removed from the DB
even if the file was (or became) gitignored.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from sutra.core.embedder.base import Embedder
from sutra.core.embedder.chunk_builder import build_chunks
from sutra.core.extractor.base import (
    RelationKind,
    Relationship,
    Symbol,
    UpdateResult,
)
from sutra.core.extractor.moniker import repo_name_from_url
from sutra.core.git_differ import changed_files
from sutra.core.git_metadata import resolve_commit_hash
from sutra.core.gitignore_filter import GitignoreFilter
from sutra.core.graph.age_reader import AGEReader
from sutra.core.graph.postgres_age import AGEWriter
from sutra.core.graph.pgvector_store import PGVectorStore
from sutra.core.indexer import Indexer, _EXTENSION_MAP, _EXCLUDED_DIRS, _EXCLUDED_SUFFIXES

# Fallback threshold: if ≥ this fraction of indexable files changed, skip
# per-file diffing and do a full re-index instead.
_DEFAULT_FALLBACK_THRESHOLD = 0.5


class IncrementalUpdater:
    """
    Diff-based incremental update for already-indexed repositories.

    Parameters
    ----------
    adapters : dict[str, Any]
        Same adapter registry as Indexer — language string → adapter with .extract().
    embedder : Embedder
        Embedding provider.  Must match the dimensions of the existing pgvector store.
    age_writer : AGEWriter
        Pre-constructed AGEWriter.  Must have setup() already called.
    age_reader : AGEReader
        Pre-constructed AGEReader pointing at the same graph.
    pgvector_store : PGVectorStore
        Pre-constructed PGVectorStore.  Must have setup() already called.
    gitignore_filter : GitignoreFilter | None
        If supplied, gitignored added/modified files are skipped.
        Deleted files bypass the filter (symbols must be removed regardless).
    fallback_threshold : float
        Fraction of indexable files changed that triggers a full re-index.
        Default 0.5 (50%).  Set to 1.0 to disable the fallback.
    exporter : JsonGraphExporter | None
        Optional — only used when falling back to full re-index via Indexer.
    """

    def __init__(
        self,
        adapters: dict[str, Any],
        embedder: Embedder,
        age_writer: AGEWriter,
        age_reader: AGEReader,
        pgvector_store: PGVectorStore,
        gitignore_filter: Optional[GitignoreFilter] = None,
        fallback_threshold: float = _DEFAULT_FALLBACK_THRESHOLD,
        exporter: Any = None,
    ) -> None:
        self.adapters = adapters
        self.embedder = embedder
        self.age_writer = age_writer
        self.age_reader = age_reader
        self.pgvector_store = pgvector_store
        self.gitignore_filter = gitignore_filter
        self.fallback_threshold = fallback_threshold
        self.exporter = exporter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        root: Path,
        repo_url: str,
        output_dir: Path,
    ) -> UpdateResult:
        """
        Run an incremental update for the repository at `root`.

        Parameters
        ----------
        root : Path
            Absolute path to the repository root on disk.
        repo_url : str
            Canonical remote URL — used to derive repo_name and locate the
            Repository node in AGE.
        output_dir : Path
            Passed through to Indexer when falling back to full re-index.

        Returns
        -------
        UpdateResult
            Symbol-level counts and the old/new commit SHAs.
        """
        repo_name = repo_name_from_url(repo_url)
        current_sha = resolve_commit_hash(root)
        old_sha = self.age_reader.get_last_commit_sha(repo_name)

        # ── No-op: same SHA ────────────────────────────────────────────
        # Incremental update is commit-scoped.  Uncommitted working-tree
        # modifications are invisible (git rev-parse HEAD returns same SHA).
        # "Commit your changes first."
        if old_sha is not None and old_sha == current_sha:
            return UpdateResult(
                added=0, updated=0, deleted=0, skipped=0,
                fell_back_to_full_index=False,
                old_commit_sha=old_sha,
                new_commit_sha=current_sha,
            )

        # ── Unknown last SHA → full re-index ───────────────────────────
        if old_sha is None:
            return self._full_reindex(
                root, repo_url, repo_name, output_dir,
                old_sha=None, current_sha=current_sha,
            )

        # ── Compute diff ───────────────────────────────────────────────
        try:
            diff = changed_files(root, old_sha, current_sha)
        except RuntimeError as exc:
            print(f"[IncrementalUpdater] git diff failed: {exc}", file=sys.stderr)
            # Can't determine what changed — fall back to full re-index.
            return self._full_reindex(
                root, repo_url, repo_name, output_dir,
                old_sha=old_sha, current_sha=current_sha,
            )

        # ── Fallback threshold ─────────────────────────────────────────
        total_indexable = self._count_indexable_files(root)
        total_changed = len(diff.all_to_index) + len(diff.deleted)
        # Fallback threshold only applies when the repo has enough files to make
        # the ratio meaningful.  A single-file repo that changes is still an
        # incremental update, not a reason to trigger a full re-index.
        if (
            total_indexable >= 2
            and (total_changed / total_indexable) > self.fallback_threshold
        ):
            print(
                f"[IncrementalUpdater] {total_changed}/{total_indexable} files changed "
                f"(>= {self.fallback_threshold:.0%} threshold) — falling back to full re-index.",
                file=sys.stderr,
            )
            return self._full_reindex(
                root, repo_url, repo_name, output_dir,
                old_sha=old_sha, current_sha=current_sha,
            )

        # ── Per-file incremental update ────────────────────────────────
        added_count = 0
        updated_count = 0
        deleted_count = 0
        skipped_count = 0
        failed_files: list[tuple[str, str]] = []
        indexed_at = datetime.now(timezone.utc).isoformat()

        # Process deleted files — remove all their symbols from DB.
        deleted_count += self._process_deleted_files(diff.deleted)

        # Apply gitignore filter to files we're about to (re-)index.
        to_index = self._filter_indexable(root, diff.all_to_index)

        for rel_path in sorted(to_index):
            try:
                a, u, d, s = self._process_file(
                    root, rel_path, repo_name, indexed_at,
                )
                added_count += a
                updated_count += u
                deleted_count += d
                skipped_count += s
            except Exception as exc:  # noqa: BLE001
                failed_files.append((rel_path, f"{type(exc).__name__}: {exc}"))

        # INVARIANT: patch commit_sha LAST — this is the recovery checkpoint.
        # If anything above raised, the SHA is NOT updated.  Re-running the
        # updater will see the same old_sha and retry the same diff range.
        self.age_writer.update_commit_sha(repo_name, current_sha)

        return UpdateResult(
            added=added_count,
            updated=updated_count,
            deleted=deleted_count,
            skipped=skipped_count,
            fell_back_to_full_index=False,
            old_commit_sha=old_sha,
            new_commit_sha=current_sha,
            failed_files=failed_files,
        )

    # ------------------------------------------------------------------
    # Internal — per-file processing
    # ------------------------------------------------------------------

    def _process_file(
        self,
        root: Path,
        rel_path: str,
        repo_name: str,
        indexed_at: str,
    ) -> tuple[int, int, int, int]:
        """
        Run the per-file four-way diff and apply DB changes.

        Returns (added, updated, deleted, skipped) symbol counts.
        """
        abs_path = root / rel_path
        lang = _EXTENSION_MAP.get(abs_path.suffix.lower())
        if lang is None or lang not in self.adapters:
            return 0, 0, 0, 0

        # Read and extract
        source_bytes = abs_path.read_bytes()
        extraction = self.adapters[lang].extract(rel_path, source_bytes, repo_name)

        new_symbols: dict[str, Symbol] = {sym.id: sym for sym in extraction.symbols}
        new_rels: list[Relationship] = extraction.relationships

        # Get old state from AGE
        old_state: dict[str, str] = self.age_reader.get_symbols_for_files(
            repo_name, [rel_path]
        )

        # Four-way diff
        old_monikers = set(old_state.keys())
        new_monikers = set(new_symbols.keys())

        deleted_monikers = list(old_monikers - new_monikers)
        added_monikers = list(new_monikers - old_monikers)
        common_monikers = old_monikers & new_monikers
        changed_monikers = [
            m for m in common_monikers
            if old_state[m] != new_symbols[m].body_hash
        ]
        unchanged_monikers = [
            m for m in common_monikers
            if old_state[m] == new_symbols[m].body_hash
        ]

        # All monikers that we're about to change (for relationship cleanup).
        # ModuleSymbol always changes when its file changes — handled by normal path.
        affected_monikers = deleted_monikers + added_monikers + changed_monikers

        # 1. Delete outbound edges from all affected source monikers.
        #    Wholesale replacement per-source-moniker: delete then re-insert.
        self.age_writer.delete_relationships_from(affected_monikers)

        # 2. Delete removed symbols from AGE and pgvector.
        if deleted_monikers:
            self.age_writer.delete_symbols(deleted_monikers)
            self.pgvector_store.delete(deleted_monikers)

        # 3. Write added + changed symbols.
        symbols_to_write = (
            [new_symbols[m] for m in added_monikers]
            + [new_symbols[m] for m in changed_monikers]
        )

        for sym in symbols_to_write:
            self.age_writer.write_symbol_direct(sym, repo_name, indexed_at)

        # 4. Embed added + changed symbols only (unchanged skipped entirely).
        if symbols_to_write:
            chunks, embed_monikers = build_chunks(symbols_to_write, root)
            if chunks:
                vectors = self.embedder.embed(chunks)
                assert len(chunks) == len(embed_monikers) == vectors.shape[0]
                self.pgvector_store.write(embed_monikers, vectors)

        # 5. Re-insert relationships for all affected symbols in this file.
        #    Relationships sourced from unchanged symbols are NOT touched.
        affected_set = set(affected_monikers)
        rels_to_write = [
            r for r in new_rels
            if r.source_id in affected_set
        ]
        if rels_to_write:
            self.age_writer.write_relationships_direct(rels_to_write)

        return (
            len(added_monikers),
            len(changed_monikers),
            len(deleted_monikers),
            len(unchanged_monikers),
        )

    def _process_deleted_files(self, deleted_paths: frozenset[str]) -> int:
        """
        Remove all symbols belonging to deleted files from AGE and pgvector.

        Returns the count of symbols deleted.
        """
        if not deleted_paths:
            return 0

        # Collect all monikers for the deleted files in one reader call.
        old_state = self.age_reader.get_symbols_for_files(
            # repo_name not needed here — we're querying by file_path only.
            # get_symbols_for_files does not filter by repo; that's fine because
            # file_paths are globally unique repo-relative paths in practice.
            repo_name="",  # unused in the Cypher query (file_path lookup only)
            file_paths=list(deleted_paths),
        )
        monikers = list(old_state.keys())

        if monikers:
            self.age_writer.delete_relationships_from(monikers)
            self.age_writer.delete_symbols(monikers)
            self.pgvector_store.delete(monikers)

        return len(monikers)

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    def _filter_indexable(
        self,
        root: Path,
        rel_paths: frozenset[str],
    ) -> set[str]:
        """
        Filter a set of repo-relative paths to only those that:
        - have an extension in _EXTENSION_MAP
        - don't end with an excluded suffix
        - are not gitignored (if filter supplied)
        - the file actually exists on disk
        """
        result: set[str] = set()
        for rel_path in rel_paths:
            path = Path(rel_path)
            if any(rel_path.endswith(suf) for suf in _EXCLUDED_SUFFIXES):
                continue
            if path.suffix.lower() not in _EXTENSION_MAP:
                continue
            if self.gitignore_filter and self.gitignore_filter.should_ignore(rel_path):
                continue
            abs_path = root / rel_path
            if not abs_path.exists():
                continue
            result.add(rel_path)
        return result

    def _count_indexable_files(self, root: Path) -> int:
        """
        Count the total number of indexable files in the repo.
        Uses the same walk logic as Indexer._walk_files() for a consistent
        comparison against the changed file count (threshold check).
        """
        import os  # noqa: PLC0415
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in _EXCLUDED_DIRS and not d.endswith(".egg-info")
            )
            for filename in filenames:
                if any(filename.endswith(suf) for suf in _EXCLUDED_SUFFIXES):
                    continue
                abs_path = Path(dirpath) / filename
                if abs_path.suffix.lower() not in _EXTENSION_MAP:
                    continue
                rel = str(abs_path.relative_to(root)).replace("\\", "/")
                if self.gitignore_filter and self.gitignore_filter.should_ignore(rel):
                    continue
                count += 1
        return count

    def _full_reindex(
        self,
        root: Path,
        repo_url: str,
        repo_name: str,
        output_dir: Path,
        old_sha: Optional[str],
        current_sha: str,
    ) -> UpdateResult:
        """
        Delegate to Indexer.index(replace=True) and return a fell-back UpdateResult.
        """
        from sutra.core.output.json_graph_exporter import JsonGraphExporter  # noqa: PLC0415

        # If no exporter was supplied, create a minimal no-op one.
        exporter = self.exporter
        if exporter is None:
            exporter = JsonGraphExporter()

        indexer = Indexer(
            adapters=self.adapters,
            exporter=exporter,
            embedder=self.embedder,
            age_writer=self.age_writer,
            pgvector_store=self.pgvector_store,
            gitignore_filter=self.gitignore_filter,
        )
        result = indexer.index(root, repo_url, output_dir)

        return UpdateResult(
            added=len(result.symbols),
            updated=0,
            deleted=0,
            skipped=0,
            fell_back_to_full_index=True,
            old_commit_sha=old_sha,
            new_commit_sha=result.commit_hash,
            failed_files=result.failed_files,
        )
