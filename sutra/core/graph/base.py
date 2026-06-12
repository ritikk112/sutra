"""
Abstract base classes for the SQL graph backend.

Replaces Apache AGE.  The seam was missing in Phase 1 — every consumer of
the graph layer imported AGEWriter / AGEReader directly.  Phase 2 introduces
explicit ABCs so the indexer, incremental updater, and any future consumer
depend on contracts, not on the AGE-specific implementation.

Two ABCs live here:

GraphWriter
    Write path used by Indexer and IncrementalUpdater.  Owns symbol upsert,
    relationship insert, deletion, and the bulk-write transaction boundary.
    Mirrors AGEWriter's public surface so PR 2 (the AGE → SQL switch) is a
    constructor-injection change in the pipelines, not a refactor of every
    call site.

IncrementalReader
    Read path used by IncrementalUpdater to compute per-file diffs.  Narrow
    on purpose — the ONLY method needed today is get_symbols_for_files.
    Per the elder/planner review (Round 5), this is split from any MCP-side
    graph-traversal interface.  MCP loads from the JSON+npy artifact, not
    from Postgres, so traversal verbs (expand_neighbors, get_callers, …)
    do NOT belong here.

IndexStateStore is intentionally not an ABC — it has one impl
(SqlIndexStateStore) and a tiny test fake is enough.  See planner Round 5
guidance: ABCs are for runtime polymorphism, not test injection.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sutra.core.extractor.base import IndexResult, Relationship, Symbol


class GraphWriter(ABC):
    """
    Write contract for the symbol/relationship graph store.

    Lifecycle
    ---------
    construct → setup() → (multiple write calls) → close()

    Or as a context manager.  setup() is idempotent — safe to call on every
    pipeline run.  close() releases the underlying connection.

    Transaction semantics
    ---------------------
    write_repository() is atomic — all symbols + relationships for one repo
    write or none of them do (modulo unresolved-relationship skipping, which
    is silent by design).  The other write_*_direct methods used by the
    incremental path are individually atomic; the updater orders them so a
    mid-run failure leaves the store in a recoverable state and Repository
    commit_sha is patched LAST (handled by IndexStateStore, not here).
    """

    @abstractmethod
    def setup(self) -> None:
        """Idempotent DDL: create tables, indexes, extensions if missing."""

    @abstractmethod
    def write_repository(self, result: IndexResult, replace: bool = False) -> None:
        """
        Upsert all symbols + resolved relationships from an IndexResult.

        When replace=True, all existing symbols belonging to result.repository
        are removed before the new symbols are written — prevents ghost
        symbols accumulating across re-index runs.  Default False (additive).

        Unresolved relationships (is_resolved=False or target_id is None) are
        skipped — they live in graph.json only.  Implementations MAY log the
        skipped count to stderr.
        """

    @abstractmethod
    def delete_symbols(self, monikers: list[str]) -> None:
        """
        Delete the given symbols and all their incident relationships.

        Used by the incremental updater for symbols that disappeared from a
        modified file or in a deleted file.  No-op when monikers is empty.
        """

    @abstractmethod
    def delete_relationships_from(self, source_monikers: list[str]) -> None:
        """
        Delete all OUTBOUND relationships from the given source monikers.

        Used by the incremental updater before re-inserting relationships for
        changed/added/deleted symbols in a file — wholesale per-source
        replacement, not relationship-level diffing.  No-op when empty.
        """

    @abstractmethod
    def write_symbol_direct(
        self,
        sym: Symbol,
        repo_name: str,
        indexed_at: str,
    ) -> None:
        """
        Upsert a single symbol outside of write_repository().

        Used by the incremental updater to write individual added/changed
        symbols without constructing a full IndexResult.
        """

    @abstractmethod
    def write_relationships_direct(self, relationships: list[Relationship]) -> int:
        """
        Bulk-insert the given relationships (skipping unresolved).

        Returns the count of skipped (unresolved) relationships for logging.
        Used by the incremental updater after writing updated symbols.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the underlying connection.  Idempotent."""

    def __enter__(self) -> "GraphWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class IncrementalReader(ABC):
    """
    Read contract used by IncrementalUpdater to compute per-file diffs.

    Narrow on purpose.  This interface does NOT carry graph-traversal verbs
    (expand_neighbors, get_callers, k-hop) — those belong on the MCP-side
    in-memory traversal interface (rustworkx-backed, lives in a different
    module).  Splitting them prevents an indexer-side caller from invoking
    a "1ms in-memory" verb that costs 500ms in SQL.
    """

    @abstractmethod
    def get_symbols_for_files(
        self,
        repo_name: str,
        file_paths: list[str],
    ) -> dict[str, str]:
        """
        Return {moniker: body_hash} for symbols belonging to the given files.

        The incremental updater uses this for the per-file four-way diff:
            old monikers not in new extraction       → deleted
            new monikers not in old                  → added
            same moniker, different body_hash        → changed
            same moniker, same body_hash             → unchanged (skip)

        Empty dict if file_paths is empty or no symbols match.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the underlying connection.  Idempotent."""

    def __enter__(self) -> "IncrementalReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
