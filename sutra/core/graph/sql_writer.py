"""
Plain-SQL graph writer.  Replaces postgres_age.AGEWriter.

Drop-in replacement for AGEWriter on the indexer side: the public method
surface mirrors AGEWriter so the only change in pipelines/full_index.py and
pipelines/incremental_update.py (PR 2) is which class is constructed.

Connection lifecycle
--------------------
    writer = SqlGraphWriter(conn_str)
    writer.setup()                           # idempotent migrations
    writer.write_repository(result)          # bulk symbols + relationships
    writer.close()

Or as a context manager.

The connection is owned by the writer (matches AGEWriter's pattern).
autocommit=True: write_repository() uses an explicit transaction
(BEGIN/COMMIT/ROLLBACK).  Direct writes used by the incremental updater
are individually atomic.

Unresolved relationships
------------------------
Skipped entirely — same as AGEWriter.  Phase 1's wire format is graph.json
for unresolved edges; the SQL store only carries resolved relationships.
The skipped count is logged to stderr.

NOT thread-safe.  One writer per pipeline run.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import psycopg2

from sutra.core.extractor.base import (
    ClassSymbol,
    FunctionSymbol,
    IndexResult,
    MethodSymbol,
    ModuleSymbol,
    Relationship,
    Symbol,
    VariableSymbol,
    Visibility,
)
from sutra.core.graph.base import GraphWriter


_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _kind_of(sym: Symbol) -> str:
    """Map a Symbol subclass to the `kind` string stored in sutra_symbols.

    isinstance(MethodSymbol) is checked BEFORE FunctionSymbol because
    MethodSymbol subclasses FunctionSymbol — Python's isinstance returns
    True for both, and we want the more specific kind.
    """
    if isinstance(sym, MethodSymbol):
        return "method"
    if isinstance(sym, FunctionSymbol):
        return "function"
    if isinstance(sym, ClassSymbol):
        return "class"
    if isinstance(sym, VariableSymbol):
        return "variable"
    if isinstance(sym, ModuleSymbol):
        return "module"
    return "unknown"


def _symbol_row(sym: Symbol, repo_name: str, indexed_at: str) -> dict[str, Any]:
    """Flatten a Symbol into a row dict matching the sutra_symbols schema.

    Per-kind fields are NULL for non-applicable kinds.  Docstrings are
    truncated to 500 chars (mirrors AGEWriter behaviour — bodies stay in
    source files).
    """
    vis = (
        sym.visibility.value
        if isinstance(sym.visibility, Visibility)
        else str(sym.visibility)
    )

    row: dict[str, Any] = {
        "moniker": sym.id,
        "repo_name": repo_name,
        "kind": _kind_of(sym),
        "name": sym.name,
        "qualified_name": sym.qualified_name,
        "file_path": sym.file_path,
        "language": sym.language,
        "visibility": vis,
        "is_exported": sym.is_exported,
        "body_hash": sym.body_hash,
        "line_start": sym.location.line_start,
        "line_end": sym.location.line_end,
        "signature": None,
        "return_type": None,
        "is_async": None,
        "complexity": None,
        "docstring": None,
        "is_static": None,
        "is_constructor": None,
        "receiver_kind": None,
        "is_abstract": None,
        "base_classes": None,
        "type_annotation": None,
        "is_constant": None,
        "indexed_at": indexed_at,
    }

    # MethodSymbol is a FunctionSymbol — populate function fields too.
    if isinstance(sym, MethodSymbol):
        row["signature"] = sym.signature or None
        row["return_type"] = sym.return_type
        row["is_async"] = sym.is_async
        row["complexity"] = sym.complexity
        row["docstring"] = sym.docstring[:500] if sym.docstring else None
        row["is_static"] = sym.is_static
        row["is_constructor"] = sym.is_constructor
        row["receiver_kind"] = sym.receiver_kind
    elif isinstance(sym, FunctionSymbol):
        row["signature"] = sym.signature or None
        row["return_type"] = sym.return_type
        row["is_async"] = sym.is_async
        row["complexity"] = sym.complexity
        row["docstring"] = sym.docstring[:500] if sym.docstring else None
    elif isinstance(sym, ClassSymbol):
        row["is_abstract"] = sym.is_abstract
        row["base_classes"] = (
            ", ".join(sym.base_classes) if sym.base_classes else None
        )
        row["docstring"] = sym.docstring[:500] if sym.docstring else None
    elif isinstance(sym, VariableSymbol):
        row["type_annotation"] = sym.type_annotation
        row["is_constant"] = sym.is_constant
    elif isinstance(sym, ModuleSymbol):
        row["docstring"] = sym.docstring[:500] if sym.docstring else None

    return row


_SYMBOL_COLUMNS = (
    "moniker", "repo_name", "kind", "name", "qualified_name",
    "file_path", "language", "visibility", "is_exported", "body_hash",
    "line_start", "line_end",
    "signature", "return_type", "is_async", "complexity", "docstring",
    "is_static", "is_constructor", "receiver_kind",
    "is_abstract", "base_classes",
    "type_annotation", "is_constant",
    "indexed_at",
)

_SYMBOL_UPSERT_SQL = (
    f"INSERT INTO sutra_symbols ({', '.join(_SYMBOL_COLUMNS)}) "
    f"VALUES ({', '.join('%s' for _ in _SYMBOL_COLUMNS)}) "
    "ON CONFLICT (moniker) DO UPDATE SET "
    + ", ".join(
        f"{col} = EXCLUDED.{col}"
        for col in _SYMBOL_COLUMNS
        if col != "moniker"
    )
)


class SqlGraphWriter(GraphWriter):
    """Plain-SQL implementation of GraphWriter.  Owns its psycopg2 connection."""

    def __init__(self, conn_str: str) -> None:
        # Credentials in conn_str are never logged; use conn.dsn for safe repr.
        self._conn = psycopg2.connect(conn_str)
        self._conn.autocommit = True

    # ------------------------------------------------------------------
    # Public API — mirrors AGEWriter so PR 2 is a constructor swap
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Run all migration files in order.  Idempotent (CREATE IF NOT EXISTS)."""
        migration_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
        with self._conn.cursor() as cur:
            for sql_file in migration_files:
                cur.execute(sql_file.read_text())

    def write_repository(
        self,
        result: IndexResult,
        replace: bool = False,
    ) -> None:
        """
        Upsert all symbols + resolved relationships from an IndexResult.

        When replace=True, all existing symbols for this repo are deleted
        before the new symbols are written (relationships go with them via
        the application-level cascade in delete_symbols).
        """
        repo = result.repository
        commit_sha = result.commit_hash
        indexed_at = result.indexed_at.isoformat()

        with self._conn.cursor() as cur:
            cur.execute("BEGIN")
            try:
                # Repository row first — symbols.repo_name FK depends on it.
                cur.execute(
                    """
                    INSERT INTO sutra_repositories (name, url, last_commit_sha, indexed_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET
                        url = EXCLUDED.url,
                        last_commit_sha = EXCLUDED.last_commit_sha,
                        indexed_at = EXCLUDED.indexed_at
                    """,
                    (repo.name, repo.url, commit_sha, indexed_at),
                )

                if replace:
                    self._delete_repo_symbols(cur, repo.name)

                for sym in result.symbols:
                    row = _symbol_row(sym, repo.name, indexed_at)
                    cur.execute(
                        _SYMBOL_UPSERT_SQL,
                        tuple(row[col] for col in _SYMBOL_COLUMNS),
                    )

                skipped = self._write_relationships(cur, result.relationships)
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise

        if skipped:
            print(
                f"[SqlGraphWriter] Skipped {skipped} unresolved relationships "
                "(will be written after Phase 2 resolution)",
                file=sys.stderr,
            )

    def delete_symbols(self, monikers: list[str]) -> None:
        """
        Delete the given symbols AND every relationship they appear in
        (as source or target).  Matches AGE's DETACH DELETE semantics
        — a deleted symbol leaves no orphan edges behind.
        """
        if not monikers:
            return
        with self._conn.cursor() as cur:
            # Delete edges first, then symbols.  Relationships have no FK
            # to symbols (see migration 003), so application-level cleanup
            # is required before the symbol row goes.
            cur.execute(
                "DELETE FROM sutra_relationships "
                "WHERE source_id = ANY(%s) OR target_id = ANY(%s)",
                (monikers, monikers),
            )
            cur.execute(
                "DELETE FROM sutra_symbols WHERE moniker = ANY(%s)",
                (monikers,),
            )

    def delete_relationships_from(self, source_monikers: list[str]) -> None:
        """Delete all OUTBOUND relationships whose source_id is in the list."""
        if not source_monikers:
            return
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM sutra_relationships WHERE source_id = ANY(%s)",
                (source_monikers,),
            )

    def write_symbol_direct(
        self,
        sym: Symbol,
        repo_name: str,
        indexed_at: str,
    ) -> None:
        """Upsert a single symbol.  Used by the incremental updater."""
        row = _symbol_row(sym, repo_name, indexed_at)
        with self._conn.cursor() as cur:
            cur.execute(
                _SYMBOL_UPSERT_SQL,
                tuple(row[col] for col in _SYMBOL_COLUMNS),
            )

    def write_relationships_direct(self, relationships: list[Relationship]) -> int:
        """
        Bulk-insert relationships in one transaction.

        Returns the count of skipped (unresolved) relationships for logging.
        """
        with self._conn.cursor() as cur:
            cur.execute("BEGIN")
            try:
                skipped = self._write_relationships(cur, relationships)
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise
        return skipped

    def close(self) -> None:
        """Close the psycopg2 connection.  Idempotent."""
        if not self._conn.closed:
            self._conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _delete_repo_symbols(self, cur: Any, repo_name: str) -> None:
        """
        Delete every symbol for a repo, plus every relationship those
        symbols appear in.  Used by replace=True before re-indexing.

        Two-step delete instead of relying on FK CASCADE because
        sutra_relationships has no FK to sutra_symbols (see migration 003
        rationale).  The order matters: edges first, then symbols.
        """
        cur.execute(
            """
            DELETE FROM sutra_relationships
            WHERE source_id IN (SELECT moniker FROM sutra_symbols WHERE repo_name = %s)
               OR target_id IN (SELECT moniker FROM sutra_symbols WHERE repo_name = %s)
            """,
            (repo_name, repo_name),
        )
        cur.execute(
            "DELETE FROM sutra_symbols WHERE repo_name = %s",
            (repo_name,),
        )

    def _write_relationships(
        self,
        cur: Any,
        relationships: list[Relationship],
    ) -> int:
        """
        Insert resolved relationships, skipping unresolved ones.

        Returns the count of skipped (unresolved) relationships.

        Rows are inserted with ON CONFLICT DO NOTHING — the composite PK
        (source_id, target_id, kind) handles dedup.  Repeat edges between
        the same pair under the same kind are silently merged.
        """
        skipped = 0
        rows: list[tuple[str, str, str, bool, str]] = []
        for rel in relationships:
            if not rel.is_resolved or rel.target_id is None:
                skipped += 1
                continue
            # metadata is stored as JSONB; psycopg2 needs a JSON-encoded string
            import json  # noqa: PLC0415
            rows.append((
                rel.source_id,
                rel.target_id,
                rel.kind.value,
                rel.is_resolved,
                json.dumps(rel.metadata or {}),
            ))

        if rows:
            cur.executemany(
                """
                INSERT INTO sutra_relationships
                    (source_id, target_id, kind, is_resolved, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source_id, target_id, kind) DO UPDATE SET
                    is_resolved = EXCLUDED.is_resolved,
                    metadata = EXCLUDED.metadata
                """,
                rows,
            )

        return skipped
