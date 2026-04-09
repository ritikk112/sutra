"""
AGE graph writer for Sutra.

Overview
--------
AGEWriter upserts symbol nodes and resolved relationship edges into an
Apache AGE graph.  It is an optional sink — Indexer accepts it as a
constructor parameter and calls write_repository() after the JSON exporter.

Connection lifecycle
--------------------
    writer = AGEWriter(conn_str, graph_name)
    writer.setup()                          # idempotent DDL
    writer.write_repository(result)         # upserts symbols + relationships
    writer.close()

Or as a context manager:

    with AGEWriter(conn_str) as writer:
        writer.setup()
        writer.write_repository(result)

Re-indexing the same repo
--------------------------
    writer.write_repository(result, replace=True)

When replace=True, all Symbol nodes BELONGS_TO the repo are DETACH DELETEd
before new symbols are written.  This prevents ghost symbols from accumulating
across re-index runs.  Without replace=True, existing symbols are upserted
(MERGE + SET) — safe for additive changes, leaves deleted symbols as ghosts.

Unresolved relationships
------------------------
Relationships with target_id=None are skipped — they live in graph.json only.
AGE edges require both endpoints to exist; placeholder nodes would pollute the
graph.  When Phase 2's LSP resolver promotes unresolved → resolved, a re-run
of write_repository() will materialize them in AGE.  The count of skipped
relationships is logged to stderr.

Parameter injection (important)
---------------------------------
psycopg2 2.9.11 does client-side parameter interpolation: %s is replaced by a
quoted string literal before Postgres parses the query.  AGE 1.6.0 requires the
third argument to cypher() to be a T_Param node (i.e. a real $1 in the parse
tree) and rejects quoted literals with "third argument must be a parameter".

Workaround: build Cypher strings with _cypher_val() which safely escapes
string content (backslashes and single quotes).  All values passing through
_cypher_val() come from internal parsed symbols (validated monikers, typed
dataclass fields) — never from raw user input.

Transaction model
-----------------
autocommit=True throughout; write_repository() uses explicit BEGIN/COMMIT so
all symbol writes are atomic.  Rollback on any error.

TODO (Priority 11): batch into transactions of N symbols for repos > ~50K
symbols to avoid long lock durations and Postgres transaction memory pressure.

Connection failure
------------------
On psycopg2 connection drop, the transaction rolls back automatically.  The
writer is then in an unusable state — re-construct it; do not attempt recovery.
Credentials in conn_str are never logged (use conn.dsn for a safe repr).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any, Optional, Union

import psycopg2
import psycopg2.extensions

from sutra.core.extractor.base import (
    ClassSymbol,
    FunctionSymbol,
    IndexResult,
    MethodSymbol,
    ModuleSymbol,
    Symbol,
    VariableSymbol,
    Visibility,
)
from sutra.core.graph.schema import (
    BELONGS_TO_LABEL,
    DEFAULT_GRAPH_NAME,
    REPO_LABEL,
    SYMBOL_LABEL,
)


# ---------------------------------------------------------------------------
# Cypher value escaping
# ---------------------------------------------------------------------------

def _cypher_val(v: Any) -> str:
    """
    Render a Python value as a safe Cypher literal.

    Supports: None → null, bool, int, float, str.
    Strings are single-quoted with backslashes and single quotes escaped.
    Never call this with data from outside the Sutra pipeline (user form input,
    external HTTP responses, etc.) — it is designed for internal trusted data.
    """
    if v is None:
        return "null"
    if isinstance(v, bool):
        # bool before int: bool is a subclass of int in Python
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(v)
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    raise TypeError(f"Unsupported type for Cypher value: {type(v).__name__!r}")


def _cypher_map(d: dict[str, Any]) -> str:
    """
    Render a dict as a Cypher map literal {k: v, ...}.
    Keys with None values are omitted (keeps node properties clean).
    """
    parts = [
        f"{key}: {_cypher_val(val)}"
        for key, val in d.items()
        if val is not None
    ]
    return "{" + ", ".join(parts) + "}"


# ---------------------------------------------------------------------------
# Symbol property extraction
# ---------------------------------------------------------------------------

def _symbol_props(sym: Symbol, indexed_at: str) -> dict[str, Any]:
    """
    Build the AGE node property dict for any symbol.

    Location: only line_start/line_end are stored (byte offsets are not useful
    for graph queries and would bloat node size).
    Lists (decorators, parameters): omitted — too complex for agtype;
    signature captures parameters textually for FunctionSymbol/MethodSymbol.
    base_classes: stored as comma-joined string.
    Docstrings: truncated to 500 chars (bodies stay in source files).
    """
    # Check isinstance(MethodSymbol) BEFORE isinstance(FunctionSymbol) because
    # MethodSymbol is a subclass of FunctionSymbol.
    if isinstance(sym, MethodSymbol):
        kind = "method"
    elif isinstance(sym, FunctionSymbol):
        kind = "function"
    elif isinstance(sym, ClassSymbol):
        kind = "class"
    elif isinstance(sym, VariableSymbol):
        kind = "variable"
    elif isinstance(sym, ModuleSymbol):
        kind = "module"
    else:
        kind = "unknown"

    vis = sym.visibility.value if isinstance(sym.visibility, Visibility) else str(sym.visibility)

    props: dict[str, Any] = {
        "kind": kind,
        "name": sym.name,
        "qualified_name": sym.qualified_name,
        "file_path": sym.file_path,
        "language": sym.language,
        "visibility": vis,
        "is_exported": sym.is_exported,
        "body_hash": sym.body_hash,
        "line_start": sym.location.line_start,
        "line_end": sym.location.line_end,
        "indexed_at": indexed_at,
    }

    if isinstance(sym, MethodSymbol):
        props["signature"] = sym.signature if sym.signature else None
        props["return_type"] = sym.return_type
        props["is_async"] = sym.is_async
        props["complexity"] = sym.complexity
        props["is_static"] = sym.is_static
        props["is_constructor"] = sym.is_constructor
        props["receiver_kind"] = sym.receiver_kind  # None filtered out by _cypher_map
        props["docstring"] = sym.docstring[:500] if sym.docstring else None
    elif isinstance(sym, FunctionSymbol):
        props["signature"] = sym.signature if sym.signature else None
        props["return_type"] = sym.return_type
        props["is_async"] = sym.is_async
        props["complexity"] = sym.complexity
        props["docstring"] = sym.docstring[:500] if sym.docstring else None
    elif isinstance(sym, ClassSymbol):
        props["is_abstract"] = sym.is_abstract
        props["base_classes"] = ", ".join(sym.base_classes) if sym.base_classes else None
        props["docstring"] = sym.docstring[:500] if sym.docstring else None
    elif isinstance(sym, VariableSymbol):
        props["type_annotation"] = sym.type_annotation
        props["is_constant"] = sym.is_constant
    elif isinstance(sym, ModuleSymbol):
        props["docstring"] = sym.docstring[:500] if sym.docstring else None

    return props


# ---------------------------------------------------------------------------
# AGEWriter
# ---------------------------------------------------------------------------

class AGEWriter:
    """
    Writes Sutra IndexResult data to an Apache AGE graph.

    Constructor parameters
    ----------------------
    conn_str : str
        psycopg2 connection string.  Credentials are never logged.
    graph_name : str
        Name of the AGE graph to write to.  Defaults to DEFAULT_GRAPH_NAME.
    """

    def __init__(
        self,
        conn_str: str,
        graph_name: str = DEFAULT_GRAPH_NAME,
    ) -> None:
        self._graph_name = graph_name
        # Connect; credentials in conn_str are never logged — use conn.dsn for repr
        self._conn = psycopg2.connect(conn_str)
        self._conn.autocommit = True
        # Configure session: LOAD 'age' and search_path are session-level settings
        # that persist for the lifetime of this connection.
        with self._conn.cursor() as cur:
            cur.execute("LOAD 'age'")
            cur.execute("SET search_path = ag_catalog, \"$user\", public")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Idempotent setup: ensures the AGE extension and graph exist.
        Safe to call multiple times — creates only what is missing.
        """
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS age")
            # Check existence before calling create_graph — avoids error on re-run
            cur.execute(
                "SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s",
                (self._graph_name,),
            )
            if cur.fetchone()[0] == 0:
                cur.execute("SELECT create_graph(%s)", (self._graph_name,))

    def write_repository(
        self,
        result: IndexResult,
        replace: bool = False,
    ) -> None:
        """
        Upsert all symbols and resolved relationships from an IndexResult.

        Parameters
        ----------
        result : IndexResult
            The assembled output of an indexing run.
        replace : bool
            When True, DETACH DELETE all existing Symbol nodes for this repo
            before writing.  Prevents ghost symbols from accumulating across
            re-index runs.  Default False (additive upsert).

        Relationship handling
        ---------------------
        Only relationships with is_resolved=True and target_id set are written
        to AGE.  Unresolved relationships (target_id=None) are skipped — they
        live in graph.json only.  The count of skipped rels is logged to stderr.

        If MATCH for a relationship endpoint finds no node (e.g. the target is
        in an external package), the edge is silently not written.  This is not
        an error; it is the expected Phase 1 behavior for external references.
        """
        repo_name = result.repository.name
        indexed_at = result.indexed_at.isoformat()

        with self._conn.cursor() as cur:
            cur.execute("BEGIN")
            try:
                self._write_repo_node(cur, result, indexed_at)

                if replace:
                    self._delete_repo_symbols(cur, repo_name)

                for sym in result.symbols:
                    self._write_symbol(cur, sym, repo_name, indexed_at)

                skipped = self._write_relationships(cur, result)
                cur.execute("COMMIT")

            except Exception:
                cur.execute("ROLLBACK")
                raise

        if skipped:
            print(
                f"[AGEWriter] Skipped {skipped} unresolved relationships "
                "(will be written after Phase 2 resolution)",
                file=sys.stderr,
            )

    def close(self) -> None:
        """Close the underlying psycopg2 connection."""
        if not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> "AGEWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal write helpers
    # ------------------------------------------------------------------

    def _run(self, cur: Any, cypher: str, return_spec: str = "r agtype") -> None:
        """Execute a Cypher query against this writer's graph."""
        sql = (
            f"SELECT * FROM cypher('{self._graph_name}', $${cypher}$$)"
            f" AS ({return_spec})"
        )
        cur.execute(sql)

    def _write_repo_node(
        self,
        cur: Any,
        result: IndexResult,
        indexed_at: str,
    ) -> None:
        """Upsert the Repository vertex."""
        repo = result.repository
        self._run(cur, f"""
            MERGE (r:{REPO_LABEL} {{name: {_cypher_val(repo.name)}}})
            SET r += {{
                url: {_cypher_val(repo.url)},
                commit_sha: {_cypher_val(result.commit_hash)},
                indexed_at: {_cypher_val(indexed_at)}
            }}
            RETURN r
        """)

    def _delete_repo_symbols(self, cur: Any, repo_name: str) -> None:
        """
        DETACH DELETE all Symbol nodes belonging to repo_name.
        Used by replace=True to clear stale symbols before re-indexing.
        """
        self._run(cur, f"""
            MATCH (n:{SYMBOL_LABEL})-[:{BELONGS_TO_LABEL}]->
                  (:{REPO_LABEL} {{name: {_cypher_val(repo_name)}}})
            DETACH DELETE n
            RETURN count(n)
        """, return_spec="cnt agtype")

    def _write_symbol(
        self,
        cur: Any,
        sym: Symbol,
        repo_name: str,
        indexed_at: str,
    ) -> None:
        """Upsert a Symbol node and its BELONGS_TO edge to the Repository."""
        props = _symbol_props(sym, indexed_at)
        props_map = _cypher_map(props)
        # Combine symbol MERGE + BELONGS_TO MERGE in a single Cypher query
        self._run(cur, f"""
            MERGE (n:{SYMBOL_LABEL} {{id: {_cypher_val(sym.id)}}})
            SET n += {props_map}
            WITH n
            MATCH (r:{REPO_LABEL} {{name: {_cypher_val(repo_name)}}})
            MERGE (n)-[:{BELONGS_TO_LABEL}]->(r)
            RETURN n
        """)

    def _write_relationships(self, cur: Any, result: IndexResult) -> int:
        """
        Write resolved relationships as edges.

        Returns the count of skipped unresolved relationships for logging.
        Edge label = RelationKind.value.upper() (e.g. "calls" → "CALLS").
        The label comes from our internal enum — safe to embed directly.
        """
        skipped = 0
        for rel in result.relationships:
            if not rel.is_resolved or rel.target_id is None:
                skipped += 1
                continue

            edge_label = rel.kind.value.upper()  # internal enum value — safe
            # If either endpoint is missing from the graph (external package),
            # MATCH returns nothing and no edge is created — not an error.
            self._run(cur, f"""
                MATCH (a:{SYMBOL_LABEL} {{id: {_cypher_val(rel.source_id)}}}),
                      (b:{SYMBOL_LABEL} {{id: {_cypher_val(rel.target_id)}}})
                MERGE (a)-[:{edge_label}]->(b)
                RETURN a
            """)

        return skipped
