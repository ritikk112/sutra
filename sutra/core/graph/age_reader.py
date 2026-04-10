"""
AGE read-only query interface for Sutra.

AGEReader provides read queries against the AGE graph — separated from AGEWriter
because read and write concerns are orthogonal and mixing them in one class
leads to a bloated interface.

Connection lifecycle
--------------------
Same pattern as AGEWriter: one psycopg2 connection per instance, autocommit=True,
explicit close() or context manager.  Credentials in conn_str are never logged.

    reader = AGEReader(conn_str, graph_name)
    sha = reader.get_last_commit_sha("my-repo")
    symbols = reader.get_symbols_for_files("my-repo", ["src/foo.py"])
    reader.close()

Or:

    with AGEReader(conn_str) as reader:
        sha = reader.get_last_commit_sha("my-repo")

Cypher building
---------------
Same _cypher_val() / _cypher_list() helpers as the writer — safe string
formatting because AGE 1.6.0 rejects psycopg2 %s interpolation.

Return format for get_symbols_for_files
----------------------------------------
Returns {moniker: body_hash} — sufficient for the per-file diff logic in
IncrementalUpdater.  The updater only needs to know which monikers existed and
whether their body changed; full symbol metadata is re-extracted from source.
"""
from __future__ import annotations

import sys
from typing import Any, Optional

import psycopg2
import psycopg2.extensions

from sutra.core.graph.schema import DEFAULT_GRAPH_NAME, REPO_LABEL, SYMBOL_LABEL
from sutra.core.graph.postgres_age import _cypher_val


class AGEReader:
    """
    Read-only interface for querying the AGE graph.

    Parameters
    ----------
    conn_str : str
        psycopg2 connection string.  Credentials are never logged.
    graph_name : str
        AGE graph name.  Must match the graph used by AGEWriter.
    """

    def __init__(
        self,
        conn_str: str,
        graph_name: str = DEFAULT_GRAPH_NAME,
    ) -> None:
        self._graph_name = graph_name
        self._conn = psycopg2.connect(conn_str)
        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            cur.execute("LOAD 'age'")
            cur.execute("SET search_path = ag_catalog, \"$user\", public")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_last_commit_sha(self, repo_name: str) -> Optional[str]:
        """
        Return the commit SHA the repo was last indexed at, or None if the
        Repository node does not exist yet (first run).

        Parameters
        ----------
        repo_name : str
            The repo name as stored in the Repository node (derived from URL).
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM cypher('{self._graph_name}', $$"
                    f"  MATCH (r:{REPO_LABEL} {{name: {_cypher_val(repo_name)}}})"
                    f"  RETURN r.commit_sha"
                    f"$$) AS (sha agtype)"
                )
                row = cur.fetchone()
        except Exception as exc:  # noqa: BLE001
            # Graph may not exist yet (first run before setup).
            print(
                f"[AGEReader] get_last_commit_sha failed: {exc}",
                file=sys.stderr,
            )
            return None

        if row is None:
            return None

        raw = row[0]
        if raw is None:
            return None

        # AGE returns agtype as a JSON-quoted string e.g. '"abc123"' or null
        sha = str(raw).strip().strip('"')
        return sha if sha and sha != "null" else None

    def get_symbols_for_files(
        self,
        repo_name: str,
        file_paths: list[str],
    ) -> dict[str, str]:
        """
        Return {moniker: body_hash} for all symbols belonging to the given files
        in the specified repo.

        Used by IncrementalUpdater to compute the per-file diff:
            - old monikers not in new extraction → deleted
            - same moniker, different body_hash → changed
            - same moniker, same body_hash → unchanged (skip)

        Parameters
        ----------
        repo_name : str
            The repo name as stored in the Repository node.
        file_paths : list[str]
            Repo-relative file paths (POSIX separators).

        Returns
        -------
        dict[str, str]
            Keys are symbol monikers (the `id` property); values are body_hash
            strings.  Empty dict if no files provided or no symbols found.
        """
        if not file_paths:
            return {}

        # Build an inline Cypher list of quoted file paths.
        paths_list = "[" + ", ".join(_cypher_val(p) for p in file_paths) + "]"

        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM cypher('{self._graph_name}', $$"
                    f"  MATCH (n:{SYMBOL_LABEL})"
                    f"  WHERE n.file_path IN {paths_list}"
                    f"  RETURN n.id, n.body_hash"
                    f"$$) AS (moniker agtype, body_hash agtype)"
                )
                rows = cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[AGEReader] get_symbols_for_files failed: {exc}",
                file=sys.stderr,
            )
            return {}

        result: dict[str, str] = {}
        for row in rows:
            moniker_raw, bh_raw = row
            if moniker_raw is None:
                continue
            moniker = str(moniker_raw).strip().strip('"')
            body_hash = str(bh_raw).strip().strip('"') if bh_raw is not None else ""
            if moniker and moniker != "null":
                result[moniker] = body_hash

        return result

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying psycopg2 connection."""
        if not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> "AGEReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
