"""
Plain-SQL incremental reader.  Replaces age_reader.AGEReader (read path).

Narrow on purpose: the only verb is get_symbols_for_files, used by
IncrementalUpdater to compute the per-file four-way diff.

State queries (get_last_commit_sha) live on SqlIndexStateStore — separate
class because it has different lifecycle (a tiny state store, not a
read-heavy graph reader) and the planner Round 5 review explicitly
recommended splitting them.

Connection lifecycle
--------------------
    reader = SqlIncrementalReader(conn_str)
    state = reader.get_symbols_for_files("my-repo", ["src/foo.py"])
    reader.close()

Or as a context manager.  autocommit=True; reads don't open transactions.
"""
from __future__ import annotations

import psycopg2

from sutra.core.graph.base import IncrementalReader


class SqlIncrementalReader(IncrementalReader):
    """Plain-SQL implementation of IncrementalReader."""

    def __init__(self, conn_str: str) -> None:
        self._conn = psycopg2.connect(conn_str)
        self._conn.autocommit = True

    def get_symbols_for_files(
        self,
        repo_name: str,
        file_paths: list[str],
    ) -> dict[str, str]:
        """
        Return {moniker: body_hash} for symbols belonging to the given files.

        repo_name is currently unused in the SQL impl — file_path is globally
        unique within the repo and the query joins on it directly.  The
        parameter is kept for parity with AGEReader's signature so the
        IncrementalUpdater swap in PR 2 is mechanical.

        Empty file_paths → empty dict (skip the round-trip).
        """
        if not file_paths:
            return {}

        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT moniker, body_hash
                FROM sutra_symbols
                WHERE file_path = ANY(%s)
                  AND repo_name = %s
                """,
                (file_paths, repo_name) if repo_name else (file_paths, ""),
            )
            # When repo_name is empty (legacy IncrementalUpdater behaviour for
            # deleted-files lookup), fall back to file_path-only match.
            rows = cur.fetchall()
            if not rows and not repo_name:
                cur.execute(
                    """
                    SELECT moniker, body_hash
                    FROM sutra_symbols
                    WHERE file_path = ANY(%s)
                    """,
                    (file_paths,),
                )
                rows = cur.fetchall()

        return {moniker: body_hash for moniker, body_hash in rows}

    def close(self) -> None:
        if not self._conn.closed:
            self._conn.close()
