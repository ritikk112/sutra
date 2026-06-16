"""
Index state store: per-repo metadata (last commit SHA) used by the
incremental update pipeline.

Concrete class, NOT an ABC.  Per planner Round 5 review:

  > DO NOT make IndexStateStore an ABC.  It has one method group
  > (get_last_commit_sha, update_commit_sha) and the only conceivable
  > second impl is "in-memory for tests."  Use a concrete class and write
  > a tiny FakeIndexStateStore in tests/conftest.py.

The store reads from / writes to the sutra_repositories table created by
migration 001.  No extra DDL.

INVARIANT: update_commit_sha is called LAST in the incremental updater
after every file has been successfully processed.  A failure before that
call leaves the old SHA in place, and a re-run will diff the same range
again (idempotent recovery).  This contract is enforced by the caller,
not by the store.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import psycopg2


class SqlIndexStateStore:
    """Per-repo state (commit SHA, last-indexed timestamp) in Postgres."""

    def __init__(self, conn_str: str) -> None:
        self._conn = psycopg2.connect(conn_str)
        self._conn.autocommit = True

    def get_last_commit_sha(self, repo_name: str) -> Optional[str]:
        """
        Return the SHA the repo was last indexed at, or None if the row
        doesn't exist (first run) or the SHA hasn't been patched yet
        (interrupted run — recovery path).
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT last_commit_sha FROM sutra_repositories WHERE name = %s",
                (repo_name,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        sha = row[0]
        return sha if sha else None

    def update_commit_sha(self, repo_name: str, new_sha: str) -> None:
        """
        Atomically set last_commit_sha + indexed_at on the repo row.

        Uses INSERT ... ON CONFLICT so callers don't have to ensure the
        repository row exists first.  Mirrors AGEWriter.update_commit_sha
        which also handled both insert + update via MERGE.
        """
        indexed_at = datetime.now(timezone.utc).isoformat()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sutra_repositories (name, url, last_commit_sha, indexed_at)
                VALUES (%s, '', %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    last_commit_sha = EXCLUDED.last_commit_sha,
                    indexed_at = EXCLUDED.indexed_at
                """,
                (repo_name, new_sha, indexed_at),
            )

    def close(self) -> None:
        if not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> "SqlIndexStateStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
