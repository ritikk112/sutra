"""
pgvector embedding store for Sutra.

PGVectorStore upserts embedding vectors (keyed by symbol moniker) into a
PostgreSQL table using the pgvector extension.  It is an optional sink —
Indexer accepts it as a constructor parameter alongside AGEWriter.

Write ordering
--------------
In Indexer.index(), pgvector is written BEFORE AGE.  If AGE fails, the
pgvector data is orphaned (keyed by moniker, not by AGE node ID) but
harmless — a re-run overwrites it.  The reverse order is worse: AGE nodes
pointing at missing vectors.

Dimension enforcement
---------------------
`dims` is a required constructor argument (no default).  The table schema
is `vector(dims)` — wrong dimensions cause pgvector to reject the INSERT with
a confusing error.  PGVectorStore validates `vectors.shape[1] == dims` at
write time and raises a clear ValueError before touching the database.

In the full pipeline, `dims` must come from `embedder.dimensions` — the
embedder and the store must agree on vector size.  Priority 11 (incremental
pipeline) will wire this via config; for now, callers supply it explicitly.

Index type
----------
HNSW index (pgvector 0.5+, confirmed available at 0.7.4).  No minimum row
count for training (unlike IVFFlat).  Created in setup(); if the index
already exists, `CREATE INDEX IF NOT EXISTS` is a no-op.

Connection lifecycle
--------------------
Same pattern as AGEWriter: one psycopg2 connection per instance, explicit
close() or context manager.  Credentials in conn_str are never logged.

    store = PGVectorStore(conn_str, dims=384)
    store.setup()
    store.write(monikers, vectors)
    results = store.search(query_vec, k=5)
    store.close()

Or:

    with PGVectorStore(conn_str, dims=384) as store:
        store.setup()
        store.write(monikers, vectors)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import psycopg2

from sutra.core.graph.schema import DEFAULT_TABLE_NAME


class PGVectorStore:
    """
    Stores and retrieves embedding vectors in PostgreSQL via pgvector.

    Parameters
    ----------
    conn_str : str
        psycopg2 connection string.  Credentials are never logged.
    dims : int
        Embedding dimensionality.  Required — no default.  Must match the
        embedder's output dimension (embedder.dimensions).
    table_name : str
        Name of the embeddings table.  Defaults to DEFAULT_TABLE_NAME.
    """

    def __init__(
        self,
        conn_str: str,
        dims: int,
        table_name: str = DEFAULT_TABLE_NAME,
    ) -> None:
        self._dims = dims
        self._table = table_name
        # Connect; credentials in conn_str are never logged
        self._conn = psycopg2.connect(conn_str)

    @property
    def dimensions(self) -> int:
        """The vector dimensionality this store was configured with."""
        return self._dims

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Idempotent setup: ensure the pgvector extension, embeddings table,
        and HNSW index all exist.  Safe to call multiple times.
        """
        with self._conn:
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        moniker TEXT PRIMARY KEY,
                        embedding vector({self._dims}),
                        indexed_at TIMESTAMPTZ DEFAULT now()
                    )
                """)
                # HNSW: no minimum row count (unlike IVFFlat), better recall,
                # modern recommendation for pgvector 0.5+.
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx
                    ON {self._table}
                    USING hnsw (embedding vector_cosine_ops)
                """)

    def write(self, monikers: list[str], vectors: np.ndarray) -> None:
        """
        Upsert embeddings for a list of monikers.

        Parameters
        ----------
        monikers : list[str]
            Symbol monikers.  One-to-one with `vectors` rows.
        vectors : np.ndarray
            Shape (N, dims), dtype float32.  N must equal len(monikers).

        Raises
        ------
        ValueError
            If vectors.shape[1] != self.dims, or len(monikers) != vectors.shape[0].
        """
        if not monikers:
            return

        if vectors.shape[1] != self._dims:
            raise ValueError(
                f"Vector dimension mismatch: store configured for {self._dims} dims, "
                f"got vectors with {vectors.shape[1]} dims.  "
                f"Ensure embedder.dimensions matches PGVectorStore dims."
            )
        if len(monikers) != vectors.shape[0]:
            raise ValueError(
                f"Length mismatch: {len(monikers)} monikers vs {vectors.shape[0]} vectors"
            )

        upsert_sql = (
            f"INSERT INTO {self._table} (moniker, embedding) VALUES (%s, %s::vector) "
            f"ON CONFLICT (moniker) DO UPDATE "
            f"SET embedding = EXCLUDED.embedding, indexed_at = now()"
        )
        with self._conn:
            with self._conn.cursor() as cur:
                for moniker, vec in zip(monikers, vectors):
                    cur.execute(upsert_sql, (moniker, vec.tolist()))

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Find the k nearest embeddings by cosine distance.

        Parameters
        ----------
        query_vec : np.ndarray
            Shape (dims,), dtype float32.
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            List of (moniker, cosine_distance) sorted nearest-first (distance 0
            means identical vectors; distance 2 means maximally dissimilar).
        """
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT moniker, embedding <=> %s::vector AS dist "
                f"FROM {self._table} "
                f"ORDER BY dist "
                f"LIMIT %s",
                (query_vec.tolist(), k),
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]

    def close(self) -> None:
        """Close the underlying psycopg2 connection."""
        if not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> "PGVectorStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
