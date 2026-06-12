"""
PGVectorStore tests + end-to-end Indexer integration with the SQL graph backend.

Migrated from tests/test_graph_writers.py when PR 2 swapped Apache AGE for
plain Postgres SQL tables.  Three former test classes consolidated:
    * TestPGVectorStore — kept verbatim (pgvector is unchanged)
    * TestAGEWriter      — dropped (replaced by tests/test_sql_graph.py)
    * TestCypherVal      — dropped (Cypher is gone)
    * TestIndexerIntegration — adapted to use SqlGraphWriter

Gating
------
    SUTRA_PG_URL=postgresql://postgres:postgers@localhost:5433/postgres \
        pytest tests/test_pgvector_and_integration.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import psycopg2
import pytest

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

_PG_URL = os.getenv("SUTRA_PG_URL", "")
pytestmark = pytest.mark.skipif(
    not _PG_URL,
    reason="SUTRA_PG_URL env var not set — skipping integration tests",
)

_TEST_TABLE = "_sutra_test_embeddings"
_TEST_DIMS = 4

# ---------------------------------------------------------------------------
# Imports (after gate)
# ---------------------------------------------------------------------------

from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.graph.pgvector_store import PGVectorStore
from sutra.core.graph.sql_writer import SqlGraphWriter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter


# ---------------------------------------------------------------------------
# Teardown helpers (NOT methods on production classes)
# ---------------------------------------------------------------------------

def _drop_table(pg_url: str, table_name: str) -> None:
    """Drop the pgvector embeddings table.  DESTRUCTIVE — test use only."""
    conn = psycopg2.connect(pg_url)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()
    conn.close()


def _drop_sql_graph_tables(pg_url: str) -> None:
    """Drop the SQL graph tables.  DESTRUCTIVE — test use only."""
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS sutra_relationships")
        cur.execute("DROP TABLE IF EXISTS sutra_symbols")
        cur.execute("DROP TABLE IF EXISTS sutra_repositories")
    conn.close()


# ---------------------------------------------------------------------------
# Module-level setup / teardown
# ---------------------------------------------------------------------------

def setup_module(module: object) -> None:
    """Create the pgvector test table once for the whole session."""
    store = PGVectorStore(_PG_URL, dims=_TEST_DIMS, table_name=_TEST_TABLE)
    store.setup()
    store.close()


def teardown_module(module: object) -> None:
    _drop_table(_PG_URL, _TEST_TABLE)


# ===========================================================================
# TestPGVectorStore — verbatim from former test_graph_writers.py
# ===========================================================================


class TestPGVectorStore:
    """Tests for the pgvector embedding store."""

    def _store(self) -> PGVectorStore:
        return PGVectorStore(_PG_URL, dims=_TEST_DIMS, table_name=_TEST_TABLE)

    def test_setup_is_idempotent(self) -> None:
        """Calling setup() twice must not raise."""
        with self._store() as s:
            s.setup()  # already ran in setup_module; call again

    def test_dimensions_property(self) -> None:
        with self._store() as s:
            assert s.dimensions == _TEST_DIMS

    def test_write_and_cosine_search(self) -> None:
        """Write 3 vectors; search with one as query — must return itself first."""
        monikers = ["sym_vec_a", "sym_vec_b", "sym_vec_c"]
        vecs = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float32,
        )

        with self._store() as s:
            s.write(monikers, vecs)
            results = s.search(vecs[0], k=3)

        assert results[0][0] == "sym_vec_a"
        assert abs(results[0][1] - 1.0) < 1e-6  # similarity ≈ 1 for exact match

    def test_upsert_overwrites_old_vector(self) -> None:
        """Write then overwrite a vector; search must reflect the new value."""
        v_old = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        v_new = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        with self._store() as s:
            s.write(["sym_upsert"], v_old)
            s.write(["sym_upsert"], v_new)
            results = s.search(v_new[0], k=1)

        assert results[0][0] == "sym_upsert"
        assert abs(results[0][1] - 1.0) < 1e-6

    def test_dimension_mismatch_raises(self) -> None:
        """Vectors of wrong dims must raise ValueError with a clear message."""
        wrong_vecs = np.ones((2, _TEST_DIMS + 1), dtype=np.float32)
        with self._store() as s:
            with pytest.raises(ValueError, match="dimension mismatch"):
                s.write(["a", "b"], wrong_vecs)

    def test_length_mismatch_raises(self) -> None:
        """Moniker count != row count must raise ValueError."""
        vecs = np.ones((3, _TEST_DIMS), dtype=np.float32)
        with self._store() as s:
            with pytest.raises(ValueError, match="Length mismatch"):
                s.write(["only_one"], vecs)

    def test_write_empty_is_noop(self) -> None:
        """Writing empty monikers/vectors must not raise or write anything."""
        vecs = np.empty((0, _TEST_DIMS), dtype=np.float32)
        with self._store() as s:
            s.write([], vecs)  # must not raise

    def test_close_marks_connection_closed(self) -> None:
        """close() must close the psycopg2 connection."""
        s = self._store()
        assert s._conn.closed == 0
        s.close()
        assert s._conn.closed != 0

    def test_context_manager_closes_on_exit(self) -> None:
        """Context manager __exit__ must close the connection."""
        with self._store() as s:
            conn = s._conn
        assert conn.closed != 0


# ===========================================================================
# TestIndexerIntegration — adapted to use SqlGraphWriter
# ===========================================================================


class TestIndexerIntegration:
    """
    End-to-end test: Indexer with SqlGraphWriter + PGVectorStore.
    Uses the checked-in fixture repo so no live files needed.
    """

    _FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_python_repo"
    _FIXTURE_URL = "https://github.com/ritikk112/sutra"
    _INTEG_TABLE = "_sutra_integ_test_embeddings"
    _INTEG_DIMS = 384

    def setup_method(self) -> None:
        """Create fresh SQL graph tables + pgvector table before each test."""
        _drop_sql_graph_tables(_PG_URL)
        writer = SqlGraphWriter(_PG_URL)
        writer.setup()
        writer.close()
        store = PGVectorStore(
            _PG_URL, dims=self._INTEG_DIMS, table_name=self._INTEG_TABLE
        )
        store.setup()
        store.close()

    def teardown_method(self) -> None:
        """Drop SQL graph tables + pgvector table after each test."""
        _drop_sql_graph_tables(_PG_URL)
        _drop_table(_PG_URL, self._INTEG_TABLE)

    def test_indexer_writes_to_sql_graph_and_pgvector(self, tmp_path: Path) -> None:
        """
        Full pipeline: Indexer writes JSON + SQL graph rows + pgvector embeddings.
        Symbol count in sutra_symbols must match IndexResult.symbols count.
        """
        graph_writer = SqlGraphWriter(_PG_URL)
        pgvec_store = PGVectorStore(
            _PG_URL, dims=self._INTEG_DIMS, table_name=self._INTEG_TABLE
        )

        indexer = Indexer(
            adapters={"python": PythonAdapter()},
            exporter=JsonGraphExporter(),
            embedder=FixtureEmbedder(dims=self._INTEG_DIMS),
            graph_writer=graph_writer,
            pgvector_store=pgvec_store,
        )

        result = indexer.index(
            root=self._FIXTURE_REPO,
            repo_url=self._FIXTURE_URL,
            output_dir=tmp_path,
        )

        graph_writer.close()
        pgvec_store.close()

        # Symbol count in SQL must equal symbols in IndexResult
        conn = psycopg2.connect(_PG_URL)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM sutra_symbols")
            sql_symbol_count = cur.fetchone()[0]
        conn.close()
        assert sql_symbol_count == len(result.symbols), (
            f"SQL has {sql_symbol_count} symbol rows; "
            f"IndexResult has {len(result.symbols)}"
        )

        # pgvector row count must equal embedding count from graph.json
        import json
        graph = json.loads((tmp_path / "graph.json").read_text())
        expected_embeddings = graph["embeddings"]["count"]

        conn = psycopg2.connect(_PG_URL)
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self._INTEG_TABLE}")
            pgvec_count = cur.fetchone()[0]
        conn.close()

        assert pgvec_count == expected_embeddings, (
            f"pgvector has {pgvec_count} rows; "
            f"graph.json reports {expected_embeddings}"
        )

    def test_indexer_works_without_writers(self, tmp_path: Path) -> None:
        """Indexer without DB writers must produce graph.json + .npy."""
        indexer = Indexer(
            adapters={"python": PythonAdapter()},
            exporter=JsonGraphExporter(),
            embedder=FixtureEmbedder(dims=self._INTEG_DIMS),
            # No graph_writer or pgvector_store — both default to None
        )
        result = indexer.index(
            root=self._FIXTURE_REPO,
            repo_url=self._FIXTURE_URL,
            output_dir=tmp_path,
        )
        assert len(result.symbols) > 0
        assert (tmp_path / "graph.json").exists()
