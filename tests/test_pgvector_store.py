"""
PGVectorStore unit and integration tests.

Gated on SUTRA_PG_URL — skipped entirely if not set.
All DB tests use a dedicated table name distinct from the production default
to prevent accidental data loss.  The table is dropped in teardown.

Usage:
    SUTRA_PG_URL=postgresql://postgres:postgers@localhost:5433/postgres \
        pytest tests/test_pgvector_store.py -v
"""
from __future__ import annotations

import os

import numpy as np
import psycopg2
import pytest

_PG_URL = os.getenv("SUTRA_PG_URL", "")
pytestmark = pytest.mark.skipif(
    not _PG_URL,
    reason="SUTRA_PG_URL env var not set — skipping pgvector store tests",
)

_TEST_TABLE = "_sutra_test_pgvec"

from sutra.core.graph.pgvector_store import PGVectorStore  # noqa: E402


# ---------------------------------------------------------------------------
# Teardown helper (destructive — test use only)
# ---------------------------------------------------------------------------

def _drop_table(table_name: str) -> None:
    conn = psycopg2.connect(_PG_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()
    conn.close()


def setup_module(_: object) -> None:
    _drop_table(_TEST_TABLE)


def teardown_module(_: object) -> None:
    _drop_table(_TEST_TABLE)


# ---------------------------------------------------------------------------
# Test 1: dims is a required constructor argument
# ---------------------------------------------------------------------------

def test_constructor_requires_dims() -> None:
    """PGVectorStore(conn_str) without dims raises TypeError."""
    with pytest.raises(TypeError, match="dims"):
        PGVectorStore(_PG_URL)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Test 2: write() rejects wrong-dim vectors before touching DB
# ---------------------------------------------------------------------------

def test_write_wrong_dims_raises_before_db() -> None:
    """write() with wrong-dim vectors raises ValueError with a clear message."""
    store = PGVectorStore(_PG_URL, dims=4, table_name=_TEST_TABLE)
    try:
        store.setup()
        bad_vectors = np.zeros((2, 8), dtype=np.float32)  # 8 dims, store expects 4
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.write(["a", "b"], bad_vectors)
    finally:
        store.close()
        _drop_table(_TEST_TABLE)


# ---------------------------------------------------------------------------
# Test 3: write() rejects mismatched monikers/vectors length
# ---------------------------------------------------------------------------

def test_write_length_mismatch_raises() -> None:
    """write() with len(monikers) != vectors.shape[0] raises ValueError."""
    store = PGVectorStore(_PG_URL, dims=4, table_name=_TEST_TABLE)
    try:
        store.setup()
        vectors = np.zeros((3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="[Ll]ength mismatch"):
            store.write(["a", "b"], vectors)  # 2 monikers, 3 vectors
    finally:
        store.close()
        _drop_table(_TEST_TABLE)


# ---------------------------------------------------------------------------
# Test 4: setup() raises on existing table with wrong dims
# ---------------------------------------------------------------------------

def test_setup_detects_existing_dim_mismatch() -> None:
    """setup() against an existing table with wrong dims raises ValueError."""
    store_4 = PGVectorStore(_PG_URL, dims=4, table_name=_TEST_TABLE)
    try:
        store_4.setup()
    finally:
        store_4.close()

    store_8 = PGVectorStore(_PG_URL, dims=8, table_name=_TEST_TABLE)
    try:
        with pytest.raises(ValueError, match="[Dd]imension mismatch"):
            store_8.setup()
    finally:
        store_8.close()
        _drop_table(_TEST_TABLE)


# ---------------------------------------------------------------------------
# Test 5: setup(recreate=True) drops and recreates at new dims
# ---------------------------------------------------------------------------

def test_setup_recreate_drops_and_recreates() -> None:
    """setup(recreate=True) drops the old table and recreates at new dims."""
    # Create at dims=4 and write one row.
    store_4 = PGVectorStore(_PG_URL, dims=4, table_name=_TEST_TABLE)
    try:
        store_4.setup()
        store_4.write(["sym::a"], np.ones((1, 4), dtype=np.float32))
    finally:
        store_4.close()

    # Recreate at dims=8 — must not raise, old data must be gone.
    store_8 = PGVectorStore(_PG_URL, dims=8, table_name=_TEST_TABLE)
    try:
        store_8.setup(recreate=True)
        store_8.write(["sym::b"], np.ones((1, 8), dtype=np.float32))

        # Use a non-zero query vector — cosine distance is undefined for zero vectors.
        results = store_8.search(np.ones(8, dtype=np.float32), k=10)
        monikers = [m for m, _ in results]
        assert "sym::b" in monikers
        assert "sym::a" not in monikers
    finally:
        store_8.close()
        _drop_table(_TEST_TABLE)


# ---------------------------------------------------------------------------
# Test 6: full round-trip at 1536 dims
# ---------------------------------------------------------------------------

def test_full_roundtrip_1536_dims() -> None:
    """setup → write 1536-dim vectors → search returns correct monikers."""
    store = PGVectorStore(_PG_URL, dims=1536, table_name=_TEST_TABLE)
    try:
        store.setup()
        rng = np.random.default_rng(42)
        vecs = rng.random((3, 1536)).astype(np.float32)
        monikers = ["sym::alpha", "sym::beta", "sym::gamma"]
        store.write(monikers, vecs)

        # Query with the exact vector for "sym::alpha" — it should be nearest.
        results = store.search(vecs[0], k=3)
        assert len(results) == 3
        top_moniker, top_dist = results[0]
        assert top_moniker == "sym::alpha"
        assert top_dist < 1e-5, f"Expected near-zero distance for exact match, got {top_dist}"

        # All three monikers must appear in results.
        result_monikers = {m for m, _ in results}
        assert result_monikers == set(monikers)
    finally:
        store.close()
        _drop_table(_TEST_TABLE)
