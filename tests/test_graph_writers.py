"""
Priority 10 — AGEWriter + PGVectorStore integration tests.

Gated on SUTRA_PG_URL env var — skipped entirely if not set.
All tests use real database connections, real AGE Cypher queries, and real
pgvector operations.  No mocks.

Usage:
    SUTRA_PG_URL=postgresql://postgres:postgers@localhost:5433/postgres \
        pytest tests/test_graph_writers.py -v

Test isolation:
    All tests use a dedicated graph name and table name that are dropped in
    teardown_module.  The names are intentionally distinct from the production
    defaults to prevent accidental data loss.

Teardown helpers:
    _drop_graph() and _drop_table() are free functions — not methods of the
    production writers.  Destructive operations must not live in production
    classes.
"""
from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg2
import pytest

# ---------------------------------------------------------------------------
# Gate: skip entire module if no DB connection string
# ---------------------------------------------------------------------------

_PG_URL = os.getenv("SUTRA_PG_URL", "")
pytestmark = pytest.mark.skipif(
    not _PG_URL,
    reason="SUTRA_PG_URL env var not set — skipping graph writer tests",
)

# Test-only graph/table names — distinct from production defaults
_TEST_GRAPH = "_sutra_test_graph"
_TEST_TABLE = "_sutra_test_embeddings"
_TEST_DIMS = 4  # tiny dims for fast tests

# ---------------------------------------------------------------------------
# Imports (after gate — avoids import errors when DB not available)
# ---------------------------------------------------------------------------

from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.extractor.base import (
    ClassSymbol,
    FunctionSymbol,
    IndexResult,
    Location,
    MethodSymbol,
    RelationKind,
    Relationship,
    Repository,
    VariableSymbol,
    Visibility,
)
from sutra.core.graph.postgres_age import AGEWriter, _cypher_val
from sutra.core.graph.pgvector_store import PGVectorStore
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter


# ---------------------------------------------------------------------------
# Teardown helpers (NOT methods on production classes)
# ---------------------------------------------------------------------------

def _drop_graph(pg_url: str, graph_name: str) -> None:
    """Drop an AGE graph. DESTRUCTIVE — test use only."""
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("LOAD 'age'")
        cur.execute("SET search_path = ag_catalog, \"$user\", public")
        cur.execute(
            "SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s",
            (graph_name,),
        )
        if cur.fetchone()[0] > 0:
            cur.execute("SELECT drop_graph(%s, true)", (graph_name,))
    conn.close()


def _drop_table(pg_url: str, table_name: str) -> None:
    """Drop the pgvector embeddings table. DESTRUCTIVE — test use only."""
    conn = psycopg2.connect(pg_url)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _loc(line_start: int = 1, line_end: int = 5) -> Location:
    return Location(
        line_start=line_start, line_end=line_end, byte_start=0, byte_end=100
    )


def _func(
    moniker: str,
    name: str,
    repo_name: str = "testrepo",
    file_path: str = "src/a.py",
    language: str = "python",
) -> FunctionSymbol:
    return FunctionSymbol(
        id=moniker,
        name=name,
        qualified_name=f"src.{name}",
        file_path=file_path,
        location=_loc(),
        body_hash=f"sha256:{'a' * 64}",
        language=language,
        visibility=Visibility.PUBLIC,
        is_exported=True,
        signature=f"def {name}() -> None",
        parameters=[],
        return_type="None",
        docstring=f"Docstring for {name}.",
        decorators=[],
        is_async=False,
        complexity=1,
    )


def _cls(
    moniker: str,
    name: str,
    repo_name: str = "testrepo",
    file_path: str = "src/a.py",
) -> ClassSymbol:
    return ClassSymbol(
        id=moniker,
        name=name,
        qualified_name=f"src.{name}",
        file_path=file_path,
        location=_loc(),
        body_hash=f"sha256:{'b' * 64}",
        language="python",
        visibility=Visibility.PUBLIC,
        is_exported=True,
        base_classes=["Base"],
        docstring=f"Class {name}.",
        decorators=[],
        is_abstract=False,
    )


def _make_index_result(
    repo_url: str,
    symbols: list,
    relationships: list | None = None,
) -> IndexResult:
    from sutra.core.extractor.moniker import repo_name_from_url
    repo_name = repo_name_from_url(repo_url)
    return IndexResult(
        repository=Repository(url=repo_url, name=repo_name),
        files=[],
        symbols=symbols,
        relationships=relationships or [],
        indexed_at=datetime.now(timezone.utc),
        commit_hash="abc123",
        languages={"python": 1},
        failed_files=[],
    )


# ---------------------------------------------------------------------------
# Module-level setup / teardown
# ---------------------------------------------------------------------------

def setup_module(module: object) -> None:
    """Create test graph and table once for the entire test session."""
    writer = AGEWriter(_PG_URL, graph_name=_TEST_GRAPH)
    writer.setup()
    writer.close()

    store = PGVectorStore(_PG_URL, dims=_TEST_DIMS, table_name=_TEST_TABLE)
    store.setup()
    store.close()


def teardown_module(module: object) -> None:
    """Drop test graph and table after all tests complete."""
    _drop_graph(_PG_URL, _TEST_GRAPH)
    _drop_table(_PG_URL, _TEST_TABLE)


# ---------------------------------------------------------------------------
# Helpers that execute Cypher for assertions
# ---------------------------------------------------------------------------

def _cypher_query(graph_name: str, cypher: str, return_spec: str = "r agtype") -> list:
    """Execute a read-only Cypher query and return raw result rows."""
    conn = psycopg2.connect(_PG_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("LOAD 'age'")
        cur.execute("SET search_path = ag_catalog, \"$user\", public")
        cur.execute(
            f"SELECT * FROM cypher('{graph_name}', $${cypher}$$) AS ({return_spec})"
        )
        rows = cur.fetchall()
    conn.close()
    return rows


def _count_nodes(graph_name: str, label: str) -> int:
    rows = _cypher_query(graph_name, f"MATCH (n:{label}) RETURN count(n)", "cnt agtype")
    return int(rows[0][0])


def _count_edges(graph_name: str, label: str) -> int:
    rows = _cypher_query(graph_name, f"MATCH ()-[e:{label}]->() RETURN count(e)", "cnt agtype")
    return int(rows[0][0])


def _get_node_prop(graph_name: str, node_id: str, prop: str) -> str:
    rows = _cypher_query(
        graph_name,
        f"MATCH (n {{id: '{node_id}'}}) RETURN n.{prop}",
        "val agtype",
    )
    if not rows:
        return None
    raw = rows[0][0]
    # AGE returns strings with surrounding double-quotes
    return raw.strip('"') if raw is not None else None


# ---------------------------------------------------------------------------
# TestAGEWriter
# ---------------------------------------------------------------------------

class TestAGEWriter:
    """Tests for the AGE graph writer."""

    def _writer(self) -> AGEWriter:
        return AGEWriter(_PG_URL, graph_name=_TEST_GRAPH)

    def test_setup_is_idempotent(self) -> None:
        """Calling setup() twice must not raise."""
        with self._writer() as w:
            w.setup()  # already ran in setup_module; call again

    def test_write_symbols_match_back(self) -> None:
        """Write 3 symbols; MATCH all back; assert id and kind properties."""
        sym_a = _func("sutra python testrepo src/a.py foo().", "foo")
        sym_b = _func("sutra python testrepo src/a.py bar().", "bar")
        sym_c = _cls("sutra python testrepo src/a.py Baz#.", "Baz")
        result = _make_index_result("https://github.com/org/testrepo", [sym_a, sym_b, sym_c])

        with self._writer() as w:
            w.write_repository(result)

        # All three Symbol nodes must exist
        rows = _cypher_query(
            _TEST_GRAPH,
            "MATCH (n:Symbol) WHERE n.id IN "
            "['sutra python testrepo src/a.py foo().', "
            " 'sutra python testrepo src/a.py bar().', "
            " 'sutra python testrepo src/a.py Baz#.'] "
            "RETURN n.id",
            "id agtype",
        )
        found_ids = {r[0].strip('"') for r in rows}
        assert sym_a.id in found_ids
        assert sym_b.id in found_ids
        assert sym_c.id in found_ids

        # Assert kind property
        kind_a = _get_node_prop(_TEST_GRAPH, sym_a.id, "kind")
        assert kind_a == "function"
        kind_c = _get_node_prop(_TEST_GRAPH, sym_c.id, "kind")
        assert kind_c == "class"

    def test_upsert_overwrites_existing_symbol(self) -> None:
        """Writing the same moniker twice must update properties, not duplicate."""
        moniker = "sutra python testrepo src/a.py upsert_me()."
        sym_v1 = _func(moniker, "upsert_me_v1")
        sym_v2 = _func(moniker, "upsert_me_v2")  # same moniker, different name

        result_v1 = _make_index_result("https://github.com/org/testrepo", [sym_v1])
        result_v2 = _make_index_result("https://github.com/org/testrepo", [sym_v2])

        with self._writer() as w:
            w.write_repository(result_v1)
            w.write_repository(result_v2)

        # Must have exactly one node with this moniker
        rows = _cypher_query(
            _TEST_GRAPH,
            f"MATCH (n:Symbol {{id: '{moniker}'}}) RETURN n.name",
            "name agtype",
        )
        assert len(rows) == 1, f"Expected 1 node, got {len(rows)}"
        # Name must reflect the second write
        assert rows[0][0].strip('"') == "upsert_me_v2"

    def test_only_resolved_relationships_written(self) -> None:
        """Resolved rels become edges; unresolved (target_id=None) are skipped."""
        src = _func("sutra python testrepo src/a.py caller().", "caller")
        tgt = _func("sutra python testrepo src/a.py callee().", "callee")

        resolved_rel = Relationship(
            source_id=src.id,
            target_id=tgt.id,
            target_name="callee",
            kind=RelationKind.CALLS,
            is_resolved=True,
        )
        unresolved_rel = Relationship(
            source_id=src.id,
            target_id=None,
            target_name="external_func",
            kind=RelationKind.CALLS,
            is_resolved=False,
        )

        result = _make_index_result(
            "https://github.com/org/testrepo",
            [src, tgt],
            relationships=[resolved_rel, unresolved_rel],
        )

        before = _count_edges(_TEST_GRAPH, "CALLS")

        with self._writer() as w:
            w.write_repository(result)

        after = _count_edges(_TEST_GRAPH, "CALLS")
        # Exactly 1 CALLS edge added (the resolved one); unresolved skipped
        assert after == before + 1

    def test_replace_mode_removes_old_symbols(self) -> None:
        """replace=True must DETACH DELETE old symbols before writing new ones."""
        url = "https://github.com/org/replacerepo"
        sym_old = _func("sutra python replacerepo src/a.py old_func().", "old_func")
        sym_new = _func("sutra python replacerepo src/a.py new_func().", "new_func")

        result_v1 = _make_index_result(url, [sym_old])
        result_v2 = _make_index_result(url, [sym_new])

        with self._writer() as w:
            w.write_repository(result_v1)
            w.write_repository(result_v2, replace=True)

        # old_func must be gone, new_func must exist
        old_rows = _cypher_query(
            _TEST_GRAPH,
            f"MATCH (n:Symbol {{id: '{sym_old.id}'}}) RETURN n.id",
            "id agtype",
        )
        assert len(old_rows) == 0, "old_func should have been deleted by replace=True"

        new_rows = _cypher_query(
            _TEST_GRAPH,
            f"MATCH (n:Symbol {{id: '{sym_new.id}'}}) RETURN n.id",
            "id agtype",
        )
        assert len(new_rows) == 1, "new_func should exist after replace=True"

    def test_two_repos_queryable_separately(self) -> None:
        """Symbols from two repos live in the same graph but are queryable separately."""
        url_a = "https://github.com/org/alpharepo"
        url_b = "https://github.com/org/betarepo"

        sym_a1 = _func("sutra python alpharepo src/a.py alpha_one().", "alpha_one")
        sym_a2 = _func("sutra python alpharepo src/a.py alpha_two().", "alpha_two")
        sym_b1 = _func("sutra python betarepo src/b.py beta_one().", "beta_one")

        result_a = _make_index_result(url_a, [sym_a1, sym_a2])
        result_b = _make_index_result(url_b, [sym_b1])

        with self._writer() as w:
            w.write_repository(result_a)
            w.write_repository(result_b)

        # Query repo A symbols via BELONGS_TO
        alpha_rows = _cypher_query(
            _TEST_GRAPH,
            "MATCH (n:Symbol)-[:BELONGS_TO]->(:Repository {name: 'alpharepo'}) RETURN n.name",
            "name agtype",
        )
        alpha_names = {r[0].strip('"') for r in alpha_rows}
        assert "alpha_one" in alpha_names
        assert "alpha_two" in alpha_names
        assert "beta_one" not in alpha_names

        beta_rows = _cypher_query(
            _TEST_GRAPH,
            "MATCH (n:Symbol)-[:BELONGS_TO]->(:Repository {name: 'betarepo'}) RETURN n.name",
            "name agtype",
        )
        beta_names = {r[0].strip('"') for r in beta_rows}
        assert "beta_one" in beta_names
        assert "alpha_one" not in beta_names

    def test_unresolved_rel_not_in_graph(self, capsys: pytest.CaptureFixture) -> None:
        """Unresolved relationships must not appear as edges; count logged to stderr."""
        src = _func("sutra python testrepo src/a.py log_test_src().", "log_test_src")
        result = _make_index_result(
            "https://github.com/org/testrepo",
            [src],
            relationships=[
                Relationship(
                    source_id=src.id,
                    target_id=None,
                    target_name="some_external",
                    kind=RelationKind.CALLS,
                    is_resolved=False,
                )
            ],
        )

        with self._writer() as w:
            w.write_repository(result)

        captured = capsys.readouterr()
        assert "Skipped 1 unresolved" in captured.err

    def test_repository_node_created(self) -> None:
        """A Repository node must be created with name, url, commit_sha."""
        url = "https://github.com/org/reponode_test"
        result = _make_index_result(url, [_func("sutra python reponode_test src/a.py f().", "f")])

        with self._writer() as w:
            w.write_repository(result)

        rows = _cypher_query(
            _TEST_GRAPH,
            "MATCH (r:Repository {name: 'reponode_test'}) RETURN r.url",
            "url agtype",
        )
        assert len(rows) >= 1
        assert rows[0][0].strip('"') == url

    def test_close_marks_connection_closed(self) -> None:
        """close() must close the psycopg2 connection (conn.closed != 0)."""
        w = self._writer()
        assert w._conn.closed == 0  # 0 means open
        w.close()
        assert w._conn.closed != 0  # non-zero means closed

    def test_context_manager_closes_on_exit(self) -> None:
        """Context manager __exit__ must close the connection."""
        with self._writer() as w:
            conn = w._conn
        assert conn.closed != 0


# ---------------------------------------------------------------------------
# TestPGVectorStore
# ---------------------------------------------------------------------------

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
        assert results[0][1] < 1e-6  # distance ≈ 0 for exact match

    def test_upsert_overwrites_old_vector(self) -> None:
        """Write then overwrite a vector; search must reflect the new value."""
        v_old = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        v_new = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        with self._store() as s:
            s.write(["sym_upsert"], v_old)
            s.write(["sym_upsert"], v_new)
            results = s.search(v_new[0], k=1)

        assert results[0][0] == "sym_upsert"
        assert results[0][1] < 1e-6

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


# ---------------------------------------------------------------------------
# TestCypherVal (unit — no DB connection needed)
# ---------------------------------------------------------------------------

class TestCypherVal:
    """Unit tests for _cypher_val() — no DB connection needed."""

    def test_none(self) -> None:
        assert _cypher_val(None) == "null"

    def test_bool_true(self) -> None:
        assert _cypher_val(True) == "true"

    def test_bool_false(self) -> None:
        assert _cypher_val(False) == "false"

    def test_int(self) -> None:
        assert _cypher_val(42) == "42"

    def test_string_plain(self) -> None:
        assert _cypher_val("hello") == "'hello'"

    def test_string_with_single_quote(self) -> None:
        result = _cypher_val("it's")
        assert "\\'" in result

    def test_string_with_backslash(self) -> None:
        result = _cypher_val("a\\b")
        assert "\\\\" in result

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError):
            _cypher_val([1, 2, 3])


# ---------------------------------------------------------------------------
# TestIndexerIntegration — end-to-end with both writers
# ---------------------------------------------------------------------------

class TestIndexerIntegration:
    """
    End-to-end test: Indexer with AGEWriter + PGVectorStore.
    Uses the checked-in fixture repo so no live files needed.
    """

    _FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_python_repo"
    _FIXTURE_URL = "https://github.com/ritikk112/sutra"
    _INTEG_GRAPH = "_sutra_integ_test_graph"
    _INTEG_TABLE = "_sutra_integ_test_embeddings"
    _INTEG_DIMS = 384

    def setup_method(self) -> None:
        """Create fresh graph + table before each test in this class."""
        writer = AGEWriter(_PG_URL, graph_name=self._INTEG_GRAPH)
        writer.setup()
        writer.close()
        store = PGVectorStore(_PG_URL, dims=self._INTEG_DIMS, table_name=self._INTEG_TABLE)
        store.setup()
        store.close()

    def teardown_method(self) -> None:
        """Drop graph + table after each test."""
        _drop_graph(_PG_URL, self._INTEG_GRAPH)
        _drop_table(_PG_URL, self._INTEG_TABLE)

    def test_indexer_writes_to_age_and_pgvector(self, tmp_path: Path) -> None:
        """
        Full pipeline: Indexer writes JSON + AGE nodes + pgvector embeddings.
        Symbol count in AGE must match symbol count in graph.json.
        """
        age_writer = AGEWriter(_PG_URL, graph_name=self._INTEG_GRAPH)
        pgvec_store = PGVectorStore(_PG_URL, dims=self._INTEG_DIMS, table_name=self._INTEG_TABLE)

        indexer = Indexer(
            adapters={"python": PythonAdapter()},
            exporter=JsonGraphExporter(),
            embedder=FixtureEmbedder(dims=self._INTEG_DIMS),
            age_writer=age_writer,
            pgvector_store=pgvec_store,
        )

        result = indexer.index(
            root=self._FIXTURE_REPO,
            repo_url=self._FIXTURE_URL,
            output_dir=tmp_path,
        )

        age_writer.close()
        pgvec_store.close()

        # Symbol count in AGE must equal symbols in IndexResult
        age_symbol_count = _count_nodes(self._INTEG_GRAPH, "Symbol")
        assert age_symbol_count == len(result.symbols), (
            f"AGE has {age_symbol_count} Symbol nodes; IndexResult has {len(result.symbols)}"
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
            f"pgvector has {pgvec_count} rows; graph.json reports {expected_embeddings}"
        )

    def test_existing_tests_unaffected_without_writers(self, tmp_path: Path) -> None:
        """Indexer without writers must work identically to pre-P10 behavior."""
        indexer = Indexer(
            adapters={"python": PythonAdapter()},
            exporter=JsonGraphExporter(),
            embedder=FixtureEmbedder(dims=self._INTEG_DIMS),
            # No age_writer or pgvector_store — both default to None
        )
        result = indexer.index(
            root=self._FIXTURE_REPO,
            repo_url=self._FIXTURE_URL,
            output_dir=tmp_path,
        )
        assert len(result.symbols) > 0
        assert (tmp_path / "graph.json").exists()
