"""
Integration tests for the plain-SQL graph backend.

Gated on SUTRA_PG_URL env var — skipped if not set.  Same pattern as
test_graph_writers.py (AGE).  Real database, real psycopg2, no mocks.

Usage:
    SUTRA_PG_URL=postgresql://postgres:postgers@localhost:5433/postgres \
        pytest tests/test_sql_graph.py -v

Test isolation:
    The three SQL tables (sutra_repositories, sutra_symbols,
    sutra_relationships) are TRUNCATE'd between tests via a fixture, then
    DROP'd in teardown_module.  The tables are the production names — this
    is acceptable because PR 1 is purely additive: production paths still
    use AGE, so no live data is at risk if you run these against a dev DB.
    Do NOT run these against a database that holds real graph data.

Teardown helpers:
    Free functions, not methods on the production classes — destructive
    operations stay out of the production code.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import psycopg2
import pytest

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

_PG_URL = os.getenv("SUTRA_PG_URL", "")
pytestmark = pytest.mark.skipif(
    not _PG_URL,
    reason="SUTRA_PG_URL env var not set — skipping SQL graph tests",
)

# ---------------------------------------------------------------------------
# Imports (after gate)
# ---------------------------------------------------------------------------

from sutra.core.extractor.base import (
    ClassSymbol,
    FunctionSymbol,
    IndexResult,
    Location,
    RelationKind,
    Relationship,
    Repository,
    VariableSymbol,
    Visibility,
)
from sutra.core.graph.sql_reader import SqlIncrementalReader
from sutra.core.graph.sql_state import SqlIndexStateStore
from sutra.core.graph.sql_writer import SqlGraphWriter


# ---------------------------------------------------------------------------
# Teardown helpers — destructive, NOT in production classes
# ---------------------------------------------------------------------------

def _drop_all_tables(pg_url: str) -> None:
    """Drop the three SQL graph tables.  DESTRUCTIVE — test use only."""
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        # Drop relationships first (no FK), then symbols (FK to repos), then repos.
        cur.execute("DROP TABLE IF EXISTS sutra_relationships")
        cur.execute("DROP TABLE IF EXISTS sutra_symbols")
        cur.execute("DROP TABLE IF EXISTS sutra_repositories")
    conn.close()


def _truncate_all(pg_url: str) -> None:
    """Empty the tables between tests, preserving DDL."""
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE sutra_relationships, sutra_symbols, sutra_repositories "
            "RESTART IDENTITY CASCADE"
        )
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
    file_path: str = "src/a.py",
    body_hash: str | None = None,
    language: str = "python",
) -> FunctionSymbol:
    return FunctionSymbol(
        id=moniker,
        name=name,
        qualified_name=f"src.{name}",
        file_path=file_path,
        location=_loc(),
        body_hash=body_hash or f"sha256:{'a' * 64}",
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


def _var(moniker: str, name: str, file_path: str = "src/a.py") -> VariableSymbol:
    return VariableSymbol(
        id=moniker,
        name=name,
        qualified_name=f"src.{name}",
        file_path=file_path,
        location=_loc(),
        body_hash=f"sha256:{'c' * 64}",
        language="python",
        visibility=Visibility.PUBLIC,
        is_exported=True,
        type_annotation="int",
        is_constant=True,
    )


def _make_result(
    repo_url: str,
    repo_name: str,
    symbols: list,
    relationships: list | None = None,
    commit_hash: str = "abc123",
) -> IndexResult:
    return IndexResult(
        repository=Repository(url=repo_url, name=repo_name),
        files=[],
        symbols=symbols,
        relationships=relationships or [],
        indexed_at=datetime.now(timezone.utc),
        commit_hash=commit_hash,
        languages={"python": 1},
        failed_files=[],
    )


# ---------------------------------------------------------------------------
# Module-level setup / teardown
# ---------------------------------------------------------------------------

def setup_module(module: object) -> None:
    """Run migrations once for the whole test session."""
    # Always start fresh — drop any leftover tables from previous runs.
    _drop_all_tables(_PG_URL)
    writer = SqlGraphWriter(_PG_URL)
    writer.setup()
    writer.close()


def teardown_module(module: object) -> None:
    _drop_all_tables(_PG_URL)


@pytest.fixture(autouse=True)
def _clean_tables() -> None:
    """Truncate between tests for isolation."""
    _truncate_all(_PG_URL)


# ---------------------------------------------------------------------------
# Read helpers — used by assertions; talk SQL directly
# ---------------------------------------------------------------------------

def _query_one(sql: str, params: tuple = ()) -> tuple | None:
    conn = psycopg2.connect(_PG_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    conn.close()
    return row


def _query_all(sql: str, params: tuple = ()) -> list:
    conn = psycopg2.connect(_PG_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.close()
    return rows


def _count_symbols() -> int:
    row = _query_one("SELECT count(*) FROM sutra_symbols")
    return int(row[0])


def _count_relationships() -> int:
    row = _query_one("SELECT count(*) FROM sutra_relationships")
    return int(row[0])


# ===========================================================================
# SqlGraphWriter
# ===========================================================================


class TestSqlGraphWriter:
    """Tests for the plain-SQL graph writer."""

    def _writer(self) -> SqlGraphWriter:
        return SqlGraphWriter(_PG_URL)

    # ---- setup ---------------------------------------------------------

    def test_setup_is_idempotent(self) -> None:
        """Running setup() twice must not error."""
        with self._writer() as w:
            w.setup()  # tables already exist from setup_module
            w.setup()  # call again — idempotent CREATE IF NOT EXISTS

    def test_setup_creates_three_tables(self) -> None:
        for table in ("sutra_repositories", "sutra_symbols", "sutra_relationships"):
            row = _query_one(
                "SELECT to_regclass(%s)::text",
                (table,),
            )
            assert row[0] == table, f"{table} missing after setup()"

    # ---- write_repository ---------------------------------------------

    def test_write_repository_persists_symbols(self) -> None:
        sym_a = _func("sutra python testrepo src/a.py foo().", "foo")
        sym_b = _func("sutra python testrepo src/a.py bar().", "bar")
        sym_c = _cls("sutra python testrepo src/a.py Baz#.", "Baz")
        result = _make_result(
            "https://github.com/org/testrepo", "testrepo", [sym_a, sym_b, sym_c]
        )

        with self._writer() as w:
            w.write_repository(result)

        assert _count_symbols() == 3
        # Repo row created
        row = _query_one(
            "SELECT name, url, last_commit_sha FROM sutra_repositories WHERE name = %s",
            ("testrepo",),
        )
        assert row is not None
        assert row[0] == "testrepo"
        assert row[1] == "https://github.com/org/testrepo"
        assert row[2] == "abc123"

        # kind column populated correctly
        row = _query_one(
            "SELECT kind FROM sutra_symbols WHERE moniker = %s", (sym_a.id,)
        )
        assert row[0] == "function"
        row = _query_one(
            "SELECT kind FROM sutra_symbols WHERE moniker = %s", (sym_c.id,)
        )
        assert row[0] == "class"

    def test_upsert_overwrites_existing_symbol(self) -> None:
        moniker = "sutra python testrepo src/a.py upsert_me()."
        v1 = _func(moniker, "upsert_me_v1")
        v2 = _func(moniker, "upsert_me_v2")  # same moniker, new name

        result_v1 = _make_result("https://github.com/org/testrepo", "testrepo", [v1])
        result_v2 = _make_result("https://github.com/org/testrepo", "testrepo", [v2])

        with self._writer() as w:
            w.write_repository(result_v1)
            w.write_repository(result_v2)

        # Exactly one row, with v2's name
        rows = _query_all(
            "SELECT name FROM sutra_symbols WHERE moniker = %s", (moniker,)
        )
        assert len(rows) == 1
        assert rows[0][0] == "upsert_me_v2"

    def test_only_resolved_relationships_persisted(self) -> None:
        src = _func("sutra python testrepo src/a.py caller().", "caller")
        tgt = _func("sutra python testrepo src/a.py callee().", "callee")
        resolved = Relationship(
            source_id=src.id,
            kind=RelationKind.CALLS,
            is_resolved=True,
            target_id=tgt.id,
        )
        unresolved = Relationship(
            source_id=src.id,
            kind=RelationKind.CALLS,
            is_resolved=False,
            target_id=None,
            target_name="some_external_func",
        )
        result = _make_result(
            "https://github.com/org/testrepo",
            "testrepo",
            [src, tgt],
            [resolved, unresolved],
        )

        with self._writer() as w:
            w.write_repository(result)

        # Only the resolved relationship should be in the table
        assert _count_relationships() == 1
        row = _query_one(
            "SELECT source_id, target_id, kind FROM sutra_relationships"
        )
        assert row[0] == src.id
        assert row[1] == tgt.id
        assert row[2] == "calls"

    def test_relationship_dedup_by_composite_pk(self) -> None:
        """Same (source, target, kind) twice → upserted, not duplicated."""
        src = _func("sutra python testrepo src/a.py s().", "s")
        tgt = _func("sutra python testrepo src/a.py t().", "t")
        rel = Relationship(
            source_id=src.id,
            kind=RelationKind.CALLS,
            is_resolved=True,
            target_id=tgt.id,
        )
        # Two copies of the same relationship in one batch
        result = _make_result(
            "https://github.com/org/testrepo",
            "testrepo",
            [src, tgt],
            [rel, rel],
        )
        with self._writer() as w:
            w.write_repository(result)

        assert _count_relationships() == 1

    # ---- replace mode --------------------------------------------------

    def test_replace_mode_clears_old_symbols(self) -> None:
        old = _func("sutra python testrepo src/a.py old().", "old")
        new = _func("sutra python testrepo src/a.py new().", "new")
        result_old = _make_result(
            "https://github.com/org/testrepo", "testrepo", [old]
        )
        result_new = _make_result(
            "https://github.com/org/testrepo", "testrepo", [new]
        )

        with self._writer() as w:
            w.write_repository(result_old)
            w.write_repository(result_new, replace=True)

        # Only the new symbol survives
        rows = _query_all("SELECT moniker FROM sutra_symbols")
        monikers = {r[0] for r in rows}
        assert monikers == {new.id}

    # ---- delete_symbols ------------------------------------------------

    def test_delete_symbols_cascades_to_relationships(self) -> None:
        a = _func("sutra python testrepo src/a.py a().", "a")
        b = _func("sutra python testrepo src/a.py b().", "b")
        c = _func("sutra python testrepo src/a.py c().", "c")
        # a → b, b → c
        rels = [
            Relationship(
                source_id=a.id, kind=RelationKind.CALLS,
                is_resolved=True, target_id=b.id,
            ),
            Relationship(
                source_id=b.id, kind=RelationKind.CALLS,
                is_resolved=True, target_id=c.id,
            ),
        ]
        result = _make_result(
            "https://github.com/org/testrepo", "testrepo", [a, b, c], rels
        )
        with self._writer() as w:
            w.write_repository(result)

        assert _count_symbols() == 3
        assert _count_relationships() == 2

        # Delete b — it should remove both edges (one as source, one as target)
        with self._writer() as w:
            w.delete_symbols([b.id])

        assert _count_symbols() == 2
        assert _count_relationships() == 0

    def test_delete_symbols_empty_is_noop(self) -> None:
        with self._writer() as w:
            w.delete_symbols([])  # no error, no-op

    # ---- delete_relationships_from ------------------------------------

    def test_delete_relationships_from_removes_outbound_only(self) -> None:
        a = _func("sutra python testrepo src/a.py a().", "a")
        b = _func("sutra python testrepo src/a.py b().", "b")
        c = _func("sutra python testrepo src/a.py c().", "c")
        rels = [
            # Outbound from a
            Relationship(
                source_id=a.id, kind=RelationKind.CALLS,
                is_resolved=True, target_id=b.id,
            ),
            # Inbound to a (a is the target)
            Relationship(
                source_id=c.id, kind=RelationKind.CALLS,
                is_resolved=True, target_id=a.id,
            ),
        ]
        result = _make_result(
            "https://github.com/org/testrepo", "testrepo", [a, b, c], rels
        )
        with self._writer() as w:
            w.write_repository(result)

        with self._writer() as w:
            w.delete_relationships_from([a.id])

        # Only the c → a edge survives
        rows = _query_all(
            "SELECT source_id, target_id FROM sutra_relationships"
        )
        assert len(rows) == 1
        assert rows[0] == (c.id, a.id)

    # ---- write_symbol_direct + write_relationships_direct -------------

    def test_write_symbol_direct(self) -> None:
        # Create the repo row first (FK requirement)
        sym = _func("sutra python testrepo src/a.py x().", "x")
        result = _make_result(
            "https://github.com/org/testrepo", "testrepo", []
        )
        indexed_at = datetime.now(timezone.utc).isoformat()
        with self._writer() as w:
            w.write_repository(result)  # creates repo row, no symbols yet
            w.write_symbol_direct(sym, "testrepo", indexed_at)

        assert _count_symbols() == 1

    def test_write_relationships_direct_returns_skipped_count(self) -> None:
        a = _func("sutra python testrepo src/a.py a().", "a")
        b = _func("sutra python testrepo src/a.py b().", "b")
        result = _make_result(
            "https://github.com/org/testrepo", "testrepo", [a, b]
        )
        rels = [
            Relationship(
                source_id=a.id, kind=RelationKind.CALLS,
                is_resolved=True, target_id=b.id,
            ),
            Relationship(
                source_id=a.id, kind=RelationKind.CALLS,
                is_resolved=False, target_id=None, target_name="external",
            ),
            Relationship(
                source_id=b.id, kind=RelationKind.CALLS,
                is_resolved=False, target_id=None, target_name="other_external",
            ),
        ]
        with self._writer() as w:
            w.write_repository(result)
            skipped = w.write_relationships_direct(rels)

        assert skipped == 2
        assert _count_relationships() == 1

    # ---- transaction rollback -----------------------------------------

    def test_write_repository_rolls_back_on_error(self) -> None:
        """If the relationship insert fails mid-way, no symbols persist either."""
        # Force an error: a relationship row with a None source_id violates NOT NULL.
        a = _func("sutra python testrepo src/a.py a().", "a")
        bad_rel = Relationship(
            source_id=None,  # type: ignore[arg-type]  intentional bad value
            kind=RelationKind.CALLS,
            is_resolved=True,
            target_id=a.id,
        )
        result = _make_result(
            "https://github.com/org/testrepo", "testrepo", [a], [bad_rel]
        )

        with pytest.raises(Exception):
            with self._writer() as w:
                w.write_repository(result)

        # The transaction must have rolled back — no symbol persisted.
        assert _count_symbols() == 0


# ===========================================================================
# SqlIncrementalReader
# ===========================================================================


class TestSqlIncrementalReader:
    """Tests for the plain-SQL incremental reader."""

    def _seed(self) -> tuple[FunctionSymbol, FunctionSymbol, FunctionSymbol]:
        """Insert three symbols across two files for read-side tests."""
        a = _func("sutra python testrepo src/a.py a().", "a", file_path="src/a.py",
                  body_hash="sha256:" + "1" * 64)
        b = _func("sutra python testrepo src/a.py b().", "b", file_path="src/a.py",
                  body_hash="sha256:" + "2" * 64)
        c = _func("sutra python testrepo src/b.py c().", "c", file_path="src/b.py",
                  body_hash="sha256:" + "3" * 64)
        result = _make_result(
            "https://github.com/org/testrepo", "testrepo", [a, b, c]
        )
        with SqlGraphWriter(_PG_URL) as w:
            w.write_repository(result)
        return a, b, c

    def test_get_symbols_for_files_returns_dict(self) -> None:
        a, b, c = self._seed()
        with SqlIncrementalReader(_PG_URL) as r:
            result = r.get_symbols_for_files("testrepo", ["src/a.py"])

        assert result == {a.id: a.body_hash, b.id: b.body_hash}

    def test_get_symbols_for_files_multi_file(self) -> None:
        a, b, c = self._seed()
        with SqlIncrementalReader(_PG_URL) as r:
            result = r.get_symbols_for_files("testrepo", ["src/a.py", "src/b.py"])
        assert set(result.keys()) == {a.id, b.id, c.id}

    def test_get_symbols_for_files_empty_list(self) -> None:
        with SqlIncrementalReader(_PG_URL) as r:
            result = r.get_symbols_for_files("testrepo", [])
        assert result == {}

    def test_get_symbols_for_files_unknown_path(self) -> None:
        self._seed()
        with SqlIncrementalReader(_PG_URL) as r:
            result = r.get_symbols_for_files("testrepo", ["src/never_existed.py"])
        assert result == {}

    def test_get_symbols_for_files_empty_repo_name_falls_through(self) -> None:
        """Legacy IncrementalUpdater pattern: deleted-files lookup uses repo_name=''."""
        a, b, c = self._seed()
        with SqlIncrementalReader(_PG_URL) as r:
            result = r.get_symbols_for_files("", ["src/a.py"])
        # Both symbols on src/a.py returned regardless of repo
        assert set(result.keys()) == {a.id, b.id}


# ===========================================================================
# SqlIndexStateStore
# ===========================================================================


class TestSqlIndexStateStore:
    """Tests for the per-repo state store."""

    def test_get_returns_none_when_repo_absent(self) -> None:
        with SqlIndexStateStore(_PG_URL) as s:
            assert s.get_last_commit_sha("nonexistent") is None

    def test_update_then_get_round_trips(self) -> None:
        with SqlIndexStateStore(_PG_URL) as s:
            s.update_commit_sha("myrepo", "abc123")
            assert s.get_last_commit_sha("myrepo") == "abc123"

    def test_update_overwrites_previous_sha(self) -> None:
        with SqlIndexStateStore(_PG_URL) as s:
            s.update_commit_sha("myrepo", "abc123")
            s.update_commit_sha("myrepo", "def456")
            assert s.get_last_commit_sha("myrepo") == "def456"

    def test_update_creates_repo_row_if_absent(self) -> None:
        """update_commit_sha is upsert — first call creates the repo row."""
        with SqlIndexStateStore(_PG_URL) as s:
            s.update_commit_sha("brand_new_repo", "first_sha")

        row = _query_one(
            "SELECT name, last_commit_sha FROM sutra_repositories WHERE name = %s",
            ("brand_new_repo",),
        )
        assert row is not None
        assert row[0] == "brand_new_repo"
        assert row[1] == "first_sha"

    def test_get_returns_none_when_sha_unset(self) -> None:
        """A repo row with last_commit_sha=NULL (e.g. interrupted run) → None."""
        # Insert a repo row directly with no SHA
        conn = psycopg2.connect(_PG_URL)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sutra_repositories (name, url) VALUES (%s, %s)",
                ("partial_repo", "https://example.com"),
            )
        conn.close()

        with SqlIndexStateStore(_PG_URL) as s:
            assert s.get_last_commit_sha("partial_repo") is None
