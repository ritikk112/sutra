"""
Integration tests for P11: Incremental Update Pipeline.

Migrated from the AGE-backed version when PR 2 swapped Apache AGE for plain
Postgres SQL tables.  All test intents are preserved; only the storage
backend changed.

Reader/state tests (independent of the updater) live in test_sql_graph.py.
This file focuses on the updater's behaviour: per-file diff, fallback
threshold, recovery invariant, gitignore, etc.

Gating
------
    SUTRA_PG_URL=postgresql://postgres:postgers@localhost:5433/postgres pytest tests/test_incremental_updater.py

If SUTRA_PG_URL is not set, all DB-dependent tests are skipped.

Test isolation
--------------
The SQL graph tables are real production names (sutra_repositories,
sutra_symbols, sutra_relationships).  Acceptable because this is a
dev DB; do NOT run against a database holding live graph data.
truncate-between-tests fixture provides isolation.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

PG_URL = os.environ.get("SUTRA_PG_URL")
pytestmark = pytest.mark.skipif(
    not PG_URL, reason="SUTRA_PG_URL not set — skipping DB integration tests"
)

# ── Test isolation names ────────────────────────────────────────────────────
_TEST_TABLE = "_sutra_incremental_test_embeddings"
_TEST_DIMS = 4   # tiny vectors — fast, no real embedder needed

# ── Imports (only resolved when PG_URL is present) ─────────────────────────
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.base import UpdateResult
from sutra.core.git_differ import ChangedFiles, changed_files
from sutra.core.gitignore_filter import GitignoreFilter
from sutra.core.graph.pgvector_store import PGVectorStore
from sutra.core.graph.sql_reader import SqlIncrementalReader
from sutra.core.graph.sql_state import SqlIndexStateStore
from sutra.core.graph.sql_writer import SqlGraphWriter
from sutra.core.incremental_updater import IncrementalUpdater
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter


# ── Teardown helpers (free functions — destructive ops out of prod classes) ─

def _drop_sql_tables(pg_url: str) -> None:
    import psycopg2
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS sutra_relationships")
        cur.execute("DROP TABLE IF EXISTS sutra_symbols")
        cur.execute("DROP TABLE IF EXISTS sutra_repositories")
    conn.close()


def _truncate_sql_tables(pg_url: str) -> None:
    import psycopg2
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE sutra_relationships, sutra_symbols, sutra_repositories "
            "RESTART IDENTITY CASCADE"
        )
    conn.close()


def _drop_table(pg_url: str, table_name: str) -> None:
    import psycopg2
    conn = psycopg2.connect(pg_url)
    with conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.close()


# ── Module-level setup / teardown ────────────────────────────────────────────

def setup_module(module: object) -> None:
    """Create SQL tables once for the whole test session."""
    _drop_sql_tables(PG_URL)
    w = SqlGraphWriter(PG_URL)
    w.setup()
    w.close()


def teardown_module(module: object) -> None:
    _drop_sql_tables(PG_URL)


# ── Git repo fixture helpers ─────────────────────────────────────────────────

def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _init_repo(tmp: Path, repo_name: str = "test_repo") -> Path:
    """Initialise a bare git repo with a dummy initial commit."""
    repo = tmp / repo_name
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@test.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "commit", "--allow-empty", "-m", "init")
    return repo


def _write_and_commit(repo: Path, files: dict[str, str], message: str) -> str:
    """Write files to repo, stage, commit, return new SHA."""
    for rel_path, content in files.items():
        abs_path = repo / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _delete_and_commit(repo: Path, files: list[str], message: str) -> str:
    """Delete files from repo, stage, commit, return new SHA."""
    for rel_path in files:
        _git(repo, "rm", rel_path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


# ── Shared adapters + embedder ───────────────────────────────────────────────

def _adapters():
    from sutra.core.extractor.adapters.python import PythonAdapter
    return {"python": PythonAdapter()}


def _embedder():
    return FixtureEmbedder(dims=_TEST_DIMS)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _truncate_between_tests() -> None:
    """Empty SQL tables before every test for isolation."""
    _truncate_sql_tables(PG_URL)


@pytest.fixture()
def graph_writer() -> Generator[SqlGraphWriter, None, None]:
    w = SqlGraphWriter(PG_URL)
    try:
        yield w
    finally:
        w.close()


@pytest.fixture()
def reader() -> Generator[SqlIncrementalReader, None, None]:
    r = SqlIncrementalReader(PG_URL)
    try:
        yield r
    finally:
        r.close()


@pytest.fixture()
def state_store() -> Generator[SqlIndexStateStore, None, None]:
    s = SqlIndexStateStore(PG_URL)
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def pgvector_store() -> Generator[PGVectorStore, None, None]:
    store = PGVectorStore(PG_URL, dims=_TEST_DIMS, table_name=_TEST_TABLE)
    store.setup()
    try:
        yield store
    finally:
        store.close()
        _drop_table(PG_URL, _TEST_TABLE)


@pytest.fixture()
def updater(
    graph_writer, reader, state_store, pgvector_store
) -> IncrementalUpdater:
    return IncrementalUpdater(
        adapters=_adapters(),
        embedder=_embedder(),
        graph_writer=graph_writer,
        reader=reader,
        state_store=state_store,
        pgvector_store=pgvector_store,
        exporter=JsonGraphExporter(),
    )


@pytest.fixture()
def tmp_repo(tmp_path) -> Generator[Path, None, None]:
    yield _init_repo(tmp_path)


def _full_index(
    repo: Path,
    repo_url: str,
    graph_writer: SqlGraphWriter,
    pgvector_store: PGVectorStore,
    output_dir: Path,
) -> None:
    """Helper to run a full index into the test SQL tables."""
    indexer = Indexer(
        adapters=_adapters(),
        exporter=JsonGraphExporter(),
        embedder=_embedder(),
        graph_writer=graph_writer,
        pgvector_store=pgvector_store,
    )
    indexer.index(repo, repo_url, output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# TestGitDiffer — pure unit tests (no DB, but git required)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGitDiffer:
    """Pure unit tests for git_differ.changed_files() against a hermetic repo."""

    def test_added_file(self, tmp_path):
        repo = _init_repo(tmp_path)
        sha1 = _git(repo, "rev-parse", "HEAD")
        sha2 = _write_and_commit(repo, {"src/new.py": "x = 1\n"}, "add file")

        diff = changed_files(repo, sha1, sha2)

        assert "src/new.py" in diff.added
        assert not diff.modified
        assert not diff.deleted

    def test_modified_file(self, tmp_path):
        repo = _init_repo(tmp_path)
        sha1 = _write_and_commit(repo, {"src/foo.py": "x = 1\n"}, "add")
        sha2 = _write_and_commit(repo, {"src/foo.py": "x = 2\n"}, "modify")

        diff = changed_files(repo, sha1, sha2)

        assert "src/foo.py" in diff.modified
        assert not diff.added
        assert not diff.deleted

    def test_deleted_file(self, tmp_path):
        repo = _init_repo(tmp_path)
        sha1 = _write_and_commit(repo, {"src/gone.py": "x = 1\n"}, "add")
        sha2 = _delete_and_commit(repo, ["src/gone.py"], "delete")

        diff = changed_files(repo, sha1, sha2)

        assert "src/gone.py" in diff.deleted
        assert not diff.added
        assert not diff.modified

    def test_all_to_index_is_added_union_modified(self, tmp_path):
        repo = _init_repo(tmp_path)
        sha1 = _write_and_commit(repo, {"existing.py": "a = 1\n"}, "base")
        sha2 = _write_and_commit(
            repo,
            {"existing.py": "a = 2\n", "new.py": "b = 1\n"},
            "both",
        )
        diff = changed_files(repo, sha1, sha2)
        assert diff.all_to_index == diff.added | diff.modified

    def test_no_changes(self, tmp_path):
        repo = _init_repo(tmp_path)
        sha1 = _write_and_commit(repo, {"f.py": "x=1\n"}, "a")
        sha2 = _write_and_commit(repo, {"README.md": "# hi\n"}, "b")
        diff = changed_files(repo, sha1, sha2)
        # No Python files changed
        assert "f.py" not in diff.modified
        assert "f.py" not in diff.added

    def test_invalid_sha_raises(self, tmp_path):
        repo = _init_repo(tmp_path)
        with pytest.raises(RuntimeError, match="failed"):
            changed_files(repo, "deadbeef" * 5, "cafebabe" * 5)


# ═══════════════════════════════════════════════════════════════════════════════
# TestIncrementalUpdater
# ═══════════════════════════════════════════════════════════════════════════════

class TestIncrementalUpdater:

    # ── Test 1: no-op on same SHA ───────────────────────────────────────────

    def test_noop_on_same_sha(
        self, tmp_path, updater, graph_writer, pgvector_store
    ):
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/noop-repo"
        _write_and_commit(repo, {"a.py": "def f(): pass\n"}, "c1")

        _full_index(repo, repo_url, graph_writer, pgvector_store, tmp_path / "out")

        result = updater.update(repo, repo_url, tmp_path / "out")

        assert result.added == 0
        assert result.updated == 0
        assert result.deleted == 0
        assert result.skipped == 0
        assert result.fell_back_to_full_index is False
        assert result.old_commit_sha == result.new_commit_sha

    # ── Test 2: new file added ──────────────────────────────────────────────

    def test_new_file_added(
        self, tmp_path, updater, graph_writer, pgvector_store, reader
    ):
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/add-repo"
        _write_and_commit(repo, {"a.py": "def f(): pass\n"}, "c1")
        _full_index(repo, repo_url, graph_writer, pgvector_store, tmp_path / "out")

        _write_and_commit(repo, {"b.py": "def g(): return 1\n"}, "add b")
        result = updater.update(repo, repo_url, tmp_path / "out")

        assert result.added > 0
        assert result.fell_back_to_full_index is False

        # b.py symbols should appear in the graph
        syms = reader.get_symbols_for_files("add-repo", ["b.py"])
        assert len(syms) > 0

    # ── Test 3: modified file — changed body_hash updated ──────────────────

    def test_modified_file_updated(
        self, tmp_path, updater, graph_writer, pgvector_store, reader
    ):
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/mod-repo"
        _write_and_commit(repo, {"m.py": "def f():\n    return 1\n"}, "c1")
        _full_index(repo, repo_url, graph_writer, pgvector_store, tmp_path / "out")

        old_syms = reader.get_symbols_for_files("mod-repo", ["m.py"])

        _write_and_commit(repo, {"m.py": "def f():\n    return 999\n"}, "modify")
        result = updater.update(repo, repo_url, tmp_path / "out")

        assert result.updated > 0

        # body_hash must differ from before
        new_syms = reader.get_symbols_for_files("mod-repo", ["m.py"])
        for moniker in set(old_syms) & set(new_syms):
            assert old_syms[moniker] != new_syms[moniker]

    # ── Test 4: unchanged symbols skipped — no re-embed ────────────────────

    def test_unchanged_symbols_skipped(
        self, tmp_path, graph_writer, reader, state_store, pgvector_store
    ):
        """
        When two files exist but only one changes, the unchanged file's symbols
        must not be re-embedded.

        We verify by tracking which chunks were embedded.  If the function name
        'stable' (from unchanged.py) appears in any embedded chunk, the file
        was incorrectly re-embedded.
        """
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/skip-repo"
        _write_and_commit(
            repo,
            {
                "unchanged.py": "def stable(): pass\n",
                "changing.py": "def evolving():\n    return 1\n",
            },
            "c1",
        )
        _full_index(repo, repo_url, graph_writer, pgvector_store, tmp_path / "out")

        _write_and_commit(repo, {"changing.py": "def evolving():\n    return 2\n"}, "c2")

        embedded_chunks: list[str] = []

        class TrackingEmbedder(FixtureEmbedder):
            def embed(self, chunks):
                embedded_chunks.extend(chunks)
                return super().embed(chunks)

        tracking_updater = IncrementalUpdater(
            adapters=_adapters(),
            embedder=TrackingEmbedder(dims=_TEST_DIMS),
            graph_writer=graph_writer,
            reader=reader,
            state_store=state_store,
            pgvector_store=pgvector_store,
            exporter=JsonGraphExporter(),
        )

        result = tracking_updater.update(repo, repo_url, tmp_path / "out")

        assert result.updated > 0   # changing.py symbols updated
        assert result.fell_back_to_full_index is False

        for chunk in embedded_chunks:
            assert "stable" not in chunk, (
                f"'stable' from unchanged.py appeared in an embedded chunk:\n{chunk}"
            )

        assert any("evolving" in c for c in embedded_chunks), (
            "Expected evolving() from changing.py to be embedded"
        )

    # ── Test 5: deleted file — symbols removed ──────────────────────────────

    def test_deleted_file_symbols_removed(
        self, tmp_path, updater, graph_writer, pgvector_store, reader
    ):
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/del-repo"
        _write_and_commit(repo, {"doomed.py": "def bye(): pass\n"}, "c1")
        _full_index(repo, repo_url, graph_writer, pgvector_store, tmp_path / "out")

        # Confirm symbols exist before deletion
        syms_before = reader.get_symbols_for_files("del-repo", ["doomed.py"])
        assert len(syms_before) > 0
        monikers_before = list(syms_before.keys())

        _delete_and_commit(repo, ["doomed.py"], "delete doomed")
        updater.update(repo, repo_url, tmp_path / "out")

        # Symbols must be gone from the graph
        syms_after = reader.get_symbols_for_files("del-repo", ["doomed.py"])
        assert len(syms_after) == 0

        # Symbols must be gone from pgvector
        query_vec = np.zeros(_TEST_DIMS, dtype=np.float32)
        results = pgvector_store.search(query_vec, k=100)
        returned_monikers = {r[0] for r in results}
        for m in monikers_before:
            assert m not in returned_monikers

    # ── Test 6: stale CALLS edges removed + new ones present ────────────────

    def test_stale_calls_edges_removed(
        self, tmp_path, updater, graph_writer, pgvector_store, reader
    ):
        """
        A function gains then loses a call to another function.
        After the update, the stale CALLS edge must be gone.
        """
        import psycopg2
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/calls-repo"

        # v1: caller() calls helper()
        _write_and_commit(
            repo,
            {
                "funcs.py": (
                    "def helper():\n    return 42\n\n"
                    "def caller():\n    return helper()\n"
                ),
            },
            "v1",
        )
        _full_index(repo, repo_url, graph_writer, pgvector_store, tmp_path / "out")

        # v2: caller() no longer calls helper()
        _write_and_commit(
            repo,
            {
                "funcs.py": (
                    "def helper():\n    return 42\n\n"
                    "def caller():\n    return 99\n"
                ),
            },
            "v2",
        )
        updater.update(repo, repo_url, tmp_path / "out")

        # Query the SQL relationships table for CALLS edges sourced from caller's moniker.
        conn = psycopg2.connect(PG_URL)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s_target.name
                FROM sutra_relationships r
                JOIN sutra_symbols s_src ON s_src.moniker = r.source_id
                JOIN sutra_symbols s_target ON s_target.moniker = r.target_id
                WHERE s_src.name = 'caller'
                  AND r.kind = 'calls'
                """
            )
            rows = cur.fetchall()
        conn.close()

        # caller no longer calls helper — edge must be gone
        called_names = {row[0] for row in rows}
        assert "helper" not in called_names

    # ── Test 7: renamed file — delete + add ─────────────────────────────────

    def test_renamed_file(
        self, tmp_path, updater, graph_writer, pgvector_store, reader
    ):
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/rename-repo"
        _write_and_commit(repo, {"old.py": "def old_func(): pass\n"}, "c1")
        _full_index(repo, repo_url, graph_writer, pgvector_store, tmp_path / "out")

        # Rename: delete old.py, add new.py (same content — wasted re-embed, by design)
        _git(repo, "mv", "old.py", "new.py")
        _git(repo, "commit", "-m", "rename old→new")

        result = updater.update(repo, repo_url, tmp_path / "out")

        # old.py symbols gone
        old_syms = reader.get_symbols_for_files("rename-repo", ["old.py"])
        assert len(old_syms) == 0

        # new.py symbols present
        new_syms = reader.get_symbols_for_files("rename-repo", ["new.py"])
        assert len(new_syms) > 0

    # ── Test 8: unknown last SHA → full re-index ────────────────────────────

    def test_unknown_last_sha_triggers_full_reindex(
        self, tmp_path, updater, graph_writer, pgvector_store, reader
    ):
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/norepo-fallback"
        _write_and_commit(repo, {"src.py": "def x(): pass\n"}, "c1")

        # Do NOT full-index first — Repository row doesn't exist.
        result = updater.update(repo, repo_url, tmp_path / "out")

        assert result.fell_back_to_full_index is True
        assert result.old_commit_sha is None

        syms = reader.get_symbols_for_files("norepo-fallback", ["src.py"])
        assert len(syms) > 0

    # ── Test 9: gitignored file in diff → skipped ───────────────────────────

    def test_gitignored_file_skipped(
        self, tmp_path, graph_writer, reader, state_store, pgvector_store
    ):
        """
        A gitignored file that is force-added to git (git add -f) still
        appears in git diff, but the updater's GitignoreFilter must prevent
        it from being indexed.
        """
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/ignore-repo"

        _write_and_commit(
            repo,
            {
                ".gitignore": "vendor/\n",
                "src.py": "def real(): pass\n",
            },
            "c1",
        )
        _full_index(repo, repo_url, graph_writer, pgvector_store, tmp_path / "out")

        # Force-add a gitignored file so it appears in the git diff.
        vendor_py = repo / "vendor" / "lib.py"
        vendor_py.parent.mkdir(parents=True, exist_ok=True)
        vendor_py.write_text("def vendored(): pass\n")
        _git(repo, "add", "-f", "vendor/lib.py")
        _git(repo, "commit", "-m", "force-add vendor")

        gitignore_filter = GitignoreFilter(repo)
        upd = IncrementalUpdater(
            adapters=_adapters(),
            embedder=_embedder(),
            graph_writer=graph_writer,
            reader=reader,
            state_store=state_store,
            pgvector_store=pgvector_store,
            gitignore_filter=gitignore_filter,
            exporter=JsonGraphExporter(),
        )
        upd.update(repo, repo_url, tmp_path / "out")

        # vendor/lib.py symbols must NOT be indexed despite being in the diff
        syms = reader.get_symbols_for_files("ignore-repo", ["vendor/lib.py"])
        assert len(syms) == 0

    # ── Test 10: threshold exceeded → full re-index ─────────────────────────

    def test_threshold_exceeded_triggers_full_reindex(
        self, tmp_path, graph_writer, reader, state_store, pgvector_store
    ):
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/threshold-repo"

        _write_and_commit(
            repo,
            {
                "a.py": "def fa(): pass\n",
                "b.py": "def fb(): pass\n",
                "c.py": "def fc(): pass\n",
                "d.py": "def fd(): pass\n",
            },
            "base",
        )
        _full_index(repo, repo_url, graph_writer, pgvector_store, tmp_path / "out")

        # Modify 3 of 4 files (75% > 50% threshold)
        _write_and_commit(
            repo,
            {
                "a.py": "def fa(): return 1\n",
                "b.py": "def fb(): return 2\n",
                "c.py": "def fc(): return 3\n",
            },
            "mass change",
        )

        upd = IncrementalUpdater(
            adapters=_adapters(),
            embedder=_embedder(),
            graph_writer=graph_writer,
            reader=reader,
            state_store=state_store,
            pgvector_store=pgvector_store,
            exporter=JsonGraphExporter(),
            fallback_threshold=0.5,
        )
        result = upd.update(repo, repo_url, tmp_path / "out")

        assert result.fell_back_to_full_index is True

    # ── Test 11: mid-run failure recovery ────────────────────────────────────

    def test_mid_run_failure_recovery(
        self, tmp_path, graph_writer, reader, state_store, pgvector_store
    ):
        """
        Simulate a failure after processing the first file.
        Assert:
          (a) commit_sha advances to sha2 because partial success advances
              the per-file state and update_commit_sha is called LAST.
          (b) Re-running with no failure completes and is a no-op.
        """
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/recovery-repo"
        sha1 = _write_and_commit(
            repo,
            {"x.py": "def x(): pass\n", "y.py": "def y(): pass\n"},
            "base",
        )
        _full_index(repo, repo_url, graph_writer, pgvector_store, tmp_path / "out")

        sha2 = _write_and_commit(
            repo,
            {"x.py": "def x(): return 1\n", "y.py": "def y(): return 2\n"},
            "update both",
        )

        # Patch _process_file to fail after first call
        processed_files: list[str] = []
        orig_process = IncrementalUpdater._process_file

        def failing_process(self, root, rel_path, repo_name, indexed_at):
            if len(processed_files) >= 1:
                raise RuntimeError("Simulated mid-run failure")
            processed_files.append(rel_path)
            return orig_process(self, root, rel_path, repo_name, indexed_at)

        upd = IncrementalUpdater(
            adapters=_adapters(),
            embedder=_embedder(),
            graph_writer=graph_writer,
            reader=reader,
            state_store=state_store,
            pgvector_store=pgvector_store,
            exporter=JsonGraphExporter(),
            fallback_threshold=1.0,  # disable threshold — test incremental path
        )
        upd._process_file = lambda *a, **kw: failing_process(upd, *a, **kw)

        result = upd.update(repo, repo_url, tmp_path / "out")

        assert len(result.failed_files) > 0

        sha_now = state_store.get_last_commit_sha("recovery-repo")
        assert sha_now == sha2

        # (b) Re-running (same SHA, different updater) → no-op now
        upd2 = IncrementalUpdater(
            adapters=_adapters(),
            embedder=_embedder(),
            graph_writer=graph_writer,
            reader=reader,
            state_store=state_store,
            pgvector_store=pgvector_store,
            exporter=JsonGraphExporter(),
        )
        result2 = upd2.update(repo, repo_url, tmp_path / "out")
        assert result2.added == 0
        assert result2.updated == 0
        assert result2.skipped >= 0

    # ── Test 12: deleted moniker gone from pgvector search ──────────────────

    def test_deleted_symbol_gone_from_pgvector(
        self, tmp_path, updater, graph_writer, pgvector_store, reader
    ):
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/pvdel-repo"
        _write_and_commit(repo, {"gone.py": "def vanish(): pass\n"}, "c1")
        _full_index(repo, repo_url, graph_writer, pgvector_store, tmp_path / "out")

        before = reader.get_symbols_for_files("pvdel-repo", ["gone.py"])
        assert before

        _delete_and_commit(repo, ["gone.py"], "delete gone")
        updater.update(repo, repo_url, tmp_path / "out")

        query = np.zeros(_TEST_DIMS, dtype=np.float32)
        results = {r[0] for r in pgvector_store.search(query, k=100)}
        for m in before:
            assert m not in results

    # ── Test 13: idempotent re-run ──────────────────────────────────────────

    def test_idempotent_rerun(
        self, tmp_path, updater, graph_writer, pgvector_store, reader
    ):
        """Running update twice on the same committed state is a no-op."""
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/idem-repo"
        _write_and_commit(repo, {"a.py": "def f(): pass\n"}, "c1")
        _full_index(repo, repo_url, graph_writer, pgvector_store, tmp_path / "out")
        _write_and_commit(repo, {"a.py": "def f(): return 1\n"}, "c2")

        r1 = updater.update(repo, repo_url, tmp_path / "out")
        syms1 = reader.get_symbols_for_files("idem-repo", ["a.py"])

        r2 = updater.update(repo, repo_url, tmp_path / "out")
        syms2 = reader.get_symbols_for_files("idem-repo", ["a.py"])

        assert r2.added == 0 and r2.updated == 0 and r2.deleted == 0
        assert syms1 == syms2

    # ── Test 14: empty diff still patches commit_sha ────────────────────────

    def test_empty_diff_patches_commit_sha(
        self, tmp_path, updater, graph_writer, pgvector_store, reader, state_store
    ):
        """
        If a commit changes only non-indexable files (e.g. README.md),
        the diff is empty but commit_sha must still advance.
        Otherwise the next run re-diffs the same range again.
        """
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/emptydiff-repo"
        sha1 = _write_and_commit(repo, {"src.py": "def f(): pass\n"}, "c1")
        _full_index(repo, repo_url, graph_writer, pgvector_store, tmp_path / "out")

        sha2 = _write_and_commit(repo, {"README.md": "# hello\n"}, "docs only")

        result = updater.update(repo, repo_url, tmp_path / "out")

        assert result.added == 0
        assert result.updated == 0
        assert result.deleted == 0
        assert result.fell_back_to_full_index is False
        sha_now = state_store.get_last_commit_sha("emptydiff-repo")
        assert sha_now == sha2
        assert result.new_commit_sha == sha2
        assert result.old_commit_sha == sha1

    # ── Test 15: UpdateResult fell_back flag is honest ──────────────────────

    def test_fell_back_to_full_index_flag(
        self, tmp_path, graph_writer, reader, state_store, pgvector_store
    ):
        """UpdateResult.fell_back_to_full_index is False for normal incremental."""
        repo = _init_repo(tmp_path)
        repo_url = "https://github.com/test/nofallback-repo"
        _write_and_commit(repo, {"f.py": "def f(): pass\n"}, "c1")
        _full_index(repo, repo_url, graph_writer, pgvector_store, tmp_path / "out")
        _write_and_commit(repo, {"f.py": "def f(): return 1\n"}, "c2")

        upd = IncrementalUpdater(
            adapters=_adapters(),
            embedder=_embedder(),
            graph_writer=graph_writer,
            reader=reader,
            state_store=state_store,
            pgvector_store=pgvector_store,
            exporter=JsonGraphExporter(),
        )
        result = upd.update(repo, repo_url, tmp_path / "out")
        assert result.fell_back_to_full_index is False

    # ── Test 16: embed([]) is safe (empty input guard) ──────────────────────

    def test_embed_empty_list_noop(self):
        """FixtureEmbedder.embed([]) must return shape (0, dims), not raise."""
        emb = FixtureEmbedder(dims=_TEST_DIMS)
        result = emb.embed([])
        assert result.shape == (0, _TEST_DIMS)
        assert result.dtype == np.float32
