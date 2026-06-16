"""
Priority 20-full — pyright LSP resolver.

The e2e fixture is the exact case the heuristic CANNOT solve: two classes
expose the same method name, the call goes through a typed receiver, and
no import names the method.  Heuristic → ambiguous → unresolved; pyright's
type inference → resolved.

Gated on pyright-langserver being installed (pip install pyright).  The
SQL round-trip test additionally needs SUTRA_PG_URL.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.extractor.base import RelationKind
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.core.resolver import HeuristicResolver, Resolver
from sutra.core.resolver.lsp_resolver import ChainResolver, LspResolver

_PYRIGHT = Path(sys.executable).parent / "pyright-langserver"
_PG_URL = os.environ.get("SUTRA_PG_URL", "")

pytestmark = pytest.mark.skipif(
    not _PYRIGHT.exists() and shutil.which("pyright-langserver") is None,
    reason="pyright-langserver not installed (pip install pyright)",
)


def _write_ambiguous_repo(repo: Path) -> None:
    """The heuristic-defeating shape: same method name on two classes,
    call through a TYPED receiver, no disambiguating import of the name."""
    repo.mkdir(parents=True)
    (repo / "alpha.py").write_text(
        "class AlphaWorker:\n"
        "    def run_task(self):\n"
        "        return 'alpha'\n",
        encoding="utf-8",
    )
    (repo / "beta.py").write_text(
        "class BetaWorker:\n"
        "    def run_task(self):\n"
        "        return 'beta'\n",
        encoding="utf-8",
    )
    (repo / "app.py").write_text(
        "from alpha import AlphaWorker\n"
        "\n"
        "def main():\n"
        "    worker = AlphaWorker()\n"
        "    return worker.run_task()\n",
        encoding="utf-8",
    )


def _index(repo: Path, out: Path, resolver=None):
    return Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
        resolver=resolver,
    ).index(root=repo, repo_url="https://github.com/test/lsp_repo", output_dir=out)


def _run_task_call(result):
    return next(
        r for r in result.relationships
        if r.kind == RelationKind.CALLS and r.target_name == "run_task"
    )


class TestLspResolution:
    def test_heuristic_alone_cannot_resolve_the_fixture(self, tmp_path) -> None:
        """Guard: if the heuristic ever solves this, the fixture no longer
        tests the LSP's unique contribution — rewrite it."""
        repo = tmp_path / "repo"
        _write_ambiguous_repo(repo)
        result = _index(repo, tmp_path / "out", resolver=HeuristicResolver())
        assert _run_task_call(result).is_resolved is False

    def test_lsp_resolves_typed_receiver_call(self, tmp_path) -> None:
        repo = tmp_path / "repo"
        _write_ambiguous_repo(repo)
        result = _index(
            repo, tmp_path / "out",
            resolver=ChainResolver(HeuristicResolver(), LspResolver(root=repo)),
        )
        call = _run_task_call(result)
        assert call.is_resolved is True
        assert call.target_id.endswith("alpha.py AlphaWorker#run_task().")
        assert call.metadata["resolved_by"] == "lsp"

    def test_chain_runs_heuristic_first(self, tmp_path) -> None:
        """In the chained run, easy edges carry heuristic rule stamps —
        pyright only paid for the residue."""
        repo = tmp_path / "repo"
        _write_ambiguous_repo(repo)
        # Add an easy local call the heuristic owns.
        (repo / "easy.py").write_text(
            "def helper():\n    return 1\n"
            "\n"
            "def caller():\n    return helper()\n",
            encoding="utf-8",
        )
        chain = ChainResolver(HeuristicResolver(), LspResolver(root=repo))
        result = _index(repo, tmp_path / "out", resolver=chain)

        helper_call = next(
            r for r in result.relationships
            if r.kind == RelationKind.CALLS and r.target_name == "helper"
        )
        assert helper_call.metadata["resolved_by"] == "local"
        assert _run_task_call(result).metadata["resolved_by"] == "lsp"

    def test_constructor_left_to_heuristic_not_double_resolved(self, tmp_path) -> None:
        repo = tmp_path / "repo"
        _write_ambiguous_repo(repo)
        chain = ChainResolver(HeuristicResolver(), LspResolver(root=repo))
        result = _index(repo, tmp_path / "out", resolver=chain)
        ctor = next(
            r for r in result.relationships
            if r.kind == RelationKind.CALLS and r.target_name == "AlphaWorker"
        )
        assert ctor.is_resolved is True
        assert ctor.metadata["resolved_by"] in ("import", "local", "unique")

    def test_lsp_resolver_is_the_abc_seam(self) -> None:
        assert issubclass(LspResolver, Resolver)
        assert issubclass(ChainResolver, Resolver)


@pytest.mark.skipif(not _PG_URL, reason="SUTRA_PG_URL not set")
class TestLspResolvedEdgesReachSql:
    def test_resolved_calls_rows_in_store(self, tmp_path) -> None:
        """The P20-full acceptance tail: LSP-resolved relationships appear
        in the SQL store with is_resolved=true (the writer skips
        unresolved edges, so presence == resolution)."""
        from sutra.core.graph.sql_writer import SqlGraphWriter

        repo = tmp_path / "repo"
        _write_ambiguous_repo(repo)

        with SqlGraphWriter(_PG_URL) as writer:
            writer.setup()
            indexer = Indexer(
                adapters={"python": PythonAdapter()},
                exporter=JsonGraphExporter(),
                embedder=FixtureEmbedder(),
                graph_writer=writer,
                resolver=ChainResolver(
                    HeuristicResolver(), LspResolver(root=repo)
                ),
            )
            indexer.index(
                root=repo,
                repo_url="https://github.com/test/lsp_repo",
                output_dir=tmp_path / "out",
                replace=True,
            )

            with writer._conn.cursor() as cur:
                cur.execute(
                    "SELECT target_id, is_resolved, metadata->>'resolved_by' "
                    "FROM sutra_relationships "
                    "WHERE kind = 'calls' AND source_id LIKE %s",
                    ("%app.py main().%",),
                )
                rows = cur.fetchall()
                # Clean up the test repo's rows.
                cur.execute(
                    "DELETE FROM sutra_symbols WHERE repo_name = %s",
                    ("lsp_repo",),
                )
                cur.execute(
                    "DELETE FROM sutra_relationships WHERE source_id LIKE %s",
                    ("sutra python lsp_repo %",),
                )
            writer._conn.commit()

        lsp_rows = [r for r in rows if r[2] == "lsp"]
        assert lsp_rows, f"no lsp-resolved rows reached SQL: {rows}"
        target_id, is_resolved, _ = lsp_rows[0]
        assert is_resolved is True
        assert target_id.endswith("AlphaWorker#run_task().")
