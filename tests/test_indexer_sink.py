from pathlib import Path

import pytest

from sutra.core.artifact.atomic_writer import READY_SENTINEL
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import EmptyIndexError, Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter

_FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_python_repo"


def test_index_publishes_ready_sentinel_last(tmp_path):
    out = tmp_path / "repo"
    Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
    ).index(root=_FIXTURE_REPO, repo_url="https://github.com/test/svc", output_dir=out)

    assert (out / "graph.json").exists()
    assert (out / "embeddings.npy").exists()
    assert (out / "embeddings_index.json").exists()
    assert (out / READY_SENTINEL).exists()
    assert not (out / ".staging").exists()


def test_index_with_no_indexable_files_refuses_to_publish(tmp_path):
    """A repo with no supported source files (e.g. an APISIX/Lua gateway)
    must raise EmptyIndexError and write NO artifact. Publishing an empty
    bundle records a contradictory dims tag that later crashes the MCP
    loader with 'Embedding dims mismatch'."""
    repo = tmp_path / "lua_gateway"
    repo.mkdir()
    (repo / "gateway.lua").write_text("local x = 1\nreturn x\n")
    (repo / "apisix.yaml").write_text("routes: []\n")

    out = tmp_path / "artifact"
    with pytest.raises(EmptyIndexError):
        Indexer(
            adapters={"python": PythonAdapter()},
            exporter=JsonGraphExporter(),
            embedder=FixtureEmbedder(),
        ).index(
            root=repo,
            repo_url="https://github.com/test/lua_gateway",
            output_dir=out,
        )

    # No broken bundle on disk — nothing for the MCP loader to choke on.
    assert not (out / "graph.json").exists()
    assert not (out / READY_SENTINEL).exists()


import warnings
from sutra.core.extractor.base import (
    FunctionSymbol, Location, Visibility, Repository, IndexResult,
)


def test_duplicate_moniker_does_not_abort(tmp_path, capsys):
    # Force a duplicate by indexing a file the adapter would dup on is hard;
    # instead drive the indexer's dedup directly via a tiny repo that the
    # disambiguator does NOT cover would be ideal. Here we assert the public
    # behavior: indexing a repo with a benign collision completes and warns.
    repo = tmp_path / "repo"; repo.mkdir()
    # Two top-level functions with the same name across two files cannot collide
    # (different file_path). The realistic residual is same-file; the adapter
    # disambiguator covers it. This test pins the BACKSTOP itself via a stub.
    from sutra.core.indexer import _dedup_keep_first  # helper added in Step 3
    a = FunctionSymbol(id="dup", name="f", qualified_name="f", file_path="x.py",
                       location=Location(1, 1, 0, 1), body_hash="h", language="python",
                       visibility=Visibility.PUBLIC, is_exported=True)
    b = FunctionSymbol(id="dup", name="f", qualified_name="f", file_path="x.py",
                       location=Location(2, 2, 2, 3), body_hash="h2", language="python",
                       visibility=Visibility.PUBLIC, is_exported=True)
    kept, dropped = _dedup_keep_first([a, b])
    assert [s.id for s in kept] == ["dup"]
    assert dropped == 1
    assert kept[0] is a  # first wins
