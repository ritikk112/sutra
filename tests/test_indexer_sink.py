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


from sutra.core.extractor.base import (
    FunctionSymbol, Location, Visibility, File, FileExtraction,
)


def test_duplicate_moniker_does_not_abort():
    # Focused unit test: pin the _dedup_keep_first backstop in isolation.
    from sutra.core.indexer import _dedup_keep_first
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


def test_keep_first_warning_emitted_end_to_end(tmp_path, capsys):
    """Integration: Indexer.index() with a stub adapter emitting two symbols sharing
    one moniker — asserts (a) no raise, (b) graph has the moniker once, (c) stderr
    warning is emitted. This covers the if-dropped print path in Indexer.index()."""
    import json

    captured_ids: list[str] = []

    class _StubAdapter:
        def extract(self, rel_path, source_bytes, repo_name):
            shared_id = f"{repo_name} python {repo_name} {rel_path} dup()."
            captured_ids.append(shared_id)  # record actual id for assertion below

            def _sym(line_start, body_hash):
                return FunctionSymbol(
                    id=shared_id, name="dup", qualified_name="dup",
                    file_path=rel_path,
                    location=Location(line_start, line_start + 1, 0, 10),
                    body_hash=body_hash, language="python",
                    visibility=Visibility.PUBLIC, is_exported=True,
                )
            return FileExtraction(
                file=File(path=rel_path, language="python",
                          size_bytes=len(source_bytes), hash="sha256:x"),
                symbols=[_sym(1, "sha256:aaa"), _sym(3, "sha256:bbb")],
                relationships=[],
            )

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "myfile.py").write_text("def dup(): pass\n")

    out = tmp_path / "out"
    result = Indexer(
        adapters={"python": _StubAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
    ).index(
        root=repo,
        repo_url="https://github.com/test/stub_repo",
        output_dir=out,
    )

    # (a) no raise — Indexer.index() completed
    # (b) the moniker appears exactly once in the published graph
    graph = json.loads((out / "graph.json").read_text())
    shared_id = captured_ids[0]  # actual id emitted by the adapter
    sym_ids = [s["id"] for s in graph["symbols"]]
    assert sym_ids.count(shared_id) == 1, f"Expected 1, got: {sym_ids.count(shared_id)}"

    # (c) stderr warning was emitted
    err = capsys.readouterr().err
    assert "dropped" in err or "duplicate" in err, f"No warning in stderr: {err!r}"
