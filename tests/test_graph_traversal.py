"""
Task 11: Traversal reachability — locals reachable via CONTAINS and CALLS.

Proves that the full pipeline (adapter → resolver → exporter → loader →
RustworkxTraversal) surfaces function-local symbols as graph nodes reachable
via:
  - expand_neighbors(outer, depth=1, kinds={"contains"}) → inner
  - get_callees(outer) → inner  (when outer calls inner)
"""
from __future__ import annotations

from sutra.core.artifact import ArtifactLoader
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.graph.traversal import RustworkxTraversal
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.core.resolver import HeuristicResolver


def _index(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text(
        "def outer():\n"
        "    def inner():\n"
        "        pass\n"
        "    return inner()\n"
    )
    out = tmp_path / "art"
    Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
        resolver=HeuristicResolver(),
    ).index(root=repo, repo_url="https://github.com/t/r", output_dir=out)
    return ArtifactLoader().load(out)


def test_expand_neighbors_reaches_local(tmp_path):
    snap = _index(tmp_path)
    outer = next(m for m in snap.symbols if m.endswith("outer()."))
    inner = next(m for m in snap.symbols if m.endswith("outer().inner()."))
    trav = RustworkxTraversal(snap)
    reached = {n.moniker for n in trav.expand_neighbors(outer, depth=1, kinds={"contains"})}
    assert inner in reached, (
        f"expand_neighbors did not reach {inner!r}. "
        f"Reached: {reached!r}. "
        f"All symbols: {list(snap.symbols)!r}. "
        f"Relationships: {[r for r in snap.relationships if r.get('kind') == 'contains']!r}"
    )


def test_get_callees_reaches_local(tmp_path):
    snap = _index(tmp_path)
    outer = next(m for m in snap.symbols if m.endswith("outer()."))
    inner = next(m for m in snap.symbols if m.endswith("outer().inner()."))
    trav = RustworkxTraversal(snap)
    callees = {n.moniker for n in trav.get_callees(outer)}
    assert inner in callees, (
        f"get_callees did not reach {inner!r}. "
        f"Callees: {callees!r}. "
        f"All symbols: {list(snap.symbols)!r}. "
        f"Relationships: {[r for r in snap.relationships if r.get('kind') == 'calls']!r}"
    )
