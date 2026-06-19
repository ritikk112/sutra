"""
Priority 19 (loader half) — AtomicArtifactWriter, SnapshotRegistry,
ArtifactWatcher, GraphTraversal.

All hermetic: fixture repo indexed with FixtureEmbedder + HeuristicResolver
into tmp dirs; real files, real commits, real loads.  The torn-artifact and
version-mismatch rejection paths are exercised through the same
build_serving_unit() the server uses.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sutra.core.artifact import ArtifactError, ArtifactLoader
from sutra.core.artifact.atomic_writer import (
    ARTIFACT_FILES,
    READY_SENTINEL,
    AtomicArtifactWriter,
)
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.graph.traversal import GraphTraversal, Neighbor, RustworkxTraversal
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.core.resolver import HeuristicResolver
from sutra.mcp.registry import (
    EmbedderCache,
    SnapshotRegistry,
    build_serving_unit,
    scan_artifacts_root,
)
from sutra.mcp.watcher import ArtifactWatcher

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE_REPO = _FIXTURES / "sample_python_repo"


def _index_fixture(out: Path) -> None:
    Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
        resolver=HeuristicResolver(),
    ).index(
        root=_FIXTURE_REPO,
        repo_url="https://github.com/test/sample_python_repo",
        output_dir=out,
    )


@pytest.fixture()
def artifact_dir(tmp_path: Path) -> Path:
    out = tmp_path / "sample_python_repo"
    _index_fixture(out)
    return out


# ---------------------------------------------------------------------------
# AtomicArtifactWriter
# ---------------------------------------------------------------------------

class TestAtomicArtifactWriter:
    def _commit_generation(self, target: Path, source: Path, gen: str) -> None:
        def write(staging: Path) -> None:
            for name in ARTIFACT_FILES:
                (staging / name).write_bytes((source / name).read_bytes())
        AtomicArtifactWriter().commit(target, write, generation=gen)

    def test_commit_writes_ready_last_with_generation(
        self, artifact_dir, tmp_path
    ) -> None:
        target = tmp_path / "served"
        self._commit_generation(target, artifact_dir, "gen-1")
        assert (target / READY_SENTINEL).read_text() == "gen-1"
        for name in ARTIFACT_FILES:
            assert (target / name).exists()
        # Staging area cleaned up.
        assert not (target / ".staging").exists()

    def test_second_commit_retains_prev_generation(
        self, artifact_dir, tmp_path
    ) -> None:
        target = tmp_path / "served"
        self._commit_generation(target, artifact_dir, "gen-1")
        first_graph = (target / "graph.json").read_bytes()
        self._commit_generation(target, artifact_dir, "gen-2")
        assert (target / "graph.json.prev").read_bytes() == first_graph
        assert (target / READY_SENTINEL).read_text() == "gen-2"

    def test_partial_write_refused_and_no_ready(self, tmp_path) -> None:
        target = tmp_path / "served"

        def half_write(staging: Path) -> None:
            (staging / "graph.json").write_text("{}")
            # embeddings.npy + embeddings_index.json deliberately missing

        with pytest.raises(FileNotFoundError, match="refusing a partial"):
            AtomicArtifactWriter().commit(target, half_write)
        assert not (target / READY_SENTINEL).exists()
        assert not (target / "graph.json").exists()   # nothing promoted

    def test_committed_artifact_loads_cleanly(self, artifact_dir, tmp_path) -> None:
        target = tmp_path / "served"
        self._commit_generation(target, artifact_dir, "gen-1")
        snap = ArtifactLoader().load(target)
        assert snap.repo_name == "test/sample_python_repo"


# ---------------------------------------------------------------------------
# SnapshotRegistry + build_serving_unit
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_build_and_swap_and_get(self, artifact_dir) -> None:
        registry = SnapshotRegistry()
        unit = build_serving_unit(artifact_dir, EmbedderCache())
        registry.swap(unit)
        assert registry.repos() == ["test/sample_python_repo"]
        assert registry.get("test/sample_python_repo") is unit
        assert registry.get("nope") is None

    def test_swap_replaces_old_unit_inflight_keeps_reference(
        self, artifact_dir
    ) -> None:
        registry = SnapshotRegistry()
        cache = EmbedderCache()
        old = build_serving_unit(artifact_dir, cache)
        registry.swap(old)
        held = registry.get("test/sample_python_repo")   # an in-flight query's ref
        new = build_serving_unit(artifact_dir, cache)
        registry.swap(new)
        assert registry.get("test/sample_python_repo") is new
        # The held reference still serves — immutability is the guarantee.
        assert held is old
        assert held.snapshot.repo_name == "test/sample_python_repo"

    def test_torn_artifact_rejected_old_keeps_serving(
        self, artifact_dir, tmp_path
    ) -> None:
        registry = SnapshotRegistry()
        cache = EmbedderCache()
        good = build_serving_unit(artifact_dir, cache)
        registry.swap(good)

        # Tear the directory: index shorter than the matrix.
        index = json.loads((artifact_dir / "embeddings_index.json").read_text())
        (artifact_dir / "embeddings_index.json").write_text(json.dumps(index[:-1]))

        with pytest.raises(ArtifactError, match="Torn artifact"):
            build_serving_unit(artifact_dir, cache)
        # Registry untouched — the old unit still serves.
        assert registry.get("test/sample_python_repo") is good

    def test_schema_version_mismatch_rejected(self, artifact_dir) -> None:
        graph = json.loads((artifact_dir / "graph.json").read_text())
        graph["schema_version"] = "999"
        (artifact_dir / "graph.json").write_text(json.dumps(graph))
        with pytest.raises(ArtifactError, match="Unsupported schema_version"):
            build_serving_unit(artifact_dir, EmbedderCache())

    def test_scan_skips_broken_keeps_good(self, tmp_path) -> None:
        root = tmp_path / "root"
        good = root / "sample_python_repo"
        _index_fixture(good)
        broken = root / "broken_repo"
        broken.mkdir()
        (broken / "graph.json").write_text("not json at all")

        registry = SnapshotRegistry()
        loaded = scan_artifacts_root(root, registry, EmbedderCache())
        assert loaded == ["test/sample_python_repo"]
        assert registry.repos() == ["test/sample_python_repo"]

    def test_scan_strict_raises_on_broken(self, tmp_path) -> None:
        root = tmp_path / "root"
        broken = root / "broken_repo"
        broken.mkdir(parents=True)
        (broken / "graph.json").write_text("not json")
        with pytest.raises(ArtifactError):
            scan_artifacts_root(
                root, SnapshotRegistry(), EmbedderCache(), strict=True
            )


# ---------------------------------------------------------------------------
# ArtifactWatcher
# ---------------------------------------------------------------------------

class TestWatcher:
    def test_fires_on_new_sentinel_only(self, artifact_dir, tmp_path) -> None:
        root = artifact_dir.parent
        fired: list[Path] = []
        watcher = ArtifactWatcher(root, fired.append, poll_seconds=0.01)

        # No sentinel yet → no fire.
        assert watcher.check_once() == []

        (artifact_dir / READY_SENTINEL).write_text("gen-1")
        assert watcher.check_once() == [artifact_dir]
        assert fired == [artifact_dir]

        # Unchanged sentinel → debounced, no re-fire.
        assert watcher.check_once() == []

    def test_fires_again_on_mtime_bump(self, artifact_dir) -> None:
        import os
        root = artifact_dir.parent
        sentinel = artifact_dir / READY_SENTINEL
        sentinel.write_text("gen-1")
        fired: list[Path] = []
        watcher = ArtifactWatcher(root, fired.append)
        # Constructor primes existing sentinels — no startup re-fire.
        assert watcher.check_once() == []

        sentinel.write_text("gen-2")
        os.utime(sentinel, (sentinel.stat().st_atime, sentinel.stat().st_mtime + 5))
        assert watcher.check_once() == [artifact_dir]

    def test_callback_errors_do_not_kill_watcher(self, artifact_dir) -> None:
        root = artifact_dir.parent

        def explode(_: Path) -> None:
            raise RuntimeError("torn artifact mid-reload")

        watcher = ArtifactWatcher(root, explode)
        (artifact_dir / READY_SENTINEL).write_text("gen-1")
        assert watcher.check_once() == [artifact_dir]   # survived
        # And keeps polling normally afterwards.
        assert watcher.check_once() == []


# ---------------------------------------------------------------------------
# GraphTraversal
# ---------------------------------------------------------------------------

class TestTraversal:
    @pytest.fixture(scope="class")
    def snapshot(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("traversal") / "repo"
        _index_fixture(out)
        return ArtifactLoader().load(out)

    def test_is_the_abc_seam(self, snapshot) -> None:
        assert issubclass(RustworkxTraversal, GraphTraversal)

    def test_callers_and_callees(self, snapshot) -> None:
        t = RustworkxTraversal(snapshot)
        create_user = next(m for m in snapshot.symbols if "create_user" in m)
        generate_id = next(m for m in snapshot.symbols if "_generate_id" in m)

        callees = t.get_callees(create_user)
        assert generate_id in [n.moniker for n in callees]
        assert all(isinstance(n, Neighbor) and n.direction == "out" for n in callees)

        callers = t.get_callers(generate_id)
        assert create_user in [n.moniker for n in callers]

    def test_expand_neighbors_depth_and_kinds(self, snapshot) -> None:
        t = RustworkxTraversal(snapshot)
        create_user = next(m for m in snapshot.symbols if "create_user" in m)

        one_hop = t.expand_neighbors(create_user, depth=1)
        assert one_hop
        assert all(n.depth == 1 for n in one_hop)

        two_hop = t.expand_neighbors(create_user, depth=2)
        assert len(two_hop) >= len(one_hop)

        calls_only = t.expand_neighbors(create_user, depth=2, kinds=["calls"])
        assert all(n.edge_kind == "calls" for n in calls_only)

    def test_unknown_moniker_empty(self, snapshot) -> None:
        t = RustworkxTraversal(snapshot)
        assert t.get_callers("sutra x y z nope().") == []
        assert t.expand_neighbors("sutra x y z nope().") == []
