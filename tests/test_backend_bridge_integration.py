from pathlib import Path

from sutra.core.artifact.atomic_writer import READY_SENTINEL
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.extractor.moniker import repo_dir_slug, repo_name_from_url
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.mcp.registry import EmbedderCache, SnapshotRegistry, scan_artifacts_root
from sutra.mcp.watcher import ArtifactWatcher
from sutra.core.resolver.lsp_resolver import LspResolver, LspResolverError

_FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_python_repo"


def _index(repo_url: str, artifacts_root: Path) -> Path:
    out = artifacts_root / repo_dir_slug(repo_name_from_url(repo_url))
    Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
    ).index(root=_FIXTURE_REPO, repo_url=repo_url, output_dir=out)
    return out


def test_two_same_named_repos_keep_distinct_identity(tmp_path):
    root = tmp_path / "artifacts"
    _index("https://github.com/team-a/svc", root)
    _index("https://github.com/team-b/svc", root)

    # Two distinct directories, no clobber.
    assert (root / "team-a__svc" / "graph.json").exists()
    assert (root / "team-b__svc" / "graph.json").exists()

    registry = SnapshotRegistry()
    loaded = scan_artifacts_root(root, registry, EmbedderCache(), strict=True)
    assert sorted(loaded) == ["team-a/svc", "team-b/svc"]
    assert registry.get("team-a/svc") is not None
    assert registry.get("team-b/svc") is not None


def test_crashing_lsp_aborts_index_and_publishes_nothing(tmp_path):
    """Spec T6, end-to-end: a pyright crash fails index() before the publish."""
    import sys
    import pytest
    fake = [sys.executable,
            str(Path(__file__).parent / "helpers" / "fake_lsp_server.py"), "crash"]
    out = tmp_path / "artifacts" / "team-a__svc"
    with pytest.raises(LspResolverError):
        Indexer(
            adapters={"python": PythonAdapter()},
            exporter=JsonGraphExporter(),
            embedder=FixtureEmbedder(),
            resolver=LspResolver(root=_FIXTURE_REPO, command=fake),
        ).index(root=_FIXTURE_REPO, repo_url="https://github.com/team-a/svc",
                output_dir=out)
    assert not (out / READY_SENTINEL).exists()   # crash before publish -> no .ready


def test_reindex_bumps_ready_and_watcher_fires(tmp_path):
    root = tmp_path / "artifacts"
    out = _index("https://github.com/team-a/svc", root)

    # Watcher primed on the existing sentinel — no startup re-fire.
    fired: list[Path] = []
    watcher = ArtifactWatcher(root, fired.append)
    assert watcher.check_once() == []

    # Re-index the same repo: same dir, fresh .ready (new mtime).
    import os
    before = (out / READY_SENTINEL).stat().st_mtime
    _index("https://github.com/team-a/svc", root)
    os.utime(out / READY_SENTINEL,
             (before + 5, before + 5))   # guarantee a visible mtime bump
    assert watcher.check_once() == [out]
