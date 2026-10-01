"""Configurable path exclusion (indexing.exclude_globs).

Example/doc trees (fastapi's docs_src/, examples/) act as ranking noise:
461 docs_src files in the fastapi index contributed several every-mode
retrieval failures (KIND_FILTER_AB.md, BATTLE_TEST.md). Test-file
exclusion is hardcoded precedent; this makes project-specific noise
dirs configurable without hardcoding anyone's layout.
"""
from __future__ import annotations

from pathlib import Path

from pipelines._common import load_exclude_globs
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "core.py").write_text("def real():\n    pass\n")
    (tmp_path / "docs_src").mkdir()
    (tmp_path / "docs_src" / "tutorial001.py").write_text("def demo():\n    pass\n")
    (tmp_path / "examples" / "deep").mkdir(parents=True)
    (tmp_path / "examples" / "deep" / "ex.py").write_text("def ex():\n    pass\n")
    return tmp_path


def _index(root: Path, out: Path, exclude_globs=()) -> set[str]:
    Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
        exclude_globs=exclude_globs,
    ).index(root=root, repo_url="https://github.com/t/r", output_dir=out)
    import json
    graph = json.loads((out / "graph.json").read_text())
    return {s["file_path"] for s in graph["symbols"]}


def test_exclude_globs_prunes_matching_paths(tmp_path):
    repo = _make_repo(tmp_path / "repo")
    files = _index(repo, tmp_path / "out", exclude_globs=("docs_src/*", "examples/*"))
    assert any(f.startswith("src/") for f in files)
    assert not any(f.startswith("docs_src/") for f in files)
    # fnmatch '*' crosses '/' — nested example files are pruned too.
    assert not any(f.startswith("examples/") for f in files)


def test_no_globs_changes_nothing(tmp_path):
    repo = _make_repo(tmp_path / "repo")
    files = _index(repo, tmp_path / "out")
    assert any(f.startswith("docs_src/") for f in files)
    assert any(f.startswith("examples/") for f in files)


def test_load_exclude_globs_from_config(tmp_path):
    cfg = tmp_path / "sutra.yaml"
    cfg.write_text(
        "embedder:\n  provider: fixture\n"
        "indexing:\n  exclude_globs:\n    - docs_src/*\n    - 'examples/*'\n"
    )
    assert load_exclude_globs(cfg) == ("docs_src/*", "examples/*")


def test_load_exclude_globs_missing_section(tmp_path):
    cfg = tmp_path / "sutra.yaml"
    cfg.write_text("embedder:\n  provider: fixture\n")
    assert load_exclude_globs(cfg) == ()
    assert load_exclude_globs(None) == ()
