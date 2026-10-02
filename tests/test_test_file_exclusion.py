"""Test-file exclusion must not swallow ordinary source files.

`*_spec.py` used to be excluded as a "pytest trailing form", but Python test
suites don't use that naming — it is RSpec/jest convention. In Python it is
ordinary source naming, and the rule silently dropped core modules from the
index. Found by the LocBench benchmark (2026-10-01): dask/_task_spec.py was
missing, so dask__dask-11608's gold functions could never be retrieved; the
same rule hid jax's partition_spec.py, keras's input_spec.py and 7 more files
across the 20 benchmark repos.
"""
from __future__ import annotations

import json
from pathlib import Path

from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter

SOURCE = {
    "dask/_task_spec.py": "class NestedContainer:\n    def to_container(self):\n        pass\n",
    "jax/_src/partition_spec.py": "class PartitionSpec:\n    pass\n",
    "keras/src/layers/input_spec.py": "class InputSpec:\n    pass\n",
    "pkg/core.py": "def real():\n    pass\n",
}
TESTS = {
    "pkg/test_core.py": "def test_real():\n    pass\n",       # pytest leading form
    "pkg/core_test.py": "def test_real2():\n    pass\n",      # pytest trailing form
    "tests/test_e2e.py": "def test_e2e():\n    pass\n",       # tests/ dir
    "pkg/tests/helpers.py": "def helper():\n    pass\n",      # nested tests/ dir
}


def _indexed_files(tmp_path: Path) -> set[str]:
    repo = tmp_path / "repo"
    for rel, text in {**SOURCE, **TESTS}.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    out = tmp_path / "out"
    Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
    ).index(root=repo, repo_url="https://github.com/t/r", output_dir=out)
    graph = json.loads((out / "graph.json").read_text())
    return {s["file_path"] for s in graph["symbols"]}


def test_spec_suffixed_source_files_are_indexed(tmp_path):
    files = _indexed_files(tmp_path)
    assert set(SOURCE) <= files


def test_python_test_files_stay_excluded(tmp_path):
    files = _indexed_files(tmp_path)
    assert not (set(TESTS) & files)
