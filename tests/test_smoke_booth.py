"""
Priority 6 — Local-only smoke test against the real booth repository.

Gated: skipped unless /home/ritik/PycharmProjects/booth exists (or SUTRA_SMOKE_REPO
is set to an alternate path).  Never runs in CI.

Purpose: catch crashes on real-world Python that the fixture doesn't exercise —
unusual syntax, large files, deeply nested structures, edge cases in the adapter.
Assertions are loose: no exact counts, just structural validity.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.extractor.moniker import is_valid_moniker
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import DEFAULT_EMBEDDING_DIMS, JsonGraphExporter

# ---------------------------------------------------------------------------
# Smoke repo location
# ---------------------------------------------------------------------------

_DEFAULT_SMOKE_REPO = "/home/ritik/PycharmProjects/booth"
_SMOKE_REPO = Path(os.getenv("SUTRA_SMOKE_REPO", _DEFAULT_SMOKE_REPO))
_BOOTH_URL = "https://github.com/ritikk112/booth"

_VALID_KINDS = frozenset({
    "contains", "extends", "implements", "imports",
    "calls", "references", "returns_type", "parameter_type",
})

pytestmark = pytest.mark.skipif(
    not _SMOKE_REPO.exists(),
    reason=f"local-only smoke test: {_SMOKE_REPO} not present",
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def booth_indexed(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("booth_output")
    indexer = Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
    )
    result = indexer.index(root=_SMOKE_REPO, repo_url=_BOOTH_URL, output_dir=out)
    with open(out / "graph.json", encoding="utf-8") as fh:
        graph = json.load(fh)
    index_file = json.loads((out / "embeddings_index.json").read_text())
    npy_array = np.load(out / "embeddings.npy")
    return {
        "result": result,
        "graph": graph,
        "index_file": index_file,
        "npy": npy_array,
        "out": out,
    }


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestSmokeStructure:
    def test_output_files_exist(self, booth_indexed) -> None:
        out = booth_indexed["out"]
        assert (out / "graph.json").exists()
        assert (out / "embeddings.npy").exists()
        assert (out / "embeddings_index.json").exists()

    def test_symbol_count_non_trivial(self, booth_indexed) -> None:
        assert len(booth_indexed["graph"]["symbols"]) > 20, (
            "Expected more than 20 symbols from a real repo"
        )

    def test_all_monikers_valid(self, booth_indexed) -> None:
        for sym in booth_indexed["graph"]["symbols"]:
            assert is_valid_moniker(sym["id"]), (
                f"Invalid moniker: {sym['id']!r}"
            )

    def test_no_duplicate_monikers(self, booth_indexed) -> None:
        ids = [s["id"] for s in booth_indexed["graph"]["symbols"]]
        dupes = [m for m, count in Counter(ids).items() if count > 1]
        assert not dupes, f"Duplicate monikers: {dupes}"

    def test_all_relationship_kinds_valid(self, booth_indexed) -> None:
        for rel in booth_indexed["graph"]["relationships"]:
            assert rel["kind"] in _VALID_KINDS, (
                f"Unknown relationship kind: {rel['kind']!r}"
            )

    def test_graph_json_roundtrips_cleanly(self, booth_indexed) -> None:
        raw = json.dumps(booth_indexed["graph"])
        assert json.loads(raw) == booth_indexed["graph"]

    def test_repository_block(self, booth_indexed) -> None:
        repo = booth_indexed["graph"]["repository"]
        assert repo["name"] == "booth"
        assert repo["url"] == _BOOTH_URL
        assert "commit_sha" in repo
        # Real git repo — commit SHA should be 40-char hex
        sha = repo["commit_sha"]
        assert sha == "unknown" or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)), (
            f"Unexpected commit_sha: {sha!r}"
        )


class TestSmokeEmbeddings:
    def test_npy_shape_matches_count(self, booth_indexed) -> None:
        count = booth_indexed["graph"]["embeddings"]["count"]
        assert booth_indexed["npy"].shape == (count, DEFAULT_EMBEDDING_DIMS)

    def test_npy_dtype(self, booth_indexed) -> None:
        assert booth_indexed["npy"].dtype == np.float32

    def test_index_length_matches_count(self, booth_indexed) -> None:
        count = booth_indexed["graph"]["embeddings"]["count"]
        assert len(booth_indexed["index_file"]) == count

    def test_three_sources_agree(self, booth_indexed) -> None:
        syms_by_id = {s["embedding_id"]: s["id"]
                      for s in booth_indexed["graph"]["symbols"]
                      if s["embedding_id"] is not None}
        for row_idx, moniker in enumerate(booth_indexed["index_file"]):
            assert syms_by_id.get(row_idx) == moniker, (
                f"Row {row_idx}: index says {moniker!r}, "
                f"graph says {syms_by_id.get(row_idx)!r}"
            )

    def test_all_index_monikers_valid(self, booth_indexed) -> None:
        for moniker in booth_indexed["index_file"]:
            assert is_valid_moniker(moniker), f"Invalid moniker in index: {moniker!r}"

    def test_non_embeddable_have_null_embedding_id(self, booth_indexed) -> None:
        for sym in booth_indexed["graph"]["symbols"]:
            if sym["kind"] in ("variable", "module"):
                assert sym["embedding_id"] is None, (
                    f"{sym['id']} should not be embeddable"
                )

    def test_embeddable_have_integer_embedding_id(self, booth_indexed) -> None:
        for sym in booth_indexed["graph"]["symbols"]:
            if sym["kind"] in ("function", "method", "class"):
                assert isinstance(sym["embedding_id"], int), (
                    f"{sym['id']} missing embedding_id"
                )
