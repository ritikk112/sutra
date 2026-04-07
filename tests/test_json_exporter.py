"""
Tests for JsonGraphExporter (Priority 5).

Strategy: build a small but complete IndexResult with every symbol type, then
verify all three output files satisfy their contracts.  No mocking — real numpy
arrays, real JSON files, real filesystem writes to a tmp_path.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from sutra.core.extractor.base import (
    ClassSymbol,
    File,
    FunctionSymbol,
    IndexResult,
    Location,
    MethodSymbol,
    ModuleSymbol,
    Parameter,
    RelationKind,
    Relationship,
    Repository,
    VariableSymbol,
    Visibility,
)
from sutra.core.output.json_graph_exporter import (
    DEFAULT_EMBEDDING_DIMS,
    SCHEMA_VERSION,
    SUTRA_VERSION,
    JsonGraphExporter,
)

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _loc(line_start: int = 1, line_end: int = 5) -> Location:
    return Location(line_start=line_start, line_end=line_end, byte_start=0, byte_end=100)


def _make_result() -> IndexResult:
    """
    One of each symbol type, two relationships, one file.
    Monikers intentionally not in alphabetical order so sorting is observable.
    """
    repo = Repository(url="https://github.com/org/myapp.git", name="myapp")

    mod = ModuleSymbol(
        id="sutra python myapp src/app.py app/",
        name="app",
        qualified_name="src.app",
        file_path="src/app.py",
        location=_loc(1, 1),
        body_hash="sha256:aaa",
        language="python",
        visibility=Visibility.PUBLIC,
        is_exported=True,
        docstring="App module.",
    )
    cls = ClassSymbol(
        id="sutra python myapp src/app.py MyClass#",
        name="MyClass",
        qualified_name="src.app.MyClass",
        file_path="src/app.py",
        location=_loc(3, 20),
        body_hash="sha256:bbb",
        language="python",
        visibility=Visibility.PUBLIC,
        is_exported=True,
        base_classes=["Base"],
        docstring="A class.",
        decorators=[],
        is_abstract=False,
    )
    method = MethodSymbol(
        id="sutra python myapp src/app.py MyClass#run().",
        name="run",
        qualified_name="src.app.MyClass.run",
        file_path="src/app.py",
        location=_loc(5, 10),
        body_hash="sha256:ccc",
        language="python",
        visibility=Visibility.PUBLIC,
        is_exported=True,
        signature="def run(self) -> None",
        parameters=[Parameter(name="self")],
        return_type="None",
        docstring=None,
        decorators=[],
        is_async=False,
        complexity=1,
        enclosing_class_id="sutra python myapp src/app.py MyClass#",
        is_static=False,
        is_constructor=False,
    )
    func = FunctionSymbol(
        id="sutra python myapp src/app.py helper().",
        name="helper",
        qualified_name="src.app.helper",
        file_path="src/app.py",
        location=_loc(22, 30),
        body_hash="sha256:ddd",
        language="python",
        visibility=Visibility.PUBLIC,
        is_exported=True,
        signature="def helper(x: int) -> str",
        parameters=[Parameter(name="x", type_annotation="int")],
        return_type="str",
        docstring="Helper fn.",
        decorators=[],
        is_async=False,
        complexity=2,
    )
    var = VariableSymbol(
        id="sutra python myapp src/app.py COUNT.",
        name="COUNT",
        qualified_name="src.app.COUNT",
        file_path="src/app.py",
        location=_loc(2, 2),
        body_hash="sha256:eee",
        language="python",
        visibility=Visibility.PUBLIC,
        is_exported=True,
        type_annotation="int",
        is_constant=True,
    )

    # Two relationships — deliberately unsorted so sort order is tested
    rel_calls = Relationship(
        source_id=method.id,
        kind=RelationKind.CALLS,
        is_resolved=False,
        target_name="helper",
        location=_loc(7, 7),
        metadata={"call_form": "direct"},
    )
    rel_contains = Relationship(
        source_id=cls.id,
        kind=RelationKind.CONTAINS,
        is_resolved=True,
        target_id=method.id,
        target_name="run",
    )

    file_obj = File(
        path="src/app.py",
        language="python",
        size_bytes=512,
        hash="sha256:fff",
    )

    return IndexResult(
        repository=repo,
        files=[file_obj],
        symbols=[func, var, cls, method, mod],  # intentionally unsorted
        relationships=[rel_calls, rel_contains],   # intentionally unsorted
        indexed_at=datetime(2026, 4, 7, tzinfo=timezone.utc),
        commit_hash="deadbeef",
        languages={"python": 1},
    )


# ---------------------------------------------------------------------------
# Tests: output files are created
# ---------------------------------------------------------------------------

class TestOutputFiles:
    def test_three_files_are_created(self, tmp_path: Path) -> None:
        JsonGraphExporter().export(_make_result(), tmp_path)
        assert (tmp_path / "graph.json").exists()
        assert (tmp_path / "embeddings.npy").exists()
        assert (tmp_path / "embeddings_index.json").exists()

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "output"
        JsonGraphExporter().export(_make_result(), out)
        assert out.is_dir()


# ---------------------------------------------------------------------------
# Tests: graph.json top-level schema
# ---------------------------------------------------------------------------

class TestGraphSchema:
    @pytest.fixture(autouse=True)
    def _export(self, tmp_path: Path) -> None:
        self.out = tmp_path
        JsonGraphExporter().export(_make_result(), tmp_path)
        with open(tmp_path / "graph.json", encoding="utf-8") as fh:
            self.graph = json.load(fh)

    def test_sutra_version_present(self) -> None:
        assert self.graph["sutra_version"] == SUTRA_VERSION

    def test_schema_version_present_and_independent(self) -> None:
        # schema_version is a separate field — consumers care about this, not tool version
        assert self.graph["schema_version"] == SCHEMA_VERSION
        assert self.graph["schema_version"] != self.graph["sutra_version"]

    def test_generated_at_present(self) -> None:
        assert "generated_at" in self.graph
        # Must be parseable as ISO datetime
        datetime.fromisoformat(self.graph["generated_at"])

    def test_repository_block(self) -> None:
        repo = self.graph["repository"]
        assert repo["name"] == "myapp"
        assert repo["url"] == "https://github.com/org/myapp.git"
        assert repo["commit_sha"] == "deadbeef"
        assert repo["languages"] == {"python": 1}

    def test_top_level_sections_present(self) -> None:
        for key in ("symbols", "relationships", "files", "embeddings"):
            assert key in self.graph

    def test_embeddings_metadata_block(self) -> None:
        emb = self.graph["embeddings"]
        assert emb["file"] == "embeddings.npy"
        assert emb["index_file"] == "embeddings_index.json"
        assert emb["dims"] == DEFAULT_EMBEDDING_DIMS
        assert emb["dtype"] == "float32"
        # count == number of embeddable symbols (func, class, method — 3 total)
        assert emb["count"] == 3

    def test_files_block(self) -> None:
        assert len(self.graph["files"]) == 1
        f = self.graph["files"][0]
        assert f["path"] == "src/app.py"
        assert f["language"] == "python"
        assert f["size_bytes"] == 512
        assert f["hash"] == "sha256:fff"


# ---------------------------------------------------------------------------
# Tests: symbol serialisation and kinds
# ---------------------------------------------------------------------------

class TestSymbolSerialisation:
    @pytest.fixture(autouse=True)
    def _export(self, tmp_path: Path) -> None:
        JsonGraphExporter().export(_make_result(), tmp_path)
        with open(tmp_path / "graph.json", encoding="utf-8") as fh:
            graph = json.load(fh)
        self.syms = {s["id"]: s for s in graph["symbols"]}

    def test_all_five_symbols_present(self) -> None:
        assert len(self.syms) == 5

    def test_module_symbol_kind(self) -> None:
        s = self.syms["sutra python myapp src/app.py app/"]
        assert s["kind"] == "module"
        assert s["docstring"] == "App module."
        assert s["embedding_id"] is None  # not embeddable

    def test_class_symbol_kind(self) -> None:
        s = self.syms["sutra python myapp src/app.py MyClass#"]
        assert s["kind"] == "class"
        assert s["base_classes"] == ["Base"]
        assert s["is_abstract"] is False
        assert isinstance(s["embedding_id"], int)  # embeddable

    def test_method_symbol_kind(self) -> None:
        s = self.syms["sutra python myapp src/app.py MyClass#run()."]
        assert s["kind"] == "method"
        assert s["enclosing_class_id"] == "sutra python myapp src/app.py MyClass#"
        assert s["is_static"] is False
        assert s["is_constructor"] is False
        assert s["signature"] == "def run(self) -> None"
        assert isinstance(s["embedding_id"], int)

    def test_function_symbol_kind(self) -> None:
        s = self.syms["sutra python myapp src/app.py helper()."]
        assert s["kind"] == "function"
        assert s["return_type"] == "str"
        assert s["complexity"] == 2
        assert len(s["parameters"]) == 1
        assert s["parameters"][0]["name"] == "x"
        assert s["parameters"][0]["type_annotation"] == "int"
        assert isinstance(s["embedding_id"], int)

    def test_variable_symbol_kind(self) -> None:
        s = self.syms["sutra python myapp src/app.py COUNT."]
        assert s["kind"] == "variable"
        assert s["type_annotation"] == "int"
        assert s["is_constant"] is True
        assert s["embedding_id"] is None  # not embeddable

    def test_visibility_serialised_as_string(self) -> None:
        for s in self.syms.values():
            assert isinstance(s["visibility"], str)

    def test_location_fields_present(self) -> None:
        for s in self.syms.values():
            loc = s["location"]
            for field in ("line_start", "line_end", "byte_start", "byte_end",
                          "column_start", "column_end"):
                assert field in loc


# ---------------------------------------------------------------------------
# Tests: stable ordering
# ---------------------------------------------------------------------------

class TestOrdering:
    @pytest.fixture(autouse=True)
    def _export(self, tmp_path: Path) -> None:
        JsonGraphExporter().export(_make_result(), tmp_path)
        with open(tmp_path / "graph.json", encoding="utf-8") as fh:
            self.graph = json.load(fh)

    def test_symbols_sorted_by_id(self) -> None:
        ids = [s["id"] for s in self.graph["symbols"]]
        assert ids == sorted(ids)

    def test_relationships_sorted_by_source_id_then_target_name(self) -> None:
        rels = self.graph["relationships"]
        keys = [(r["source_id"], r["target_name"] or "") for r in rels]
        assert keys == sorted(keys)

    def test_files_sorted_by_path(self) -> None:
        paths = [f["path"] for f in self.graph["files"]]
        assert paths == sorted(paths)


# ---------------------------------------------------------------------------
# Tests: relationship serialisation
# ---------------------------------------------------------------------------

class TestRelationshipSerialisation:
    @pytest.fixture(autouse=True)
    def _export(self, tmp_path: Path) -> None:
        JsonGraphExporter().export(_make_result(), tmp_path)
        with open(tmp_path / "graph.json", encoding="utf-8") as fh:
            self.rels = json.load(fh)["relationships"]

    def test_two_relationships_present(self) -> None:
        assert len(self.rels) == 2

    def test_unresolved_relationship(self) -> None:
        rel = next(r for r in self.rels if r["kind"] == "calls")
        assert rel["is_resolved"] is False
        assert rel["target_id"] is None
        assert rel["target_name"] == "helper"
        assert rel["metadata"]["call_form"] == "direct"
        assert rel["location"] is not None

    def test_resolved_relationship(self) -> None:
        rel = next(r for r in self.rels if r["kind"] == "contains")
        assert rel["is_resolved"] is True
        assert rel["target_id"] == "sutra python myapp src/app.py MyClass#run()."
        assert rel["location"] is None  # not set on this relationship

    def test_kind_serialised_as_string(self) -> None:
        for r in self.rels:
            assert isinstance(r["kind"], str)


# ---------------------------------------------------------------------------
# Tests: embeddings contract (three sources of truth must agree)
# ---------------------------------------------------------------------------

class TestEmbeddingsContract:
    @pytest.fixture(autouse=True)
    def _export(self, tmp_path: Path) -> None:
        self.out = tmp_path
        JsonGraphExporter().export(_make_result(), tmp_path)
        with open(tmp_path / "graph.json", encoding="utf-8") as fh:
            self.graph = json.load(fh)
        with open(tmp_path / "embeddings_index.json", encoding="utf-8") as fh:
            self.index = json.load(fh)
        self.array = np.load(tmp_path / "embeddings.npy")

    def test_npy_shape(self) -> None:
        # 3 embeddable symbols (class, method, function) × 384 dims
        assert self.array.shape == (3, DEFAULT_EMBEDDING_DIMS)

    def test_npy_dtype(self) -> None:
        assert self.array.dtype == np.float32

    def test_index_length_matches_count(self) -> None:
        assert len(self.index) == 3
        assert self.graph["embeddings"]["count"] == 3

    def test_three_sources_agree(self) -> None:
        """
        Row N in .npy == embeddings_index.json[N] (moniker)
                      == graph.json symbol with embedding_id == N
        """
        syms_by_embedding_id = {
            s["embedding_id"]: s["id"]
            for s in self.graph["symbols"]
            if s["embedding_id"] is not None
        }
        for row_idx, moniker in enumerate(self.index):
            # index file agrees with graph.json
            assert syms_by_embedding_id[row_idx] == moniker, (
                f"Row {row_idx}: embeddings_index.json says {moniker!r} "
                f"but graph.json has embedding_id {row_idx} → {syms_by_embedding_id[row_idx]!r}"
            )

    def test_only_embeddable_symbols_have_embedding_id(self) -> None:
        for s in self.graph["symbols"]:
            if s["kind"] in ("function", "class", "method"):
                assert isinstance(s["embedding_id"], int), (
                    f"{s['id']} is embeddable but has embedding_id={s['embedding_id']!r}"
                )
            else:
                assert s["embedding_id"] is None, (
                    f"{s['id']} is not embeddable but has embedding_id={s['embedding_id']!r}"
                )

    def test_embedding_ids_are_contiguous_from_zero(self) -> None:
        ids = sorted(
            s["embedding_id"]
            for s in self.graph["symbols"]
            if s["embedding_id"] is not None
        )
        assert ids == list(range(len(ids)))


# ---------------------------------------------------------------------------
# Tests: determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_result_produces_identical_npy(self, tmp_path: Path) -> None:
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        result = _make_result()
        JsonGraphExporter().export(result, out1)
        JsonGraphExporter().export(result, out2)
        arr1 = np.load(out1 / "embeddings.npy")
        arr2 = np.load(out2 / "embeddings.npy")
        np.testing.assert_array_equal(arr1, arr2)

    def test_same_result_produces_identical_graph_symbols(self, tmp_path: Path) -> None:
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        result = _make_result()
        JsonGraphExporter().export(result, out1)
        JsonGraphExporter().export(result, out2)
        with open(out1 / "graph.json") as f1, open(out2 / "graph.json") as f2:
            g1, g2 = json.load(f1), json.load(f2)
        # generated_at will differ; compare everything else
        for key in ("symbols", "relationships", "files", "repository", "embeddings",
                    "sutra_version", "schema_version"):
            assert g1[key] == g2[key], f"Mismatch in {key!r}"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_index_result(self, tmp_path: Path) -> None:
        result = IndexResult(
            repository=Repository(url="https://github.com/org/empty.git", name="empty"),
            files=[],
            symbols=[],
            relationships=[],
            indexed_at=datetime(2026, 4, 7, tzinfo=timezone.utc),
            commit_hash="0000000",
            languages={},
        )
        JsonGraphExporter().export(result, tmp_path)
        array = np.load(tmp_path / "embeddings.npy")
        assert array.shape == (0, DEFAULT_EMBEDDING_DIMS)
        assert array.dtype == np.float32
        with open(tmp_path / "embeddings_index.json") as fh:
            assert json.load(fh) == []
        with open(tmp_path / "graph.json") as fh:
            graph = json.load(fh)
        assert graph["symbols"] == []
        assert graph["relationships"] == []
        assert graph["embeddings"]["count"] == 0

    def test_custom_embedding_dims(self, tmp_path: Path) -> None:
        result = _make_result()
        JsonGraphExporter(embedding_dims=128).export(result, tmp_path)
        array = np.load(tmp_path / "embeddings.npy")
        assert array.shape[1] == 128
        with open(tmp_path / "graph.json") as fh:
            graph = json.load(fh)
        assert graph["embeddings"]["dims"] == 128

    def test_only_non_embeddable_symbols(self, tmp_path: Path) -> None:
        result = IndexResult(
            repository=Repository(url="https://github.com/org/x.git", name="x"),
            files=[],
            symbols=[
                ModuleSymbol(
                    id="sutra python x a.py a/",
                    name="a",
                    qualified_name="a",
                    file_path="a.py",
                    location=_loc(),
                    body_hash="sha256:000",
                    language="python",
                    visibility=Visibility.PUBLIC,
                    is_exported=True,
                ),
                VariableSymbol(
                    id="sutra python x a.py V.",
                    name="V",
                    qualified_name="a.V",
                    file_path="a.py",
                    location=_loc(),
                    body_hash="sha256:111",
                    language="python",
                    visibility=Visibility.PUBLIC,
                    is_exported=True,
                ),
            ],
            relationships=[],
            indexed_at=datetime(2026, 4, 7, tzinfo=timezone.utc),
            commit_hash="abc",
            languages={"python": 1},
        )
        JsonGraphExporter().export(result, tmp_path)
        array = np.load(tmp_path / "embeddings.npy")
        assert array.shape == (0, DEFAULT_EMBEDDING_DIMS)
        with open(tmp_path / "graph.json") as fh:
            graph = json.load(fh)
        assert graph["embeddings"]["count"] == 0
        for s in graph["symbols"]:
            assert s["embedding_id"] is None
