"""
Priority 6 — End-to-end gating test (hermetic).

Indexes tests/fixtures/sample_python_repo/ — a checked-in fixture whose
content is fully controlled.  All assertions are exact: symbol counts,
relationship counts by kind, specific monikers, embedding contract.

Snapshot: graph.json (minus generated_at) is compared against
tests/fixtures/sample_python_repo_expected.json using the UPDATE_SNAPSHOTS
env flag.

    First run:   UPDATE_SNAPSHOTS=1 pytest tests/test_e2e_fixture_repo.py
    Thereafter:  pytest tests/test_e2e_fixture_repo.py
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
# Paths
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE_REPO = _FIXTURES / "sample_python_repo"
_SNAPSHOT_PATH = _FIXTURES / "sample_python_repo_expected.json"
_REPO_URL = "https://github.com/test/sample_python_repo"

# ---------------------------------------------------------------------------
# Fixture repo contents (verified by reading the source manually)
#
#   src/services/user.py:
#     ModuleSymbol            × 1
#     ClassSymbol             × 2  (Base, UserService)
#     MethodSymbol            × 2  (__init__, create_user)
#     FunctionSymbol          × 2  (_generate_id, bootstrap)
#     VariableSymbol          × 1  (MAX_RETRIES)
#
#   Relationships:
#     CONTAINS                × 7  (5 module→top-level + 2 class→method)
#     EXTENDS                 × 1  (UserService → Base)
#     CALLS                   × 3  (create_user→_generate_id, bootstrap→UserService, bootstrap→create_user)
#     IMPORTS                 × 2  (from pathlib import Path, import os)
# ---------------------------------------------------------------------------

EXPECTED_SYMBOL_COUNT = 8
EXPECTED_REL_COUNTS = {
    "contains": 7,
    "extends": 1,
    "calls": 3,
    "imports": 2,
}
EXPECTED_EMBEDDABLE_COUNT = 4   # Base, UserService, __init__, create_user, _generate_id, bootstrap
# Embeddable: ClassSymbol (Base, UserService) + MethodSymbol (__init__, create_user)
#           + FunctionSymbol (_generate_id, bootstrap)
# = 2 + 2 + 2 = 6
EXPECTED_EMBEDDABLE_COUNT = 6


# ---------------------------------------------------------------------------
# Shared indexer result
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def indexed(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("e2e_output")
    indexer = Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
    )
    result = indexer.index(root=_FIXTURE_REPO, repo_url=_REPO_URL, output_dir=out)
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
# Tests: output files
# ---------------------------------------------------------------------------

class TestOutputFiles:
    def test_all_three_files_created(self, indexed) -> None:
        out = indexed["out"]
        assert (out / "graph.json").exists()
        assert (out / "embeddings.npy").exists()
        assert (out / "embeddings_index.json").exists()

    def test_no_extra_files(self, indexed) -> None:
        out = indexed["out"]
        created = {p.name for p in out.iterdir() if p.is_file()}
        assert created == {"graph.json", "embeddings.npy", "embeddings_index.json"}


# ---------------------------------------------------------------------------
# Tests: graph.json schema
# ---------------------------------------------------------------------------

class TestGraphSchema:
    def test_required_top_level_keys(self, indexed) -> None:
        graph = indexed["graph"]
        for key in ("sutra_version", "schema_version", "generated_at",
                    "repository", "symbols", "relationships", "files",
                    "embeddings", "failed_files"):
            assert key in graph, f"Missing top-level key: {key!r}"

    def test_repository_block(self, indexed) -> None:
        repo = indexed["graph"]["repository"]
        assert repo["name"] == "sample_python_repo"
        assert repo["url"] == _REPO_URL
        assert "commit_sha" in repo
        assert "languages" in repo
        assert repo["languages"] == {"python": 1}

    def test_graph_json_roundtrips_cleanly(self, indexed) -> None:
        """
        No non-serialisable types (Path, enum, raw dataclass) should leak in.
        If json.dumps raises, the exporter has a serialisation bug.
        """
        raw = json.dumps(indexed["graph"])
        assert json.loads(raw) == indexed["graph"]

    def test_no_failed_files(self, indexed) -> None:
        assert indexed["graph"]["failed_files"] == []
        assert indexed["result"].failed_files == []


# ---------------------------------------------------------------------------
# Tests: exact symbol counts
# ---------------------------------------------------------------------------

class TestSymbolCounts:
    def test_total_symbol_count(self, indexed) -> None:
        assert len(indexed["graph"]["symbols"]) == EXPECTED_SYMBOL_COUNT

    def test_symbol_kind_breakdown(self, indexed) -> None:
        kinds = Counter(s["kind"] for s in indexed["graph"]["symbols"])
        assert kinds["module"] == 1
        assert kinds["class"] == 2
        assert kinds["method"] == 2
        assert kinds["function"] == 2
        assert kinds["variable"] == 1

    def test_all_monikers_valid(self, indexed) -> None:
        for sym in indexed["graph"]["symbols"]:
            assert is_valid_moniker(sym["id"]), (
                f"Invalid moniker: {sym['id']!r}"
            )

    def test_known_module_moniker(self, indexed) -> None:
        ids = {s["id"] for s in indexed["graph"]["symbols"]}
        expected = (
            "sutra python sample_python_repo "
            "src/services/user.py src/services/user/"
        )
        assert expected in ids

    def test_known_constructor_moniker(self, indexed) -> None:
        ids = {s["id"] for s in indexed["graph"]["symbols"]}
        expected = (
            "sutra python sample_python_repo "
            "src/services/user.py UserService#__init__()."
        )
        assert expected in ids

    def test_known_variable_moniker(self, indexed) -> None:
        ids = {s["id"] for s in indexed["graph"]["symbols"]}
        expected = (
            "sutra python sample_python_repo "
            "src/services/user.py MAX_RETRIES."
        )
        assert expected in ids

    def test_constructor_fields(self, indexed) -> None:
        syms = {s["id"]: s for s in indexed["graph"]["symbols"]}
        init = syms[
            "sutra python sample_python_repo "
            "src/services/user.py UserService#__init__()."
        ]
        assert init["kind"] == "method"
        assert init["is_constructor"] is True
        assert init["enclosing_class_id"] == (
            "sutra python sample_python_repo src/services/user.py UserService#"
        )

    def test_variable_is_constant(self, indexed) -> None:
        syms = {s["id"]: s for s in indexed["graph"]["symbols"]}
        var = syms[
            "sutra python sample_python_repo src/services/user.py MAX_RETRIES."
        ]
        assert var["kind"] == "variable"
        assert var["type_annotation"] == "int"
        assert var["is_constant"] is True

    def test_class_has_base(self, indexed) -> None:
        syms = {s["id"]: s for s in indexed["graph"]["symbols"]}
        cls = syms[
            "sutra python sample_python_repo src/services/user.py UserService#"
        ]
        assert cls["base_classes"] == ["Base"]
        assert cls["docstring"] == "Manages user operations."

    def test_module_docstring(self, indexed) -> None:
        syms = {s["id"]: s for s in indexed["graph"]["symbols"]}
        mod = syms[
            "sutra python sample_python_repo "
            "src/services/user.py src/services/user/"
        ]
        assert mod["docstring"] == "User service module."

    def test_no_duplicate_monikers(self, indexed) -> None:
        ids = [s["id"] for s in indexed["graph"]["symbols"]]
        assert len(ids) == len(set(ids)), "Duplicate monikers found"

    def test_all_symbols_reference_existing_file(self, indexed) -> None:
        file_paths = {f["path"] for f in indexed["graph"]["files"]}
        for sym in indexed["graph"]["symbols"]:
            assert sym["file_path"] in file_paths, (
                f"Symbol {sym['id']!r} references unknown file {sym['file_path']!r}"
            )


# ---------------------------------------------------------------------------
# Tests: exact relationship counts
# ---------------------------------------------------------------------------

class TestRelationshipCounts:
    def test_total_relationship_count(self, indexed) -> None:
        total = sum(EXPECTED_REL_COUNTS.values())
        assert len(indexed["graph"]["relationships"]) == total

    def test_relationship_counts_by_kind(self, indexed) -> None:
        counts = Counter(r["kind"] for r in indexed["graph"]["relationships"])
        for kind, expected in EXPECTED_REL_COUNTS.items():
            assert counts[kind] == expected, (
                f"Expected {expected} {kind!r} relationships, got {counts[kind]}"
            )

    def test_all_relationship_kinds_are_valid(self, indexed) -> None:
        valid_kinds = {"contains", "extends", "implements", "imports",
                       "calls", "references", "returns_type", "parameter_type"}
        for rel in indexed["graph"]["relationships"]:
            assert rel["kind"] in valid_kinds, f"Unknown kind: {rel['kind']!r}"

    def test_extends_relationship(self, indexed) -> None:
        extends = [r for r in indexed["graph"]["relationships"] if r["kind"] == "extends"]
        assert len(extends) == 1
        rel = extends[0]
        assert rel["source_id"] == (
            "sutra python sample_python_repo src/services/user.py UserService#"
        )
        assert rel["target_name"] == "Base"
        assert rel["is_resolved"] is False

    def test_imports_relationships(self, indexed) -> None:
        imports = [r for r in indexed["graph"]["relationships"] if r["kind"] == "imports"]
        target_names = {r["target_name"] for r in imports}
        assert "Path" in target_names
        assert "os" in target_names

    def test_calls_from_bootstrap(self, indexed) -> None:
        bootstrap_id = (
            "sutra python sample_python_repo "
            "src/services/user.py bootstrap()."
        )
        calls = [
            r for r in indexed["graph"]["relationships"]
            if r["kind"] == "calls" and r["source_id"] == bootstrap_id
        ]
        assert len(calls) == 2
        target_names = {r["target_name"] for r in calls}
        assert "UserService" in target_names
        assert "create_user" in target_names

    def test_calls_from_create_user(self, indexed) -> None:
        create_user_id = (
            "sutra python sample_python_repo "
            "src/services/user.py UserService#create_user()."
        )
        calls = [
            r for r in indexed["graph"]["relationships"]
            if r["kind"] == "calls" and r["source_id"] == create_user_id
        ]
        assert len(calls) == 1
        assert calls[0]["target_name"] == "_generate_id"


# ---------------------------------------------------------------------------
# Tests: embeddings contract (three sources of truth)
# ---------------------------------------------------------------------------

class TestEmbeddingsContract:
    def test_npy_shape(self, indexed) -> None:
        assert indexed["npy"].shape == (EXPECTED_EMBEDDABLE_COUNT, DEFAULT_EMBEDDING_DIMS)

    def test_npy_dtype(self, indexed) -> None:
        assert indexed["npy"].dtype == np.float32

    def test_index_length(self, indexed) -> None:
        assert len(indexed["index_file"]) == EXPECTED_EMBEDDABLE_COUNT

    def test_graph_embeddings_count(self, indexed) -> None:
        assert indexed["graph"]["embeddings"]["count"] == EXPECTED_EMBEDDABLE_COUNT

    def test_three_sources_agree(self, indexed) -> None:
        """Row N in .npy == embeddings_index.json[N] == graph.json symbol with embedding_id N."""
        syms_by_id = {s["embedding_id"]: s["id"]
                      for s in indexed["graph"]["symbols"]
                      if s["embedding_id"] is not None}
        for row_idx, moniker in enumerate(indexed["index_file"]):
            assert syms_by_id.get(row_idx) == moniker, (
                f"Row {row_idx}: index says {moniker!r}, "
                f"graph says {syms_by_id.get(row_idx)!r}"
            )

    def test_non_embeddable_symbols_have_null_embedding_id(self, indexed) -> None:
        for sym in indexed["graph"]["symbols"]:
            if sym["kind"] in ("variable", "module"):
                assert sym["embedding_id"] is None, (
                    f"{sym['id']} should not have embedding_id"
                )

    def test_embeddable_symbols_have_integer_embedding_id(self, indexed) -> None:
        for sym in indexed["graph"]["symbols"]:
            if sym["kind"] in ("function", "method", "class"):
                assert isinstance(sym["embedding_id"], int), (
                    f"{sym['id']} missing embedding_id"
                )

    def test_embedding_ids_are_contiguous_from_zero(self, indexed) -> None:
        ids = sorted(
            s["embedding_id"]
            for s in indexed["graph"]["symbols"]
            if s["embedding_id"] is not None
        )
        assert ids == list(range(len(ids)))

    def test_all_index_monikers_are_valid(self, indexed) -> None:
        for moniker in indexed["index_file"]:
            assert is_valid_moniker(moniker), f"Invalid moniker in index: {moniker!r}"


# ---------------------------------------------------------------------------
# Snapshot test — whole graph.json minus generated_at
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_graph_matches_snapshot(self, indexed) -> None:
        """
        Compare graph.json (minus generated_at) against the checked-in snapshot.

        To regenerate:
            UPDATE_SNAPSHOTS=1 pytest tests/test_e2e_fixture_repo.py::TestSnapshot
        """
        # Exclude fields that change on every run or every commit:
        # - generated_at: wall-clock timestamp
        # - commit_sha (inside repository): the fixture dir lives inside the sutra
        #   git repo, so git rev-parse HEAD returns sutra's own HEAD — it changes
        #   on every new sutra commit and is meaningless for this fixture.
        actual = {k: v for k, v in indexed["graph"].items() if k != "generated_at"}
        if "repository" in actual:
            actual = {**actual, "repository": {
                k: v for k, v in actual["repository"].items() if k != "commit_sha"
            }}
        actual_str = json.dumps(actual, indent=2, sort_keys=True)

        if not _SNAPSHOT_PATH.exists() and not os.getenv("UPDATE_SNAPSHOTS"):
            pytest.fail(
                f"Snapshot not found: {_SNAPSHOT_PATH}\n"
                "Run: UPDATE_SNAPSHOTS=1 pytest tests/test_e2e_fixture_repo.py"
            )

        if os.getenv("UPDATE_SNAPSHOTS"):
            _SNAPSHOT_PATH.write_text(actual_str, encoding="utf-8")

        expected_str = _SNAPSHOT_PATH.read_text(encoding="utf-8")
        assert actual_str == expected_str, (
            "graph.json snapshot mismatch. "
            "If the change is intentional, regenerate with: "
            "UPDATE_SNAPSHOTS=1 pytest tests/test_e2e_fixture_repo.py"
        )
