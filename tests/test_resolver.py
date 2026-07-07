"""
Priority 20-lite — Heuristic in-repo CALLS resolver.

Hermetic: tiny real repos written to tmp_path and indexed through the real
Indexer + PythonAdapter — every rule (local / import / unique / ambiguous)
exercised on real extraction output, no synthetic Relationship objects.

Gated: acceptance on the real booth repo — ≥60% of matchable unresolved
CALLS resolved.  ("Matchable" = callee short-name exists as a repo symbol;
builtins/third-party are structurally out of reach for ANY in-repo
resolver, LSP included, and sit outside the denominator.)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.core.resolver import HeuristicResolver, ResolutionStats, Resolver
from sutra.core.resolver.heuristic import _module_matches, _module_path_of_file

_BOOTH = Path("/home/ritik/PycharmProjects/booth")


def _index(root: Path, out: Path, resolver=None):
    indexer = Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
        resolver=resolver,
    )
    return indexer.index(
        root=root, repo_url="https://github.com/test/resolver_repo", output_dir=out
    )


def _calls(result):
    from sutra.core.extractor.base import RelationKind
    return [r for r in result.relationships if r.kind == RelationKind.CALLS]


# ---------------------------------------------------------------------------
# Module-path helpers
# ---------------------------------------------------------------------------

class TestModuleMatching:
    def test_module_path_of_file(self) -> None:
        assert _module_path_of_file("a/b/c.py") == "a.b.c"
        assert _module_path_of_file("database/__init__.py") == "database"
        assert _module_path_of_file("src/lib/api-client.ts") == "src.lib.api-client"
        assert _module_path_of_file("binding/binding.go") == "binding.binding"

    @pytest.mark.parametrize("imp,mod,expected", [
        ("database", "database", True),                      # py absolute
        ("models.meeting", "models.meeting", True),
        ("database.base", "database", False),                # submodule ≠ package
        ("@/lib/api-client", "src.lib.api-client", True),    # TS alias
        ("./utils", "src.lib.utils", True),                  # TS relative
        ("github.com/gin-gonic/gin/binding", "binding", True),  # Go pkg tail
        ("other.module", "models.meeting", False),
    ])
    def test_module_matches(self, imp, mod, expected) -> None:
        assert _module_matches(imp, mod) is expected


# ---------------------------------------------------------------------------
# Resolution rules on real indexed mini-repos
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_file_repo(tmp_path: Path) -> Path:
    """helpers.py defines helper(); app.py imports and calls it."""
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "helpers.py").write_text(
        "def helper():\n    return 1\n", encoding="utf-8"
    )
    (repo / "app.py").write_text(
        "from pkg.helpers import helper\n"
        "\n"
        "def main():\n"
        "    return helper()\n",
        encoding="utf-8",
    )
    return repo


class TestRules:
    def test_local_rule_same_file_call(self, tmp_path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "mod.py").write_text(
            "def _inner():\n    return 1\n"
            "\n"
            "def outer():\n    return _inner()\n",
            encoding="utf-8",
        )
        result = _index(repo, tmp_path / "out", resolver=HeuristicResolver())
        call = next(r for r in _calls(result) if r.target_name == "_inner")
        assert call.is_resolved is True
        assert call.target_id.endswith("mod.py _inner().")
        assert call.metadata["resolved_by"] == "local"

    def test_import_rule_cross_file_call(self, two_file_repo, tmp_path) -> None:
        result = _index(two_file_repo, tmp_path / "out", resolver=HeuristicResolver())
        call = next(r for r in _calls(result) if r.target_name == "helper")
        assert call.is_resolved is True
        assert call.target_id.endswith("pkg/helpers.py helper().")
        # locals were empty, import map had it — rule must be import.
        assert call.metadata["resolved_by"] == "import"

    def test_import_rule_disambiguates_duplicates(self, tmp_path) -> None:
        """Two modules define process(); the import decides which one."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "alpha.py").write_text("def process():\n    return 'a'\n")
        (repo / "beta.py").write_text("def process():\n    return 'b'\n")
        (repo / "app.py").write_text(
            "from beta import process\n"
            "\n"
            "def main():\n    return process()\n"
        )
        result = _index(repo, tmp_path / "out", resolver=HeuristicResolver())
        call = next(r for r in _calls(result) if r.target_name == "process")
        assert call.is_resolved is True
        assert call.target_id.endswith("beta.py process().")

    def test_ambiguous_without_import_stays_unresolved(self, tmp_path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "alpha.py").write_text("def process():\n    return 'a'\n")
        (repo / "beta.py").write_text("def process():\n    return 'b'\n")
        (repo / "app.py").write_text(
            "def main():\n    return process()\n"   # no import — ambiguous
        )
        result = _index(repo, tmp_path / "out", resolver=HeuristicResolver())
        call = next(r for r in _calls(result) if r.target_name == "process")
        assert call.is_resolved is False
        assert call.target_id is None

    def test_unique_rule_repo_wide(self, tmp_path) -> None:
        """Method call with exactly one method of that name in the repo."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "dal.py").write_text(
            "class DAL:\n"
            "    def insert_record(self, row):\n"
            "        return row\n"
        )
        (repo / "svc.py").write_text(
            "def save(dal, row):\n"
            "    return dal.insert_record(row)\n"
        )
        result = _index(repo, tmp_path / "out", resolver=HeuristicResolver())
        call = next(r for r in _calls(result) if r.target_name == "insert_record")
        assert call.is_resolved is True
        assert call.target_id.endswith("DAL#insert_record().")
        assert call.metadata["resolved_by"] == "unique"

    def test_constructor_call_resolves_to_class(self, two_file_repo, tmp_path) -> None:
        (two_file_repo / "models.py").write_text(
            "class Thing:\n    pass\n", encoding="utf-8"
        )
        (two_file_repo / "factory.py").write_text(
            "from models import Thing\n"
            "\n"
            "def make():\n    return Thing()\n",
            encoding="utf-8",
        )
        result = _index(two_file_repo, tmp_path / "out", resolver=HeuristicResolver())
        call = next(r for r in _calls(result) if r.target_name == "Thing")
        assert call.is_resolved is True
        assert call.target_id.endswith("models.py Thing#")

    def test_already_resolved_untouched_and_stats(self, two_file_repo, tmp_path) -> None:
        result = _index(two_file_repo, tmp_path / "out")
        resolver = HeuristicResolver()
        # Pre-resolve one edge artificially, then run the resolver.
        call = next(r for r in _calls(result) if r.target_name == "helper")
        call.is_resolved = True
        call.target_id = "sutra python x y z()."
        stats = resolver.resolve(result.symbols, result.relationships)
        assert call.target_id == "sutra python x y z()."   # untouched
        assert isinstance(stats, ResolutionStats)
        assert stats.unresolved_before == stats.total_calls - 1

    def test_resolver_is_the_abc_seam(self) -> None:
        assert issubclass(HeuristicResolver, Resolver)


# ---------------------------------------------------------------------------
# The embedding invariant — vectors identical with and without the resolver
# ---------------------------------------------------------------------------

class TestEmbeddingInvariant:
    def test_vectors_byte_identical(self, two_file_repo, tmp_path) -> None:
        """Resolvers must never change chunks: target_name is the chunk
        input, and resolvers only touch target_id/is_resolved/metadata."""
        out_plain = tmp_path / "plain"
        out_resolved = tmp_path / "resolved"
        _index(two_file_repo, out_plain)
        _index(two_file_repo, out_resolved, resolver=HeuristicResolver())

        a = np.load(out_plain / "embeddings.npy")
        b = np.load(out_resolved / "embeddings.npy")
        assert a.shape == b.shape
        assert np.array_equal(a, b)

        # But the graphs differ exactly in resolution state.
        ga = json.loads((out_plain / "graph.json").read_text())
        gb = json.loads((out_resolved / "graph.json").read_text())
        unresolved_a = sum(
            1 for r in ga["relationships"]
            if r["kind"] == "calls" and not r["is_resolved"]
        )
        unresolved_b = sum(
            1 for r in gb["relationships"]
            if r["kind"] == "calls" and not r["is_resolved"]
        )
        assert unresolved_b < unresolved_a


# ---------------------------------------------------------------------------
# Acceptance — booth (gated on the real repo being present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _BOOTH.exists(), reason="booth repo not present")
class TestBoothAcceptance:
    def test_resolution_rate_at_least_60_percent(self, tmp_path) -> None:
        result = _index(_BOOTH, tmp_path / "out")
        stats = HeuristicResolver().resolve(result.symbols, result.relationships)
        print(
            f"\nbooth: total={stats.total_calls} unresolved={stats.unresolved_before} "
            f"matchable={stats.matchable} resolved={stats.resolved} "
            f"rate={stats.resolution_rate:.0%} by_rule={stats.by_rule}"
        )
        assert stats.matchable > 100   # sanity: the intra-repo base exists
        assert stats.resolution_rate >= 0.60, (
            f"P20-lite acceptance failed: {stats.resolution_rate:.0%} < 60% "
            f"(by_rule={stats.by_rule})"
        )


# ---------------------------------------------------------------------------
# Task 8 — Scope-gated local resolution (Tier-2 local symbols)
# ---------------------------------------------------------------------------

from sutra.core.extractor.base import FunctionSymbol, Location, Visibility, Relationship, RelationKind
from sutra.core.resolver.heuristic import HeuristicResolver


def _fn(id_, name, *, is_local=False, enclosing=None):
    return FunctionSymbol(
        id=id_, name=name, qualified_name=name, file_path="f.py",
        location=Location(1, 2, 0, 5), body_hash="sha256:x", language="python",
        visibility=Visibility.PUBLIC, is_exported=True, is_local=is_local,
        enclosing_moniker=enclosing,
    )


def _calls_rel(src, name):
    return Relationship(source_id=src, kind=RelationKind.CALLS, is_resolved=False,
                        target_name=name, metadata={"call_form": "direct"})


def test_local_scope_call_resolves_to_local_not_toplevel():
    outer = _fn("sutra python r f.py outer().", "inner_caller")  # caller is outer
    outer.name = "outer"; outer.id = "sutra python r f.py outer()."
    top_inner = _fn("sutra python r f.py inner().", "inner")
    local_inner = _fn("sutra python r f.py outer().inner().", "inner",
                      is_local=True, enclosing="sutra python r f.py outer().")
    # outer calls inner -> must resolve to the LOCAL inner, not the top-level one
    rel = _calls_rel("sutra python r f.py outer().", "inner")
    HeuristicResolver().resolve([outer, top_inner, local_inner], [rel])
    assert rel.is_resolved and rel.target_id == local_inner.id


def test_toplevel_call_not_broken_by_sibling_local():
    # A top-level caller calling top-level inner must still resolve, even though
    # a same-named local exists in the same file (regression guard).
    top_caller = _fn("sutra python r f.py main().", "main")
    top_inner = _fn("sutra python r f.py inner().", "inner")
    local_inner = _fn("sutra python r f.py outer().inner().", "inner",
                      is_local=True, enclosing="sutra python r f.py outer().")
    rel = _calls_rel("sutra python r f.py main().", "inner")
    HeuristicResolver().resolve([top_caller, top_inner, local_inner], [rel])
    assert rel.is_resolved and rel.target_id == top_inner.id


def test_local_never_resolved_cross_scope():
    # A caller outside the local's scope must NOT resolve to it (no top-level inner here).
    other = _fn("sutra python r f.py other().", "other")
    local_inner = _fn("sutra python r f.py outer().inner().", "inner",
                      is_local=True, enclosing="sutra python r f.py outer().")
    rel = _calls_rel("sutra python r f.py other().", "inner")
    HeuristicResolver().resolve([other, local_inner], [rel])
    assert not rel.is_resolved


def test_ambiguous_local_in_same_scope_stays_unresolved():
    # The caller (inner_fn) is nested inside outer_fn.
    # inner_fn's scope has TWO locals named "helper" (ambiguous — adapter would
    # disambiguate, but resolver must be defensively correct for malformed input).
    # outer_fn's scope has ONE local named "helper" (would be reached if the
    # resolver incorrectly skips past the ambiguous innermost scope).
    # A top-level NON-local "helper" exists so that, if the resolver wrongly falls
    # through to by_name, it would find and resolve to the non-local — exposing the bug.
    # Correct behaviour (precision over recall): stop at innermost ambiguous scope
    # and leave the call UNRESOLVED — never fall through to the cross-file pool.
    outer_fn = _fn("sutra python r f.py outer().", "outer")
    inner_fn = _fn("sutra python r f.py outer().inner_fn().", "inner_fn",
                   is_local=True, enclosing="sutra python r f.py outer().")
    # Two ambiguous locals inside inner_fn's scope (same enclosing + same name)
    h1 = _fn("sutra python r f.py outer().inner_fn().helper().", "helper",
             is_local=True, enclosing="sutra python r f.py outer().inner_fn().")
    h2 = _fn("sutra python r f.py outer().inner_fn().helper(1).", "helper",
             is_local=True, enclosing="sutra python r f.py outer().inner_fn().")
    # One unambiguous local in outer_fn's scope — resolver must NOT fall back here
    h_outer = _fn("sutra python r f.py outer().helper().", "helper",
                  is_local=True, enclosing="sutra python r f.py outer().")
    # Top-level NON-local: if resolver wrongly falls through to by_name it would
    # resolve to this symbol — the assertion below catches that regression.
    top_helper = _fn("sutra python r f.py helper().", "helper")
    rel = _calls_rel("sutra python r f.py outer().inner_fn().", "helper")
    HeuristicResolver().resolve([outer_fn, inner_fn, h1, h2, h_outer, top_helper], [rel])
    assert not rel.is_resolved
