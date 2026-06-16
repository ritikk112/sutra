"""
Priority 12 — Retrieval eval harness.

Two layers:

1. Hermetic machinery tests (always run): index the checked-in
   sample_python_repo with FixtureEmbedder, load the artifact through
   ArtifactLoader, and exercise SearchResult / BaselineRetriever /
   dataset / metrics / harness on real code paths.  FixtureEmbedder is
   deterministic (vector seeded by sha256 of the chunk text), so feeding
   a symbol's exact chunk text as the query MUST rank that symbol #1
   with cosine ≈ 1.0 — a real end-to-end assertion with no mocks.

2. Real-eval baseline gate (skipped unless artifacts exist): runs the
   checked-in datasets (tests/eval/datasets/) against artifacts built by
   scripts/build_eval_artifacts.py with the pinned local embedder, and
   asserts aggregate metrics within ±0.05 of the checked-in baseline
   snapshot (tests/eval/baselines/baseline_metrics.json).  Drift names
   the regressed queries.

   Build artifacts:   python scripts/build_eval_artifacts.py
   Update snapshot:   UPDATE_EVAL_BASELINE=1 pytest tests/test_eval_harness.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from sutra.core.artifact import ArtifactError, ArtifactLoader
from sutra.core.embedder.chunk_builder import build_chunks
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.core.retrieval import SearchResult
from sutra.core.retrieval.baseline import BaselineRetriever
from sutra.core.retrieval.eval import (
    compare,
    load_dataset,
    load_datasets,
    run_eval,
)
from sutra.core.retrieval.eval.metrics import (
    first_hit_rank,
    must_include_coverage,
    recall_at_k,
    reciprocal_rank,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE_REPO = _FIXTURES / "sample_python_repo"
_EVAL_DIR = Path(__file__).parent / "eval"
_DATASETS_DIR = _EVAL_DIR / "datasets"
_ARTIFACTS_DIR = _EVAL_DIR / "artifacts"
_BASELINE_PATH = _EVAL_DIR / "baselines" / "baseline_metrics.json"

# Tolerance for the baseline gate: ±5 percentage points absolute.
_TOLERANCE = 0.05


# ---------------------------------------------------------------------------
# Shared hermetic artifact (sample_python_repo + FixtureEmbedder)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("p12_artifact")
    embedder = FixtureEmbedder()
    indexer = Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=embedder,
    )
    result = indexer.index(
        root=_FIXTURE_REPO,
        repo_url="https://github.com/test/sample_python_repo",
        output_dir=out,
    )
    chunks, monikers = build_chunks(result.symbols, _FIXTURE_REPO, result.relationships)
    return {
        "dir": out,
        "result": result,
        "chunks": chunks,
        "monikers": monikers,
        "embedder": embedder,
    }


@pytest.fixture(scope="module")
def snapshot(artifact):
    return ArtifactLoader().load(artifact["dir"])


@pytest.fixture(scope="module")
def retriever(snapshot, artifact):
    return BaselineRetriever(snapshot, artifact["embedder"])


# ---------------------------------------------------------------------------
# SearchResult contract
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_frozen(self) -> None:
        r = SearchResult(moniker="m", score=0.5)
        with pytest.raises(AttributeError):
            r.score = 0.9  # type: ignore[misc]

    def test_provenance_defaults_empty(self) -> None:
        assert SearchResult(moniker="m", score=0.5).provenance == {}

    def test_slots(self) -> None:
        # slots=True means no per-instance __dict__ — the hot-path memory win.
        r = SearchResult(moniker="m", score=0.5)
        assert not hasattr(r, "__dict__")


# ---------------------------------------------------------------------------
# ArtifactLoader
# ---------------------------------------------------------------------------

class TestArtifactLoader:
    def test_loads_fixture_artifact(self, snapshot, artifact) -> None:
        assert snapshot.repo_name == "sample_python_repo"
        assert snapshot.schema_version == "1"
        assert snapshot.embedding_model_id == "fixture-384"
        assert snapshot.embedding_dims == 384
        assert snapshot.vectors.shape == (len(artifact["monikers"]), 384)
        assert snapshot.moniker_order == sorted(artifact["monikers"])
        # row_of inverts moniker_order
        for moniker, row in snapshot.row_of.items():
            assert snapshot.moniker_order[row] == moniker
        # every symbol moniker is loadable
        for m in snapshot.moniker_order:
            assert m in snapshot.symbols

    def test_missing_file_raises(self, artifact, tmp_path) -> None:
        import shutil
        broken = tmp_path / "broken"
        shutil.copytree(artifact["dir"], broken)
        (broken / "embeddings.npy").unlink()
        with pytest.raises(ArtifactError, match="missing embeddings.npy"):
            ArtifactLoader().load(broken)

    def test_torn_artifact_row_count_mismatch(self, artifact, tmp_path) -> None:
        import shutil
        torn = tmp_path / "torn"
        shutil.copytree(artifact["dir"], torn)
        index = json.loads((torn / "embeddings_index.json").read_text())
        (torn / "embeddings_index.json").write_text(json.dumps(index[:-1]))
        with pytest.raises(ArtifactError, match="Torn artifact"):
            ArtifactLoader().load(torn)

    def test_unsupported_schema_version(self, artifact, tmp_path) -> None:
        import shutil
        bad = tmp_path / "bad_schema"
        shutil.copytree(artifact["dir"], bad)
        graph = json.loads((bad / "graph.json").read_text())
        graph["schema_version"] = "99"
        (bad / "graph.json").write_text(json.dumps(graph))
        with pytest.raises(ArtifactError, match="Unsupported schema_version"):
            ArtifactLoader().load(bad)

    def test_embedding_id_mismatch(self, artifact, tmp_path) -> None:
        import shutil
        bad = tmp_path / "bad_eid"
        shutil.copytree(artifact["dir"], bad)
        graph = json.loads((bad / "graph.json").read_text())
        embedded = [s for s in graph["symbols"] if s["embedding_id"] is not None]
        # Swap two embedding_ids — index file no longer agrees with graph.json
        embedded[0]["embedding_id"], embedded[1]["embedding_id"] = (
            embedded[1]["embedding_id"], embedded[0]["embedding_id"],
        )
        (bad / "graph.json").write_text(json.dumps(graph))
        with pytest.raises(ArtifactError, match="embedding_id"):
            ArtifactLoader().load(bad)


# ---------------------------------------------------------------------------
# BaselineRetriever
# ---------------------------------------------------------------------------

class TestBaselineRetriever:
    def test_exact_chunk_query_ranks_first(self, retriever, artifact) -> None:
        """FixtureEmbedder is deterministic: querying with a symbol's exact
        chunk text must return that symbol at rank 1 with cosine ≈ 1.0."""
        for chunk, moniker in zip(artifact["chunks"], artifact["monikers"]):
            results = retriever.search(chunk, top_k=3)
            assert results[0].moniker == moniker
            assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_results_sorted_and_bounded(self, retriever) -> None:
        results = retriever.search("anything", top_k=4)
        assert len(results) == 4
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_provenance_carries_vector_channel(self, retriever) -> None:
        results = retriever.search("anything", top_k=1)
        assert "vector" in results[0].provenance
        assert results[0].provenance["vector"] == results[0].score

    def test_top_k_larger_than_corpus(self, retriever, artifact) -> None:
        results = retriever.search("anything", top_k=500)
        assert len(results) == len(artifact["monikers"])

    def test_model_mismatch_rejected(self, snapshot) -> None:
        with pytest.raises(ValueError, match="model mismatch"):
            BaselineRetriever(snapshot, FixtureEmbedder(dims=128))

    def test_dims_validated_against_artifact(self, snapshot) -> None:
        class WrongDims(FixtureEmbedder):
            @property
            def model_id(self) -> str:
                return "fixture-384"   # lie about identity, dims still wrong

            @property
            def dimensions(self) -> int:
                return 128

        with pytest.raises(ValueError, match="dims mismatch"):
            BaselineRetriever(snapshot, WrongDims())


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

class TestDataset:
    def _write(self, tmp_path: Path, text: str) -> Path:
        p = tmp_path / "ds.yaml"
        p.write_text(text, encoding="utf-8")
        return p

    def test_loads_valid_dataset(self, tmp_path) -> None:
        p = self._write(tmp_path, """\
repo: booth
cases:
  - query: "which function saves the listing in db"
    category: behavioral
    expected: ["sutra python booth a.py save()."]
    must_include: ["sutra python booth a.py save()."]
    kind_filter: function
  - query: "Listing model"
    category: entity
    expected: ["sutra python booth m.py Listing#"]
""")
        cases = load_dataset(p)
        assert len(cases) == 2
        assert cases[0].repo == "booth"
        assert cases[0].kind_filter == "function"
        assert cases[0].must_include == ("sutra python booth a.py save().",)
        assert cases[1].must_include == ()
        assert cases[1].kind_filter is None

    def test_missing_repo_rejected(self, tmp_path) -> None:
        p = self._write(tmp_path, "cases:\n  - query: q\n")
        with pytest.raises(ValueError, match="missing the top-level 'repo'"):
            load_dataset(p)

    def test_empty_cases_rejected(self, tmp_path) -> None:
        p = self._write(tmp_path, "repo: r\ncases: []\n")
        with pytest.raises(ValueError, match="no cases"):
            load_dataset(p)

    def test_bad_category_rejected(self, tmp_path) -> None:
        p = self._write(tmp_path, """\
repo: r
cases:
  - query: q
    category: nonsense
    expected: ["m"]
""")
        with pytest.raises(ValueError, match="unknown category"):
            load_dataset(p)

    def test_empty_expected_rejected(self, tmp_path) -> None:
        p = self._write(tmp_path, """\
repo: r
cases:
  - query: q
    category: entity
    expected: []
""")
        with pytest.raises(ValueError, match="at least one moniker"):
            load_dataset(p)


# ---------------------------------------------------------------------------
# Metrics (pure functions, synthetic ranked lists)
# ---------------------------------------------------------------------------

def _results(*monikers: str) -> list[SearchResult]:
    return [
        SearchResult(moniker=m, score=1.0 - i * 0.1)
        for i, m in enumerate(monikers)
    ]


class TestMetrics:
    def test_first_hit_rank(self) -> None:
        rs = _results("a", "b", "c")
        assert first_hit_rank(["b"], rs) == 2
        assert first_hit_rank(["c", "a"], rs) == 1     # any-of: earliest wins
        assert first_hit_rank(["zzz"], rs) is None

    def test_recall_at_k(self) -> None:
        rs = _results("a", "b", "c")
        assert recall_at_k(["c"], rs, k=2) == 0.0
        assert recall_at_k(["c"], rs, k=3) == 1.0
        assert recall_at_k(["a"], rs, k=1) == 1.0

    def test_reciprocal_rank(self) -> None:
        rs = _results("a", "b", "c")
        assert reciprocal_rank(["b"], rs) == pytest.approx(0.5)
        assert reciprocal_rank(["zzz"], rs) == 0.0

    def test_must_include_coverage(self) -> None:
        rs = _results("a", "b", "c")
        assert must_include_coverage(["a", "c"], rs, k=3) == 1.0
        assert must_include_coverage(["a", "zzz"], rs, k=3) == 0.5
        assert must_include_coverage([], rs, k=3) is None


# ---------------------------------------------------------------------------
# Harness (run_eval + compare on the real retriever)
# ---------------------------------------------------------------------------

class TestHarness:
    def _cases_from_chunks(self, artifact, n: int = 3):
        """Real eval cases whose queries are exact chunk texts — the
        FixtureEmbedder guarantees rank-1 hits, so expected metrics are 1.0."""
        from sutra.core.retrieval.eval.dataset import EvalCase
        cases = []
        for chunk, moniker in list(zip(artifact["chunks"], artifact["monikers"]))[:n]:
            cases.append(EvalCase(
                query=chunk,
                category="exact-name",
                repo="sample_python_repo",
                expected=(moniker,),
            ))
        return cases

    def test_run_eval_perfect_on_exact_chunks(self, retriever, artifact) -> None:
        cases = self._cases_from_chunks(artifact)
        report = run_eval(retriever, cases)
        agg = report.aggregate()
        assert agg["n"] == len(cases)
        assert agg["recall@1"] == 1.0
        assert agg["recall@10"] == 1.0
        assert agg["mrr"] == 1.0

    def test_run_eval_with_repo_mapping(self, retriever, artifact) -> None:
        cases = self._cases_from_chunks(artifact, n=2)
        report = run_eval({"sample_python_repo": retriever}, cases)
        assert report.aggregate()["recall@1"] == 1.0

    def test_run_eval_missing_repo_raises(self, retriever, artifact) -> None:
        cases = self._cases_from_chunks(artifact, n=1)
        with pytest.raises(KeyError, match="No retriever supplied"):
            run_eval({"other_repo": retriever}, cases)

    def test_report_dict_and_summary(self, retriever, artifact) -> None:
        report = run_eval(retriever, self._cases_from_chunks(artifact))
        d = report.to_dict()
        assert set(d) == {"ks", "overall", "by_category", "per_query"}
        assert d["per_query"][0]["first_hit_rank"] == 1
        assert "OVERALL" in report.summary()
        json.dumps(d)   # must be JSON-serializable

    def test_compare_identical_reports_zero_delta(self, retriever, artifact) -> None:
        cases = self._cases_from_chunks(artifact)
        a = run_eval(retriever, cases)
        b = run_eval(retriever, cases)
        cmp = compare(a, b)
        assert all(v == 0.0 for v in cmp.overall_delta.values())
        assert cmp.improved == [] and cmp.regressed == []

    def test_compare_detects_regression(self, retriever, artifact) -> None:
        """Degrade the candidate by querying with garbage — every exact-chunk
        case regresses and compare() must name each one."""
        cases = self._cases_from_chunks(artifact)
        baseline = run_eval(retriever, cases)

        class Degraded:
            """Real retriever, deliberately wrong query — not a mock of
            the contract, an adversarial use of it."""
            def __init__(self, inner) -> None:
                self._inner = inner

            def search(self, query: str, top_k: int = 10):
                results = self._inner.search(query, top_k=top_k + 5)
                return results[5:]   # drop the true top-5


        candidate = run_eval(Degraded(retriever), cases)
        cmp = compare(baseline, candidate)
        assert len(cmp.regressed) == len(cases)
        assert cmp.overall_delta["recall@1"] == -1.0
        assert "regressed" in cmp.summary()

    def test_compare_mismatched_cases_raise(self, retriever, artifact) -> None:
        a = run_eval(retriever, self._cases_from_chunks(artifact, n=2))
        b = run_eval(retriever, self._cases_from_chunks(artifact, n=3))
        with pytest.raises(ValueError, match="same cases"):
            compare(a, b)


# ---------------------------------------------------------------------------
# Real-eval baseline gate (needs built artifacts — skip cleanly otherwise)
# ---------------------------------------------------------------------------

def _artifacts_ready() -> bool:
    if not _DATASETS_DIR.exists() or not _ARTIFACTS_DIR.exists():
        return False
    for ds in _DATASETS_DIR.glob("*.yaml"):
        import yaml
        repo = (yaml.safe_load(ds.read_text()) or {}).get("repo")
        if repo and not (_ARTIFACTS_DIR / repo / "graph.json").exists():
            return False
    return True


@pytest.mark.skipif(
    not _artifacts_ready(),
    reason="Eval artifacts not built — run: python scripts/build_eval_artifacts.py",
)
class TestBaselineGate:
    def test_dataset_monikers_exist_in_artifacts(self) -> None:
        """Every expected / must_include moniker must be a real symbol in its
        repo's artifact — catches dataset typos and reference-repo drift."""
        cases = load_datasets(_DATASETS_DIR)
        loader = ArtifactLoader()
        symbols_by_repo = {
            repo: set(loader.load(_ARTIFACTS_DIR / repo).symbols)
            for repo in sorted({c.repo for c in cases})
        }
        missing: list[str] = []
        for case in cases:
            for m in (*case.expected, *case.must_include):
                if m not in symbols_by_repo[case.repo]:
                    missing.append(f"  [{case.repo}] {case.query!r}: {m}")
        assert not missing, (
            "Dataset monikers not found in artifacts (typo or repo drift):\n"
            + "\n".join(missing)
        )

    @pytest.fixture(scope="class")
    def report(self):
        from sutra.core.embedder.local import LocalEmbedder

        cases = load_datasets(_DATASETS_DIR)
        assert len(cases) >= 30, (
            f"P12 acceptance requires ≥30 queries; datasets have {len(cases)}."
        )
        embedder = LocalEmbedder()   # pinned all-MiniLM-L6-v2, 384 dims
        loader = ArtifactLoader()
        retrievers = {}
        for repo in sorted({c.repo for c in cases}):
            snap = loader.load(_ARTIFACTS_DIR / repo)
            retrievers[repo] = BaselineRetriever(snap, embedder)
        return run_eval(retrievers, cases)

    def test_baseline_within_tolerance_of_snapshot(self, report) -> None:
        current = report.to_dict()

        if os.environ.get("UPDATE_EVAL_BASELINE") == "1":
            _BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _BASELINE_PATH.write_text(
                json.dumps(current, indent=2), encoding="utf-8"
            )
            pytest.skip("Baseline snapshot updated — re-run without the flag.")

        if not _BASELINE_PATH.exists():
            pytest.fail(
                "No baseline snapshot. Generate one with:\n"
                "  UPDATE_EVAL_BASELINE=1 pytest tests/test_eval_harness.py"
            )

        saved = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))

        # Build rank lookup to name regressed queries on drift.
        saved_ranks = {
            (q["repo"], q["query"]): q["first_hit_rank"]
            for q in saved["per_query"]
        }
        drifted: list[str] = []
        for metric, expected in saved["overall"].items():
            if metric == "n":
                continue
            actual = current["overall"][metric]
            if abs(actual - expected) > _TOLERANCE:
                regressed = [
                    f"    {q['query']!r} (rank {saved_ranks.get((q['repo'], q['query']))} "
                    f"→ {q['first_hit_rank']})"
                    for q in current["per_query"]
                    if (q["first_hit_rank"] or 10**9)
                    > (saved_ranks.get((q["repo"], q["query"])) or 10**9)
                ]
                drifted.append(
                    f"  {metric}: snapshot={expected:.3f} now={actual:.3f}\n"
                    + "\n".join(regressed)
                )
        assert not drifted, (
            "Baseline metrics drifted beyond ±5%:\n" + "\n".join(drifted)
        )
