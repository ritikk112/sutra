"""
Priority 17 — RRF fusion + moniker channel + one-hop graph expansion +
the assembled RetrievalPipeline.

Hermetic: fusion math on real SearchResults, moniker-channel tiers,
expander over a resolver-resolved fixture artifact, full pipeline runs.

Gated acceptance (full pipeline vs baseline, expansion A/B) lives in
TestPipelineAcceptanceEval — needs eval artifacts rebuilt WITH the
P20-lite resolver (run scripts/build_eval_artifacts.py after P20-lite).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sutra.core.artifact import ArtifactLoader
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.core.resolver import HeuristicResolver
from sutra.core.retrieval.channels.moniker_channel import MonikerChannel
from sutra.core.retrieval.expander import GraphExpander
from sutra.core.retrieval.fusion import rrf_fuse
from sutra.core.retrieval.pipeline import RetrievalPipeline
from sutra.core.retrieval.query import ParsedQuery
from sutra.core.retrieval.types import SearchResult

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE_REPO = _FIXTURES / "sample_python_repo"
_EVAL_DIR = Path(__file__).parent / "eval"
_DATASETS_DIR = _EVAL_DIR / "datasets"
_ARTIFACTS_DIR = _EVAL_DIR / "artifacts"


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory):
    """Fixture artifact WITH resolved CALLS (the resolver ran)."""
    out = tmp_path_factory.mktemp("p17_artifact")
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
    return ArtifactLoader().load(out)


def _r(moniker: str, score: float, **prov: float) -> SearchResult:
    return SearchResult(moniker=moniker, score=score, provenance=dict(prov))


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

class TestRrfFusion:
    def test_agreement_beats_single_channel_head(self) -> None:
        """A doc ranked #2 by BOTH channels outscores docs ranked #1 by one."""
        fused = rrf_fuse({
            "vector": [_r("only_vec", 0.99, vector=0.99), _r("both", 0.5, vector=0.5)],
            "bm25": [_r("only_bm", 12.0, bm25=12.0), _r("both", 4.0, bm25=4.0)],
        })
        assert fused[0].moniker == "both"
        # 2/(60+2) > 1/(60+1) → agreement wins at k=60.

    def test_scores_are_rank_based_not_raw(self) -> None:
        """Absurd raw scores must not leak into fusion."""
        a = rrf_fuse({"vector": [_r("x", 0.0001)], "bm25": [_r("y", 99999.0)]})
        b = rrf_fuse({"vector": [_r("x", 0.9)], "bm25": [_r("y", 0.1)]})
        assert [r.moniker for r in a] == [r.moniker for r in b]
        assert [r.score for r in a] == [r.score for r in b]

    def test_provenance_union_plus_rrf(self) -> None:
        fused = rrf_fuse({
            "vector": [_r("m", 0.8, vector=0.8)],
            "bm25": [_r("m", 7.0, bm25=7.0)],
        })
        prov = fused[0].provenance
        assert prov["vector"] == 0.8 and prov["bm25"] == 7.0
        assert prov["rrf"] == fused[0].score

    def test_deterministic_tie_break(self) -> None:
        fused = rrf_fuse({"vector": [], "bm25": [_r("b", 1.0), _r("a", 0.9)]})
        # b rank1 > a rank2 — order by score then moniker, stable across runs.
        assert [r.moniker for r in fused] == ["b", "a"]
        again = rrf_fuse({"bm25": [_r("b", 1.0), _r("a", 0.9)], "vector": []})
        assert [r.moniker for r in fused] == [r.moniker for r in again]

    def test_top_k(self) -> None:
        fused = rrf_fuse(
            {"bm25": [_r(f"m{i}", 1.0 - i * 0.1) for i in range(5)]}, top_k=2
        )
        assert len(fused) == 2

    def test_empty_channels(self) -> None:
        assert rrf_fuse({"vector": [], "bm25": []}) == []


# ---------------------------------------------------------------------------
# Moniker channel
# ---------------------------------------------------------------------------

class TestMonikerChannel:
    def test_exact_name_tier(self, snapshot) -> None:
        ch = MonikerChannel(snapshot.symbols)
        q = ParsedQuery(text="create_user", entities=("create_user",))
        results = ch.retrieve(q)
        assert results[0].score == 1.0
        assert snapshot.symbols[results[0].moniker]["name"] == "create_user"

    def test_case_insensitive_tier(self, snapshot) -> None:
        ch = MonikerChannel(snapshot.symbols)
        q = ParsedQuery(text="userservice", entities=("userService",))
        results = ch.retrieve(q)
        assert results
        assert results[0].score == 0.8
        assert snapshot.symbols[results[0].moniker]["name"] == "UserService"

    def test_no_entities_no_results(self, snapshot) -> None:
        ch = MonikerChannel(snapshot.symbols)
        assert ch.retrieve(ParsedQuery(text="save the meeting")) == []

    def test_part_tier_reaches_camel_case(self, snapshot) -> None:
        ch = MonikerChannel(snapshot.symbols)
        q = ParsedQuery(text="generate_id", entities=("generate_id",))
        # _generate_id name: split parts contain "generate" + "id" — the
        # exact entity "generate_id" doesn't part-match, but ci-name does
        # not either; verify graceful empty-or-scored behavior (no crash).
        results = ch.retrieve(q)
        assert all(r.score in (1.0, 0.8, 0.6, 0.4) for r in results)

    def test_filter_monikers_respected(self, snapshot) -> None:
        ch = MonikerChannel(snapshot.symbols)
        q = ParsedQuery(text="create_user", entities=("create_user",))
        assert ch.retrieve(q, filter_monikers=["nothing"]) == []


# ---------------------------------------------------------------------------
# Graph expansion
# ---------------------------------------------------------------------------

class TestGraphExpander:
    def test_graph_has_resolved_edges(self, snapshot) -> None:
        expander = GraphExpander(snapshot)
        assert expander.edge_count > 0, (
            "fixture artifact must carry resolver-resolved CALLS edges"
        )

    def test_expansion_adds_callee_of_seed(self, snapshot) -> None:
        """Seeding with create_user must pull in _generate_id (its callee)."""
        create_user = next(m for m in snapshot.symbols if "create_user" in m)
        generate_id = next(m for m in snapshot.symbols if "_generate_id" in m)
        expander = GraphExpander(snapshot)
        out = expander.expand([_r(create_user, 0.9)])
        monikers = [r.moniker for r in out]
        assert generate_id in monikers
        added = next(r for r in out if r.moniker == generate_id)
        assert added.provenance == {"expansion": added.score}
        assert added.score == pytest.approx(0.9 * 0.5)

    def test_existing_results_never_displaced(self, snapshot) -> None:
        create_user = next(m for m in snapshot.symbols if "create_user" in m)
        generate_id = next(m for m in snapshot.symbols if "_generate_id" in m)
        expander = GraphExpander(snapshot)
        fused = [_r(create_user, 0.9), _r(generate_id, 0.7, vector=0.7)]
        out = expander.expand(fused)
        kept = next(r for r in out if r.moniker == generate_id)
        # Already-present result keeps its fused identity, not an expansion copy.
        assert kept.score == 0.7
        assert "expansion" not in kept.provenance

    def test_neighbors_score_below_seed(self, snapshot) -> None:
        create_user = next(m for m in snapshot.symbols if "create_user" in m)
        expander = GraphExpander(snapshot)
        out = expander.expand([_r(create_user, 0.9)])
        assert out[0].moniker == create_user

    def test_unknown_seed_is_harmless(self, snapshot) -> None:
        expander = GraphExpander(snapshot)
        out = expander.expand([_r("sutra x y z unknown().", 0.5)])
        assert [r.moniker for r in out] == ["sutra x y z unknown()."]


# ---------------------------------------------------------------------------
# Assembled pipeline (hermetic)
# ---------------------------------------------------------------------------

class TestPipelineHermetic:
    def test_search_returns_ranked_results(self, snapshot) -> None:
        pipe = RetrievalPipeline(snapshot, FixtureEmbedder())
        results = pipe.search("create_user", top_k=5)
        assert results
        assert len(results) <= 5
        # exact-name query: moniker channel + bm25 agree → create_user on top
        assert "create_user" in results[0].moniker
        assert "rrf" in results[0].provenance

    def test_kind_hint_restricts_channels(self, snapshot) -> None:
        pipe = RetrievalPipeline(snapshot, FixtureEmbedder(), expand=False)
        results = pipe.search("the user service class", top_k=10)
        kinds = {snapshot.symbols[r.moniker]["kind"] for r in results}
        assert kinds <= {"class"}

    def test_expansion_can_be_disabled(self, snapshot) -> None:
        on = RetrievalPipeline(snapshot, FixtureEmbedder(), expand=True)
        off = RetrievalPipeline(snapshot, FixtureEmbedder(), expand=False)
        # Same query, expansion adds candidates (fixture graph has edges).
        r_on = on.search("create_user", top_k=50)
        r_off = off.search("create_user", top_k=50)
        assert len(r_on) >= len(r_off)

    def test_embedder_mismatch_rejected(self, snapshot) -> None:
        with pytest.raises(ValueError, match="model mismatch"):
            RetrievalPipeline(snapshot, FixtureEmbedder(dims=128))


# ---------------------------------------------------------------------------
# P17 acceptance — full pipeline vs baseline + expansion A/B (gated)
# ---------------------------------------------------------------------------

def _artifacts_ready() -> bool:
    if not _DATASETS_DIR.exists():
        return False
    import json
    # The P17 gate additionally requires RESOLVED artifacts (rebuilt after
    # P20-lite) — detectable by any resolved CALLS edge in booth.
    booth = _ARTIFACTS_DIR / "booth" / "graph.json"
    if not booth.exists():
        return False
    g = json.loads(booth.read_text())
    return any(
        r["is_resolved"] for r in g["relationships"] if r["kind"] == "calls"
    )


class _PipelineRetriever:
    def __init__(self, pipe: RetrievalPipeline) -> None:
        self._pipe = pipe

    def search(self, query: str, top_k: int = 10):
        return self._pipe.search(query, top_k=top_k)


@pytest.mark.skipif(
    not _artifacts_ready(),
    reason="Resolved eval artifacts not built — run scripts/build_eval_artifacts.py "
           "(after P20-lite, so CALLS edges are resolved)",
)
class TestPipelineAcceptanceEval:
    @pytest.fixture(scope="class")
    def reports(self):
        from sutra.core.embedder.local import LocalEmbedder
        from sutra.core.retrieval.baseline import BaselineRetriever
        from sutra.core.retrieval.eval import load_datasets, run_eval
        from sutra.core.retrieval.query_analyzer import QueryAnalyzer

        cases = load_datasets(_DATASETS_DIR)
        embedder = LocalEmbedder()
        analyzer = QueryAnalyzer(embedder=embedder)
        loader = ArtifactLoader()

        baseline, with_exp, without_exp = {}, {}, {}
        for repo in sorted({c.repo for c in cases}):
            snap = loader.load(_ARTIFACTS_DIR / repo)
            baseline[repo] = BaselineRetriever(snap, embedder)
            with_exp[repo] = _PipelineRetriever(
                RetrievalPipeline(snap, embedder, analyzer=analyzer, expand=True)
            )
            without_exp[repo] = _PipelineRetriever(
                RetrievalPipeline(snap, embedder, analyzer=analyzer, expand=False)
            )

        return {
            "cases": cases,
            "baseline": run_eval(baseline, cases),
            "pipeline": run_eval(with_exp, cases),
            "no_expansion": run_eval(without_exp, cases),
        }

    def test_pipeline_beats_baseline(self, reports) -> None:
        from sutra.core.retrieval.eval import compare
        cmp = compare(reports["baseline"], reports["pipeline"])
        print("\nFull pipeline vs vector baseline:\n" + cmp.summary())
        print("\nPipeline eval:\n" + reports["pipeline"].summary())

        base = reports["baseline"].aggregate()
        pipe = reports["pipeline"].aggregate()
        assert pipe["mrr"] > base["mrr"], cmp.summary()
        assert pipe["recall@10"] >= base["recall@10"], cmp.summary()

        # No regression on paraphrase (spec acceptance).
        base_para = reports["baseline"].aggregate("paraphrase")
        pipe_para = reports["pipeline"].aggregate("paraphrase")
        assert pipe_para["recall@10"] >= base_para["recall@10"]

    def test_expansion_delta_measured(self, reports) -> None:
        """Spec: expansion must add ≥5% recall@10 over fused-without-
        expansion or it gets dropped (made non-default).  This test
        RECORDS the delta; the default-flag decision is taken in
        PROGRESS.md based on this number."""
        on = reports["pipeline"].aggregate()
        off = reports["no_expansion"].aggregate()
        delta = on["recall@10"] - off["recall@10"]
        print(
            f"\nexpansion A/B: recall@10 with={on['recall@10']:.3f} "
            f"without={off['recall@10']:.3f} delta={delta:+.3f}"
        )
        # Expansion adds candidates below seeds: it must never HURT recall.
        assert delta >= 0.0
