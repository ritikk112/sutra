"""
Priority 15 — BM25 channel + Channel ABC (+ the VectorStore seam).

Hermetic layer: the checked-in sample_python_repo artifact (FixtureEmbedder)
exercises tokenizer, InMemoryVectorStore, VectorChannel, and Bm25Channel on
real code paths.

Gated layer (needs built eval artifacts): BM25-only retrieval over the P12
datasets — the P15 acceptance run.  Expected shape: exact-name strong,
behavioral weaker than the vector baseline.  Numbers are documented in
PROGRESS.md; the assertions here are the structural floors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sutra.core.artifact import ArtifactLoader
from sutra.core.embedder.chunk_builder import build_chunks
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.core.retrieval.channels import Bm25Channel, Channel, VectorChannel
from sutra.core.retrieval.query import ParsedQuery
from sutra.core.retrieval.text import split_identifier, tokenize
from sutra.core.vector_store import InMemoryVectorStore
from sutra.core.retrieval.eval import load_datasets, run_eval

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE_REPO = _FIXTURES / "sample_python_repo"
_EVAL_DIR = Path(__file__).parent / "eval"
_DATASETS_DIR = _EVAL_DIR / "datasets"
_ARTIFACTS_DIR = _EVAL_DIR / "artifacts"


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("p15_artifact")
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
    snapshot = ArtifactLoader().load(out)
    return {
        "snapshot": snapshot,
        "embedder": embedder,
        "chunks": chunks,
        "monikers": monikers,
    }


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_split_camel_case(self) -> None:
        assert split_identifier("getUserIdFromToken") == [
            "get", "user", "id", "from", "token",
        ]

    def test_split_snake_case(self) -> None:
        assert split_identifier("create_user_profile") == [
            "create", "user", "profile",
        ]

    def test_split_acronym(self) -> None:
        assert split_identifier("HTTPSConnection") == ["https", "connection"]

    def test_split_pascal_with_digits(self) -> None:
        assert split_identifier("BulkImportV2") == ["bulk", "import", "v", "2"]

    def test_tokenize_keeps_whole_identifier_and_parts(self) -> None:
        tokens = tokenize("getUserIdFromToken")
        assert "getuseridfromtoken" in tokens     # exact-name queries match
        assert "token" in tokens                  # word queries match

    def test_tokenize_plain_words_not_doubled(self) -> None:
        assert tokenize("save the meeting") == ["save", "the", "meeting"]

    def test_tokenize_drops_punctuation(self) -> None:
        tokens = tokenize("def create_user(self, name: str) -> User:")
        assert "create_user" in tokens and "user" in tokens
        assert all("(" not in t and ":" not in t for t in tokens)


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------

class TestInMemoryVectorStore:
    def test_exact_vector_is_top_hit_with_similarity_one(self, artifact) -> None:
        snap = artifact["snapshot"]
        store = InMemoryVectorStore(snap)
        row = snap.row_of[snap.moniker_order[0]]
        hits = store.search(snap.vectors[row], k=3)
        assert hits[0][0] == snap.moniker_order[0]
        assert hits[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_filter_monikers_restricts_before_topk(self, artifact) -> None:
        snap = artifact["snapshot"]
        store = InMemoryVectorStore(snap)
        target = snap.moniker_order[-1]
        hits = store.search(snap.vectors[0], k=5, filter_monikers=[target])
        assert [m for m, _ in hits] == [target]

    def test_filter_with_unknown_monikers_returns_empty(self, artifact) -> None:
        store = InMemoryVectorStore(artifact["snapshot"])
        assert store.search(
            artifact["snapshot"].vectors[0], k=5, filter_monikers=["nope"]
        ) == []

    def test_dims_mismatch_raises(self, artifact) -> None:
        store = InMemoryVectorStore(artifact["snapshot"])
        with pytest.raises(ValueError, match="shape"):
            store.search(np.ones(7, dtype=np.float32), k=3)


# ---------------------------------------------------------------------------
# VectorChannel
# ---------------------------------------------------------------------------

class TestVectorChannel:
    def test_is_a_channel_named_vector(self, artifact) -> None:
        ch = VectorChannel(InMemoryVectorStore(artifact["snapshot"]))
        assert isinstance(ch, Channel)
        assert ch.name == "vector"

    def test_requires_embedding(self, artifact) -> None:
        ch = VectorChannel(InMemoryVectorStore(artifact["snapshot"]))
        with pytest.raises(ValueError, match="embedding"):
            ch.retrieve(ParsedQuery(text="no embedding here"))

    def test_exact_chunk_embedding_ranks_first(self, artifact) -> None:
        ch = VectorChannel(InMemoryVectorStore(artifact["snapshot"]))
        chunk, moniker = artifact["chunks"][0], artifact["monikers"][0]
        qvec = artifact["embedder"].embed([chunk])[0]
        results = ch.retrieve(ParsedQuery(text=chunk, embedding=qvec), top_k=3)
        assert results[0].moniker == moniker
        assert results[0].provenance == {"vector": results[0].score}


# ---------------------------------------------------------------------------
# Bm25Channel
# ---------------------------------------------------------------------------

class TestBm25Channel:
    def test_is_a_channel_named_bm25(self, artifact) -> None:
        ch = Bm25Channel(artifact["snapshot"])
        assert isinstance(ch, Channel)
        assert ch.name == "bm25"

    def test_exact_name_query_ranks_symbol_first(self, artifact) -> None:
        ch = Bm25Channel(artifact["snapshot"])
        results = ch.retrieve(ParsedQuery(text="create_user"), top_k=5)
        assert results, "exact-name query must produce candidates"
        assert "create_user" in results[0].moniker

    def test_identifier_parts_match(self, artifact) -> None:
        """Querying with words must reach snake_case identifiers."""
        ch = Bm25Channel(artifact["snapshot"])
        results = ch.retrieve(ParsedQuery(text="generate id"), top_k=5)
        assert any("_generate_id" in r.moniker for r in results)

    def test_modules_are_in_the_corpus(self, artifact) -> None:
        """Load-bearing: modules are NOT embedded, so only the lexical
        channel can ever retrieve them (the api-client known-failure)."""
        ch = Bm25Channel(artifact["snapshot"])
        results = ch.retrieve(ParsedQuery(text="services user module"), top_k=10)
        kinds = {
            artifact["snapshot"].symbols[r.moniker]["kind"] for r in results
        }
        assert "module" in kinds

    def test_no_match_returns_empty_not_noise(self, artifact) -> None:
        ch = Bm25Channel(artifact["snapshot"])
        assert ch.retrieve(ParsedQuery(text="zzzqqqxxx"), top_k=5) == []

    def test_scores_sorted_and_provenance_keyed(self, artifact) -> None:
        ch = Bm25Channel(artifact["snapshot"])
        results = ch.retrieve(ParsedQuery(text="user service create"), top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert all("bm25" in r.provenance for r in results)

    def test_filter_monikers(self, artifact) -> None:
        snap = artifact["snapshot"]
        ch = Bm25Channel(snap)
        unfiltered = ch.retrieve(ParsedQuery(text="create user"), top_k=10)
        assert len(unfiltered) > 1
        keep = unfiltered[1].moniker
        filtered = ch.retrieve(
            ParsedQuery(text="create user"), top_k=10, filter_monikers=[keep]
        )
        assert [r.moniker for r in filtered] == [keep]


# ---------------------------------------------------------------------------
# P15 acceptance — BM25-only over the real datasets (gated)
# ---------------------------------------------------------------------------

def _artifacts_ready() -> bool:
    if not _DATASETS_DIR.exists() or not _ARTIFACTS_DIR.exists():
        return False
    import yaml
    for ds in _DATASETS_DIR.glob("*.yaml"):
        repo = (yaml.safe_load(ds.read_text()) or {}).get("repo")
        if repo and not (_ARTIFACTS_DIR / repo / "graph.json").exists():
            return False
    return True


class _Bm25OnlyRetriever:
    """Real Bm25Channel behind the harness's Searcher protocol."""

    def __init__(self, channel: Bm25Channel) -> None:
        self._channel = channel

    def search(self, query: str, top_k: int = 10):
        return self._channel.retrieve(ParsedQuery(text=query), top_k=top_k)


@pytest.mark.skipif(
    not _artifacts_ready(),
    reason="Eval artifacts not built — run: python scripts/build_eval_artifacts.py",
)
class TestBm25AcceptanceEval:
    @pytest.fixture(scope="class")
    def report(self):
        cases = load_datasets(_DATASETS_DIR)
        loader = ArtifactLoader()
        retrievers = {
            repo: _Bm25OnlyRetriever(Bm25Channel(loader.load(_ARTIFACTS_DIR / repo)))
            for repo in sorted({c.repo for c in cases})
        }
        return run_eval(retrievers, cases)

    def test_exact_name_strong(self, report) -> None:
        """BM25's home turf: exact-name queries must be near-perfect."""
        agg = report.aggregate("exact-name")
        assert agg["recall@5"] >= 0.75, report.summary()

    def test_module_lexically_reachable_when_named(self, report) -> None:
        """Modules are NOT embedded, so the lexical channel is the only path
        to them.  A query naming the module must reach it."""
        cases = load_datasets(_DATASETS_DIR)
        loader = ArtifactLoader()
        ch = Bm25Channel(loader.load(_ARTIFACTS_DIR / "outreach-web-ui"))
        results = ch.retrieve(ParsedQuery(text="api client"), top_k=10)
        assert any(
            r.moniker.endswith("src/lib/api-client/") for r in results
        ), [r.moniker for r in results[:5]]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Structural gap, documented at P15: the token-attaching code is an "
            "anonymous axios interceptor — its only symbol is the api-client "
            "module, which carries no query-overlapping text (no docstring, no "
            "signature).  NO current channel can win this query.  Candidate "
            "future fix: synthesize module text (exports + leading comments) "
            "into the BM25 corpus / an embedded module chunk.  If this test "
            "starts passing, a later priority closed the gap — promote it."
        ),
    )
    def test_known_failure_token_attach_reachable(self, report) -> None:
        q = next(
            qe for qe in report.per_query
            if "attaches token" in qe.case.query
        )
        assert q.first_hit_rank is not None and q.first_hit_rank <= 10

    def test_print_deltas_for_progress_log(self, report) -> None:
        """Not an assertion — emits the acceptance numbers (pytest -s)."""
        print("\nBM25-only eval:\n" + report.summary())
