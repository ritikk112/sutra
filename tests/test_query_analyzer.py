"""
Priority 16 — Query analyzer + kind filter.

Hermetic: lexicon-driven parsing (kind hints, verbs, entities), embedding
computed exactly once, kind filter pre/post variants over the checked-in
fixture artifact.

Gated acceptance (needs built eval artifacts): vector retrieval with
analyzer-driven kind pre-restriction vs the plain vector baseline over the
P12 datasets — behavioral queries must stop returning request models.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sutra.core.artifact import ArtifactLoader
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.core.retrieval.channels import VectorChannel
from sutra.core.retrieval.eval import compare, load_datasets, run_eval
from sutra.core.retrieval.kind_filter import allowed_monikers, apply_kind_filter
from sutra.core.retrieval.query import ParsedQuery
from sutra.core.retrieval.query_analyzer import QueryAnalyzer
from sutra.core.retrieval.types import SearchResult
from sutra.core.vector_store import InMemoryVectorStore

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE_REPO = _FIXTURES / "sample_python_repo"
_EVAL_DIR = Path(__file__).parent / "eval"
_DATASETS_DIR = _EVAL_DIR / "datasets"
_ARTIFACTS_DIR = _EVAL_DIR / "artifacts"


@pytest.fixture(scope="module")
def analyzer():
    return QueryAnalyzer()   # lexical-only — no embedder


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("p16_artifact")
    Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
    ).index(
        root=_FIXTURE_REPO,
        repo_url="https://github.com/test/sample_python_repo",
        output_dir=out,
    )
    return ArtifactLoader().load(out)


# ---------------------------------------------------------------------------
# Kind hints
# ---------------------------------------------------------------------------

class TestKindHints:
    @pytest.mark.parametrize("query,expected", [
        # explicit callable nouns
        ("which function saves the meeting in the database", {"function", "method"}),
        ("hook that polls the progress of a bulk import job", {"function", "method"}),
        ("middleware that recovers from panics", {"function", "method"}),
        # explicit class nouns
        # "model"/"models" were removed from the class nouns (KIND_FILTER_AB.md):
        # in ML-adjacent corpora they name data, not a class definition.
        ("the Meeting database model", None),
        ("request model for creating a visitor by a partner", {"function", "method"}),
        ("props for the edit deal modal", {"class"}),
        ("the router group type that manages route prefixes", {"class"}),
        ("database connection singleton", {"class"}),
        # module nouns
        # "module"/"modules" were likewise removed (they describe WHERE, not kind).
        ("the api client module", None),
        # verb-derived fallback (no explicit kind noun)
        ("save an uploaded multipart form file to disk", {"function", "method"}),
        ("start the http server and listen for requests", {"function", "method"}),
        ("debounce a changing value", {"function", "method"}),
    ])
    def test_hints(self, analyzer, query, expected) -> None:
        want = frozenset(expected) if expected is not None else None
        assert analyzer.parse(query, embed=False).kind_hint == want

    def test_no_signal_means_no_hint(self, analyzer) -> None:
        # Pure entity query: no kind noun, no behavioral verb.
        assert analyzer.parse("lazyRetry", embed=False).kind_hint is None

    def test_ambiguous_nouns_mean_no_hint(self, analyzer) -> None:
        # "function" (callable) + "model" (class) → conflicting groups.
        parsed = analyzer.parse(
            "the function on the model class", embed=False
        )
        assert parsed.kind_hint is None

    def test_explicit_noun_beats_verbs(self, analyzer) -> None:
        # "creating" is a behavioral verb, but "dataclass" is explicit → class.
        # (previously used "model", which is no longer a kind noun)
        parsed = analyzer.parse(
            "request dataclass for creating a visitor", embed=False
        )
        assert parsed.kind_hint == frozenset({"class"})

    @pytest.mark.parametrize("query", [
        # Found on pehchaan (accuracy battery): "image type" misread as a
        # class request filtered out the expected method entirely.
        "validate that an uploaded file is an allowed image type",
        "parse the content type header",
        "detect the mime type of an upload",
        "convert the file type before saving",
    ])
    def test_data_phrase_type_is_not_a_kind_noun(self, analyzer, query) -> None:
        """'type' preceded by content/image/file/mime/media/data names a
        DATA property, not a symbol kind — the verb-derived callable hint
        (or no hint) must win, never a class filter."""
        parsed = analyzer.parse(query, embed=False)
        assert parsed.kind_hint != frozenset({"class"})

    def test_bare_type_still_hints_class(self, analyzer) -> None:
        # The gin entity query must keep working: bare "type" = kind noun.
        parsed = analyzer.parse(
            "the router group type that manages route prefixes", embed=False
        )
        assert parsed.kind_hint == frozenset({"class"})


# ---------------------------------------------------------------------------
# Verbs + entities
# ---------------------------------------------------------------------------

class TestVerbsAndEntities:
    def test_verb_lemmatization(self, analyzer) -> None:
        parsed = analyzer.parse(
            "which function saves the meeting after uploading", embed=False
        )
        assert "save" in parsed.verbs
        assert "upload" in parsed.verbs

    def test_retries_lemma(self, analyzer) -> None:
        parsed = analyzer.parse("function that retries the transaction", embed=False)
        assert "retry" in parsed.verbs

    def test_entities_identifier_morphology(self, analyzer) -> None:
        parsed = analyzer.parse("PostgresDALWrapper find_one", embed=False)
        assert parsed.entities == ("PostgresDALWrapper", "find_one")

    def test_entities_camel_case(self, analyzer) -> None:
        parsed = analyzer.parse("where is lazyRetry used", embed=False)
        assert "lazyRetry" in parsed.entities

    def test_plain_words_are_not_entities(self, analyzer) -> None:
        parsed = analyzer.parse("save the meeting in the database", embed=False)
        assert parsed.entities == ()


# ---------------------------------------------------------------------------
# Embedding behavior
# ---------------------------------------------------------------------------

class TestEmbedding:
    def test_embeds_once_when_asked(self) -> None:
        class CountingEmbedder(FixtureEmbedder):
            calls = 0

            def embed(self, chunks):
                CountingEmbedder.calls += 1
                return super().embed(chunks)

        analyzer = QueryAnalyzer(embedder=CountingEmbedder())
        parsed = analyzer.parse("save the meeting")
        assert parsed.embedding is not None
        assert parsed.embedding.shape == (384,)
        assert CountingEmbedder.calls == 1

    def test_embed_false_skips_embedding(self) -> None:
        analyzer = QueryAnalyzer(embedder=FixtureEmbedder())
        assert analyzer.parse("save the meeting", embed=False).embedding is None

    def test_no_embedder_is_fine(self, analyzer) -> None:
        assert analyzer.parse("save the meeting").embedding is None


# ---------------------------------------------------------------------------
# Kind filter
# ---------------------------------------------------------------------------

class TestKindFilter:
    def test_allowed_monikers_restricts_to_hint(self, snapshot) -> None:
        q = ParsedQuery(text="q", kind_hint=frozenset({"class"}))
        allowed = allowed_monikers(q, snapshot.symbols)
        assert allowed
        assert all(snapshot.symbols[m]["kind"] == "class" for m in allowed)

    def test_allowed_monikers_none_without_hint(self, snapshot) -> None:
        assert allowed_monikers(ParsedQuery(text="q"), snapshot.symbols) is None

    def test_post_filter_drops_contradicting_kinds(self, snapshot) -> None:
        results = [
            SearchResult(moniker=m, score=1.0) for m in snapshot.symbols
        ]
        q = ParsedQuery(text="q", kind_hint=frozenset({"function", "method"}))
        kept = apply_kind_filter(results, q, snapshot.symbols)
        assert kept
        assert all(
            snapshot.symbols[r.moniker]["kind"] in ("function", "method")
            for r in kept
        )

    def test_post_filter_keeps_unknown_monikers(self, snapshot) -> None:
        results = [SearchResult(moniker="sutra unknown x y z().", score=1.0)]
        q = ParsedQuery(text="q", kind_hint=frozenset({"class"}))
        assert apply_kind_filter(results, q, snapshot.symbols) == results

    def test_pre_restriction_through_vector_channel(self, snapshot) -> None:
        """End-to-end: hint → allowed set → VectorChannel filter_monikers."""
        embedder = FixtureEmbedder()
        channel = VectorChannel(InMemoryVectorStore(snapshot))
        q = ParsedQuery(
            text="any",
            embedding=embedder.embed(["any"])[0],
            kind_hint=frozenset({"class"}),
        )
        results = channel.retrieve(
            q, top_k=10, filter_monikers=allowed_monikers(q, snapshot.symbols)
        )
        assert results
        assert all(
            snapshot.symbols[r.moniker]["kind"] == "class" for r in results
        )


# ---------------------------------------------------------------------------
# P16 acceptance — kind-filtered vector vs plain vector baseline (gated)
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


class _KindFilteredVectorRetriever:
    """Vector channel with analyzer-driven kind pre-restriction — the P16
    acceptance configuration (baseline + kind filter, nothing else)."""

    def __init__(self, snapshot, analyzer: QueryAnalyzer) -> None:
        self._snapshot = snapshot
        self._analyzer = analyzer
        self._channel = VectorChannel(InMemoryVectorStore(snapshot))

    def search(self, query: str, top_k: int = 10):
        parsed = self._analyzer.parse(query)
        return self._channel.retrieve(
            parsed,
            top_k=top_k,
            filter_monikers=allowed_monikers(parsed, self._snapshot.symbols),
        )


@pytest.mark.skipif(
    not _artifacts_ready(),
    reason="Eval artifacts not built — run: python scripts/build_eval_artifacts.py",
)
class TestKindFilterAcceptanceEval:
    @pytest.fixture(scope="class")
    def reports(self):
        from sutra.core.embedder.local import LocalEmbedder
        from sutra.core.retrieval.baseline import BaselineRetriever

        cases = load_datasets(_DATASETS_DIR)
        embedder = LocalEmbedder()
        analyzer = QueryAnalyzer(embedder=embedder)
        loader = ArtifactLoader()

        plain, filtered = {}, {}
        for repo in sorted({c.repo for c in cases}):
            snap = loader.load(_ARTIFACTS_DIR / repo)
            plain[repo] = BaselineRetriever(snap, embedder)
            filtered[repo] = _KindFilteredVectorRetriever(snap, analyzer)

        return run_eval(plain, cases), run_eval(filtered, cases)

    def test_kind_filter_lifts_behavioral_and_regresses_nothing(self, reports) -> None:
        baseline, candidate = reports
        cmp = compare(baseline, candidate)
        print("\nKind-filter deltas vs vector baseline:\n" + cmp.summary())
        print("\nKind-filtered eval:\n" + candidate.summary())

        # The filter must not lose queries it previously answered.
        base = baseline.aggregate()
        cand = candidate.aggregate()
        assert cand["recall@10"] >= base["recall@10"], cmp.summary()
        # And it must measurably lift early precision overall (that is its job).
        assert cand["mrr"] > base["mrr"], cmp.summary()
