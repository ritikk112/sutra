"""
Priority 18 — Cross-encoder reranker (opt-in).

Hermetic layer: rerank-document synthesis from the artifact (no model, no
network) and the optional-dependency contract.

Gated layer (needs built eval artifacts AND downloads a small real model on
first run): rerank() behavior with cross-encoder/ms-marco-MiniLM-L-6-v2.
NOTE: MiniLM is used here ONLY to exercise the mechanics cheaply — the P18
acceptance eval measured it NET NEGATIVE on quality (MRR −0.032, 8 query
regressions) while the spec'd BAAI/bge-reranker-v2-m3 lifted MRR +0.104.
bge is the only recommended reranker; it is exercised by
scripts/run_rerank_eval.py, not in the suite (42s/query on CPU).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sutra.core.artifact import ArtifactLoader
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.core.retrieval.reranker import (
    DEFAULT_RERANK_MODEL,
    build_rerank_text,
    rerank,
)
from sutra.core.retrieval.types import SearchResult

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE_REPO = _FIXTURES / "sample_python_repo"
_EVAL_DIR = Path(__file__).parent / "eval"
_SMALL_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("p18_artifact")
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
# Rerank-document synthesis (artifact-only — the zero-infra constraint)
# ---------------------------------------------------------------------------

class TestBuildRerankText:
    def test_method_document_has_header_fields(self, snapshot) -> None:
        moniker = next(
            m for m, s in snapshot.symbols.items()
            if s["kind"] == "method" and "create_user" in m
        )
        text = build_rerank_text(moniker, snapshot)
        assert text.startswith("Method: ")
        assert "File: src/services/user.py" in text
        assert "Signature: " in text

    def test_calls_line_rebuilt_from_relationships(self, snapshot) -> None:
        """The Calls: line comes from the artifact's CALLS edges, NOT from
        source bytes — the consumer has no source tree."""
        moniker = next(m for m in snapshot.symbols if "create_user" in m)
        text = build_rerank_text(moniker, snapshot)
        assert "Calls: " in text
        assert "_generate_id" in text

    def test_class_document_has_extends(self, snapshot) -> None:
        moniker = next(
            m for m, s in snapshot.symbols.items()
            if s["kind"] == "class" and s["name"] == "UserService"
        )
        text = build_rerank_text(moniker, snapshot)
        assert text.startswith("Class: ")
        assert "Extends: Base" in text

    def test_unknown_moniker_degrades_to_name(self, snapshot) -> None:
        assert build_rerank_text("sutra x y z unknown().", snapshot) == (
            "sutra x y z unknown()."
        )

    def test_default_model_is_the_spec_pin(self) -> None:
        assert DEFAULT_RERANK_MODEL == "BAAI/bge-reranker-v2-m3"

    def test_empty_candidates_no_model_needed(self, snapshot) -> None:
        # Must return [] BEFORE touching the model arg — model never loads.
        assert rerank("q", [], snapshot, model="not/a-real-model") == []


# ---------------------------------------------------------------------------
# Real-model behavior (gated; downloads ~80MB once, then hermetic via HF cache)
# ---------------------------------------------------------------------------

def _eval_artifacts_ready() -> bool:
    return (_EVAL_DIR / "artifacts" / "booth" / "graph.json").exists()


@pytest.mark.skipif(
    not _eval_artifacts_ready(),
    reason="Eval artifacts not built — run: python scripts/build_eval_artifacts.py",
)
class TestRerankWithRealModel:
    @pytest.fixture(scope="class")
    def model(self):
        from sutra.core.retrieval.reranker import load_rerank_model
        return load_rerank_model(_SMALL_MODEL)

    @pytest.fixture(scope="class")
    def booth(self):
        return ArtifactLoader().load(_EVAL_DIR / "artifacts" / "booth")

    def test_relevant_candidate_rises(self, model, booth) -> None:
        """Bury the true answer at the bottom of a candidate list; the
        cross-encoder must pull it into the top 3."""
        target = "sutra python booth service/meeting.py MeetingService#create_meeting()."
        distractors = [
            m for m, s in booth.symbols.items()
            if s["kind"] in ("function", "method") and "meeting" not in m.lower()
        ][:19]
        candidates = [
            SearchResult(moniker=m, score=0.9) for m in distractors
        ] + [SearchResult(moniker=target, score=0.01, provenance={"vector": 0.01})]

        out = rerank(
            "which function saves the meeting in the database",
            candidates, booth, model=model,
        )
        ranks = {r.moniker: i for i, r in enumerate(out, start=1)}
        assert ranks[target] <= 3, out[:5]

    def test_provenance_extended_not_replaced(self, model, booth) -> None:
        target = "sutra python booth service/meeting.py MeetingService#create_meeting()."
        out = rerank(
            "save meeting",
            [SearchResult(moniker=target, score=0.5, provenance={"vector": 0.5})],
            booth, model=model,
        )
        assert out[0].provenance["vector"] == 0.5
        assert "rerank" in out[0].provenance
        assert out[0].score == out[0].provenance["rerank"]

    def test_top_k_truncates(self, model, booth) -> None:
        monikers = list(booth.symbols)[:10]
        out = rerank(
            "anything",
            [SearchResult(moniker=m, score=1.0) for m in monikers],
            booth, model=model, top_k=4,
        )
        assert len(out) == 4
        scores = [r.score for r in out]
        assert scores == sorted(scores, reverse=True)
