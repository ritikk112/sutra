from __future__ import annotations

from typing import Any, Optional, Sequence

from sutra.core.artifact.loader import ArtifactSnapshot
from sutra.core.embedder.base import Embedder
from sutra.core.retrieval.baseline import validate_embedder_matches_snapshot
from sutra.core.retrieval.channels.base import Channel
from sutra.core.retrieval.channels.bm25_channel import Bm25Channel
from sutra.core.retrieval.channels.moniker_channel import MonikerChannel
from sutra.core.retrieval.channels.vector_channel import VectorChannel
from sutra.core.retrieval.expander import GraphExpander
from sutra.core.retrieval.fusion import DEFAULT_RRF_K, rrf_fuse
from sutra.core.retrieval.kind_filter import allowed_monikers, boost_kinds
from sutra.core.retrieval.query_analyzer import QueryAnalyzer
from sutra.core.retrieval.types import SearchResult
from sutra.core.vector_store.in_memory import InMemoryVectorStore

DEFAULT_CANDIDATES_PER_CHANNEL = 50

# Measured on the 36-query battle-test set (benchmarks/battle_test/): a 1.5x
# vector weight with rrf_k=20 recovers vector-only hits (data symbols, thin-
# text golds) that unweighted RRF's agreement scoring buried, at flat
# recall@5/MRR.  {} restores classic unweighted RRF.
DEFAULT_CHANNEL_WEIGHTS = {"vector": 1.5}
# Reranker input width: top of the expanded list, not the raw channels.
DEFAULT_RERANK_CANDIDATES = 50


class RetrievalPipeline:
    """
    The Phase 2 retrieval pipeline — every stage in-memory, built from one
    artifact snapshot at construction:

        analyzer → channels (vector ∥ bm25 ∥ moniker, kind-pre-restricted)
                 → RRF fusion → one-hop graph expansion
                 [→ cross-encoder reranker, opt-in per query]

    Composition over configuration: channels and the expander are
    injectable for eval A/Bs, but the defaults ARE the product
    configuration.  No stage imports a concrete sibling — channels meet
    fusion only through list[SearchResult].

    The reranker stays opt-in (`rerank=True` per query) because it is the
    one stage with real wall-clock cost; everything else is sub-10ms at
    our scale.
    """

    def __init__(
        self,
        snapshot: ArtifactSnapshot,
        embedder: Embedder,
        analyzer: Optional[QueryAnalyzer] = None,
        channels: Optional[Sequence[Channel]] = None,
        expander: Optional[GraphExpander] = None,
        # Default OFF per the P17 acceptance rule: expansion measured
        # +0.000 recall@10 on the eval datasets (threshold was ≥+0.05).
        # Caveat recorded in PROGRESS.md: recall@10 was already 0.969 —
        # ceiling effect; re-measure if harder datasets land.  The MCP
        # graph tools (callers/callees/neighbors) are unaffected — they
        # use GraphTraversal directly.
        expand: bool = False,
        rrf_k: int = DEFAULT_RRF_K,
        candidates_per_channel: int = DEFAULT_CANDIDATES_PER_CHANNEL,
        rerank_model: Any = None,
        # How an inferred kind hint is applied (KIND_FILTER_AB.md):
        #   "soft" (default) — post-fusion score boost by `kind_boost`;
        #     wrong hints demote the gold slightly instead of erasing it.
        #   "hard" — the original pre-filter: non-matching kinds are removed
        #     from every channel's candidate pool before ranking.  A wrong
        #     hint zeroes recall for that query; kept for A/B comparison.
        #   "off"  — hints are ignored entirely.
        kind_mode: str = "soft",
        kind_boost: float = 1.3,
        # Boost for hints derived from the behavioral-VERB fallback (weaker
        # evidence than an explicit kind noun).  None = same as kind_boost.
        kind_boost_verb: Optional[float] = None,
        # Per-channel RRF weights; unlisted channels → 1.0.  None = the
        # measured default (vector 1.5); pass {} for classic unweighted RRF.
        channel_weights: Optional[dict[str, float]] = None,
    ) -> None:
        if kind_mode not in ("hard", "soft", "off"):
            raise ValueError(
                f"kind_mode must be 'hard', 'soft' or 'off', got {kind_mode!r}"
            )
        validate_embedder_matches_snapshot(snapshot, embedder)
        self._snapshot = snapshot
        self._analyzer = analyzer or QueryAnalyzer(embedder=embedder)
        self._channels: Sequence[Channel] = channels if channels is not None else (
            VectorChannel(InMemoryVectorStore(snapshot)),
            Bm25Channel(snapshot),
            MonikerChannel(snapshot.symbols),
        )
        self._expander = expander or GraphExpander(snapshot)
        self._expand = expand
        self._rrf_k = rrf_k
        self._candidates = candidates_per_channel
        self._rerank_model = rerank_model
        self._kind_mode = kind_mode
        self._kind_boost = kind_boost
        self._kind_boost_verb = kind_boost_verb
        self._channel_weights = (
            DEFAULT_CHANNEL_WEIGHTS if channel_weights is None else channel_weights
        )

    @property
    def snapshot(self) -> ArtifactSnapshot:
        return self._snapshot

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = False,
    ) -> list[SearchResult]:
        parsed = self._analyzer.parse(query)
        # Hard mode is the only mode that restricts the candidate pool;
        # soft mode lets every channel rank the full corpus and applies
        # the hint after fusion (see boost_kinds).
        allowed = (
            allowed_monikers(parsed, self._snapshot.symbols)
            if self._kind_mode == "hard"
            else None
        )

        per_channel = {
            ch.name: ch.retrieve(
                parsed, top_k=self._candidates, filter_monikers=allowed
            )
            for ch in self._channels
        }

        fused = rrf_fuse(
            per_channel, k=self._rrf_k, channel_weights=self._channel_weights
        )
        if self._kind_mode == "soft":
            fused = boost_kinds(
                fused, parsed, self._snapshot.symbols, self._kind_boost,
                verb_weight=self._kind_boost_verb,
            )

        results = (
            self._expander.expand(fused) if self._expand else fused
        )

        if rerank:
            from sutra.core.retrieval.reranker import (  # noqa: PLC0415
                DEFAULT_RERANK_MODEL,
                load_rerank_model,
                rerank as _rerank,
            )
            if self._rerank_model is None:
                self._rerank_model = load_rerank_model(DEFAULT_RERANK_MODEL)
            elif isinstance(self._rerank_model, str):
                self._rerank_model = load_rerank_model(self._rerank_model)
            return _rerank(
                query,
                results[:DEFAULT_RERANK_CANDIDATES],
                self._snapshot,
                model=self._rerank_model,
                top_k=top_k,
            )

        return results[:top_k]
