from __future__ import annotations

from typing import Collection, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from sutra.core.artifact.loader import ArtifactSnapshot
from sutra.core.retrieval.channels.base import Channel
from sutra.core.retrieval.query import ParsedQuery
from sutra.core.retrieval.text import tokenize
from sutra.core.retrieval.types import SearchResult

# BM25F-style field weights, emulated by token repetition (rank-bm25 has no
# native field weighting).  Spec: name (A), qualified_name (A),
# signature (B), docstring (C).
_FIELD_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("name", 3),
    ("qualified_name", 3),
    ("signature", 2),
    ("docstring", 1),
)


class Bm25Channel(Channel):
    """
    In-memory lexical channel, built at boot from the artifact — the
    consumer never touches a database (no Postgres tsvector).

    Corpus: EVERY symbol in graph.json, including modules and variables.
    This is deliberate and load-bearing: the canonical known-failure query
    ("function that attaches token") can only be answered by a *module*
    symbol (the code is an anonymous interceptor) — and modules are never
    embedded, so the vector channel is structurally blind to them.  The
    lexical channel is where those symbols become reachable.
    """

    name = "bm25"

    def __init__(self, snapshot: ArtifactSnapshot) -> None:
        # Deterministic corpus order — sorted monikers, same as the exporter.
        self._monikers: list[str] = sorted(snapshot.symbols)
        corpus = [
            self._symbol_tokens(snapshot.symbols[m]) for m in self._monikers
        ]
        # BM25Okapi rejects an empty corpus; guard for degenerate artifacts.
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._row_of = {m: i for i, m in enumerate(self._monikers)}

    @staticmethod
    def _symbol_tokens(sym: dict) -> list[str]:
        tokens: list[str] = []
        for field, weight in _FIELD_WEIGHTS:
            value = sym.get(field)
            if not value:
                continue
            tokens.extend(tokenize(str(value)) * weight)
        # Degenerate symbols (no text at all) still need one token so the
        # corpus rows stay aligned with self._monikers.
        return tokens or ["<empty>"]

    def retrieve(
        self,
        query: ParsedQuery,
        top_k: int = 50,
        filter_monikers: Optional[Collection[str]] = None,
    ) -> list[SearchResult]:
        if self._bm25 is None:
            return []

        query_tokens = tokenize(query.text)
        if not query_tokens:
            return []

        scores = np.asarray(self._bm25.get_scores(query_tokens))

        if filter_monikers is not None:
            mask = np.full(scores.shape, -np.inf)
            rows = [self._row_of[m] for m in filter_monikers if m in self._row_of]
            if not rows:
                return []
            mask[rows] = 0.0
            scores = scores + mask

        k = min(top_k, scores.shape[0])
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            SearchResult(
                moniker=self._monikers[i],
                score=float(scores[i]),
                provenance={self.name: float(scores[i])},
            )
            for i in top_idx
            # Zero BM25 score = no query term matched at all — not a candidate.
            if np.isfinite(scores[i]) and scores[i] > 0.0
        ]
