from __future__ import annotations

from typing import Collection, Optional

from sutra.core.retrieval.channels.base import Channel
from sutra.core.retrieval.query import ParsedQuery
from sutra.core.retrieval.types import SearchResult
from sutra.core.vector_store.base import VectorStore


class VectorChannel(Channel):
    """
    The embedding channel: cosine similarity via a VectorStore.

    In the consumer stack the store is InMemoryVectorStore (NumPy over the
    artifact); the channel itself is store-agnostic.  It does NOT embed the
    query — the analyzer does that once (ParsedQuery.embedding) so no
    channel pays for or duplicates embedding work.
    """

    name = "vector"

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def retrieve(
        self,
        query: ParsedQuery,
        top_k: int = 50,
        filter_monikers: Optional[Collection[str]] = None,
    ) -> list[SearchResult]:
        if query.embedding is None:
            raise ValueError(
                "VectorChannel requires ParsedQuery.embedding — the query "
                "analyzer must embed the query before vector retrieval."
            )
        hits = self._store.search(
            query.embedding, k=top_k, filter_monikers=filter_monikers
        )
        return [
            SearchResult(moniker=m, score=sim, provenance={self.name: sim})
            for m, sim in hits
        ]
