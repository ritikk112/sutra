from __future__ import annotations

from sutra.core.artifact.loader import ArtifactSnapshot
from sutra.core.embedder.base import Embedder
from sutra.core.retrieval.channels.vector_channel import VectorChannel
from sutra.core.retrieval.query import ParsedQuery
from sutra.core.retrieval.types import SearchResult
from sutra.core.vector_store.in_memory import InMemoryVectorStore


def validate_embedder_matches_snapshot(
    snapshot: ArtifactSnapshot, embedder: Embedder
) -> None:
    """
    Refuse to mix vector spaces — shared by every consumer that pairs a
    query-time embedder with an artifact (baseline retriever, pipeline, MCP).
    """
    if (
        snapshot.embedding_model_id is not None
        and snapshot.embedding_model_id != embedder.model_id
    ):
        raise ValueError(
            f"Embedding model mismatch: artifact {snapshot.path} was built "
            f"with {snapshot.embedding_model_id!r} but the query embedder is "
            f"{embedder.model_id!r}.  Re-index or switch embedders."
        )
    if snapshot.embedding_dims != embedder.dimensions:
        raise ValueError(
            f"Embedding dims mismatch: artifact has {snapshot.embedding_dims} "
            f"dims but the query embedder produces {embedder.dimensions}."
        )


class BaselineRetriever:
    """
    P12 baseline: single-channel brute-force cosine over embeddings.npy.

    This is deliberately the dumbest possible retriever — it IS the thing
    the rest of Phase 2 has to beat.  Since P15 it delegates to the shared
    VectorChannel/InMemoryVectorStore stack (one cosine implementation in
    the codebase), but its public contract and scores are unchanged.

    Concrete class on purpose: the `Retriever` ABC is deferred until a
    second top-level retriever exists (see SUTRA_PHASE2.md design
    decisions).  The harness depends only on the structural contract
    `search(query, top_k) -> list[SearchResult]`.
    """

    def __init__(self, snapshot: ArtifactSnapshot, embedder: Embedder) -> None:
        validate_embedder_matches_snapshot(snapshot, embedder)
        self._snapshot = snapshot
        self._embedder = embedder
        self._channel = VectorChannel(InMemoryVectorStore(snapshot))

    @property
    def snapshot(self) -> ArtifactSnapshot:
        return self._snapshot

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Embed `query`, score every row by cosine similarity, return top_k."""
        qvec = self._embedder.embed([query])[0]
        parsed = ParsedQuery(text=query, embedding=qvec)
        return self._channel.retrieve(parsed, top_k=top_k)
