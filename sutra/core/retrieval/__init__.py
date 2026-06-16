"""
Phase 2 retrieval stack — runs entirely in-memory from the artifact.

The consumer (MCP) and the eval harness share this stack:
    artifact (graph.json + embeddings.npy) → ArtifactSnapshot →
    channels (vector / BM25 / moniker) → kind filter → RRF fusion →
    graph expansion → optional reranker → list[SearchResult]

No module in this package may touch Postgres.  Postgres is indexer-side
only (incremental-update bookkeeping); the retrieval pipeline is built
from the artifact at boot and serves queries in-process.
"""
from sutra.core.retrieval.types import SearchResult

__all__ = ["SearchResult"]
