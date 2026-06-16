"""
Vector store seam — one read contract, two worlds.

`VectorStore` (ABC) defines similarity search over (moniker, vector) pairs.
Two implementations with sharply different cost models:

- `PGVectorStore` (sutra.core.graph.pgvector_store) — indexer-side, HNSW
  in Postgres.  Used by the indexing pipelines.
- `InMemoryVectorStore` — consumer-side, NumPy brute-force cosine over the
  artifact's embeddings.npy.  Used by retrieval channels and MCP.  At our
  scale (~150k vectors max) a matmul beats an indexed SQL round-trip.

`filter_monikers` exists so the P16 kind filter can pre-restrict the
candidate set instead of post-filtering a truncated top-k.
"""
from sutra.core.vector_store.base import VectorStore
from sutra.core.vector_store.in_memory import InMemoryVectorStore

__all__ = ["InMemoryVectorStore", "VectorStore"]
