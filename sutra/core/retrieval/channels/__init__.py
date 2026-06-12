"""
Retrieval channels — independent candidate generators fused by RRF (P17).

Each channel sees the same ParsedQuery (analyzed once, P16) and returns a
sorted list[SearchResult] with its own provenance key.  Channel scores are
NOT comparable across channels — fusion uses ranks, never raw scores.
"""
from sutra.core.retrieval.channels.base import Channel
from sutra.core.retrieval.channels.bm25_channel import Bm25Channel
from sutra.core.retrieval.channels.vector_channel import VectorChannel

__all__ = ["Bm25Channel", "Channel", "VectorChannel"]
