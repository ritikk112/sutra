"""
Artifact loading — the consumer-side entry point.

An "artifact" is the directory the indexer exports:
    graph.json            symbols + relationships + metadata (versioned)
    embeddings.npy        float32 (N, dims) matrix
    embeddings_index.json ordered moniker list; row N ↔ monikers[N]
    .ready                sentinel written last by AtomicArtifactWriter (P19)

`ArtifactLoader` is pure (path → `ArtifactSnapshot`) and performs all
boot-time integrity validation.  The MCP-side `SnapshotRegistry` and
`ArtifactWatcher` (P19) compose on top of it; the eval harness (P12) and
retrieval channels load snapshots through it directly.
"""
from sutra.core.artifact.loader import (
    ArtifactError,
    ArtifactLoader,
    ArtifactSnapshot,
)

__all__ = ["ArtifactError", "ArtifactLoader", "ArtifactSnapshot"]
