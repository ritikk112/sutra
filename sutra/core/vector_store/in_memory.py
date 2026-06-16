from __future__ import annotations

from typing import Collection, Optional

import numpy as np

from sutra.core.artifact.loader import ArtifactSnapshot
from sutra.core.vector_store.base import VectorStore


class InMemoryVectorStore(VectorStore):
    """
    Consumer-side vector store: brute-force cosine over the artifact matrix.

    The matrix is L2-normalized once at construction; every query is then a
    single matmul + argpartition.  No index, no persistence, no locks —
    snapshots are immutable, a new artifact means a new store.
    """

    def __init__(self, snapshot: ArtifactSnapshot) -> None:
        self._monikers = snapshot.moniker_order
        self._row_of = snapshot.row_of
        self._dims = snapshot.embedding_dims

        norms = np.linalg.norm(snapshot.vectors, axis=1, keepdims=True)
        self._normalized = snapshot.vectors / np.maximum(norms, 1e-12)

    @property
    def dimensions(self) -> int:
        return self._dims

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        filter_monikers: Optional[Collection[str]] = None,
    ) -> list[tuple[str, float]]:
        if self._normalized.shape[0] == 0 or k <= 0:
            return []
        if query_vec.shape != (self._dims,):
            raise ValueError(
                f"query_vec has shape {query_vec.shape}; expected ({self._dims},)."
            )

        qnorm = np.linalg.norm(query_vec)
        if qnorm < 1e-12:
            return []
        qvec = query_vec / qnorm

        sims = self._normalized @ qvec   # (N,)

        if filter_monikers is not None:
            # Restrict BEFORE the top-k cut.  Rows outside the filter get -inf.
            mask = np.full(sims.shape, -np.inf, dtype=np.float32)
            rows = [self._row_of[m] for m in filter_monikers if m in self._row_of]
            if not rows:
                return []
            mask[rows] = 0.0
            sims = sims + mask

        k = min(k, sims.shape[0])
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        return [
            (self._monikers[i], float(sims[i]))
            for i in top_idx
            if np.isfinite(sims[i])
        ]
