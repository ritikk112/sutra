from __future__ import annotations

import hashlib

import numpy as np

from sutra.core.embedder.base import Embedder

# Default dimensions for the fixture embedder — matches all-MiniLM-L6-v2.
DEFAULT_FIXTURE_DIMS = 384


class FixtureEmbedder(Embedder):
    """
    Deterministic fixture embedder for tests and CI.

    Each chunk gets a vector seeded from sha256(chunk_text).  Two runs on the
    same chunk list always produce identical bytes.  No network calls, no API key.

    This class owns all fixture vector generation.  The JsonGraphExporter no
    longer generates its own vectors — it receives (vectors, monikers) from
    whoever calls it, which in tests and CI will be a FixtureEmbedder.
    """

    def __init__(self, dims: int = DEFAULT_FIXTURE_DIMS) -> None:
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, chunks: list[str]) -> np.ndarray:
        if not chunks:
            return np.empty((0, self._dims), dtype=np.float32)

        rows: list[np.ndarray] = []
        for chunk in chunks:
            seed = int(hashlib.sha256(chunk.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            rows.append(rng.random(self._dims).astype(np.float32))

        return np.stack(rows).astype(np.float32)
