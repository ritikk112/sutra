from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Embedder(ABC):
    """
    Abstract interface for all embedding providers.

    Contract:
    - `embed(chunks)` returns float32 array of shape (len(chunks), self.dimensions).
    - Empty input → shape (0, self.dimensions).
    - Raises on batch failure after retries (fail-loud, caller re-runs).
    - `dimensions` is known at construction, before any embed() call.
    """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions per embedding vector."""

    @abstractmethod
    def embed(self, chunks: list[str]) -> np.ndarray:
        """
        Embed text chunks and return float32 array of shape (len(chunks), self.dimensions).

        Batching is the embedder's responsibility — the caller passes the full
        chunk list and the embedder batches internally.  This means event-loop
        setup cost (for async providers) is paid once per embed() call, not per batch.
        """

    def usage_stats(self) -> dict[str, Any] | None:
        """
        Optional provider-specific usage metadata for the most recent embed run.

        Returns None for providers that do not expose token/accounting data.
        """
        return None
