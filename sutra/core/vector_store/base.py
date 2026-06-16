from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection, Optional

import numpy as np


class VectorStore(ABC):
    """
    Read contract for similarity search over (moniker, embedding) pairs.

    Score semantics — the load-bearing part of this contract:
        search() returns COSINE SIMILARITY, higher is better, sorted
        descending.  (pgvector's native `<=>` is cosine *distance*;
        implementations convert: similarity = 1 - distance.)

    `filter_monikers`, when given, restricts the search to that candidate
    set BEFORE the top-k cut — a kind filter that drops 80% of candidates
    must not also shrink the result list.
    """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensionality this store holds."""

    @abstractmethod
    def search(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        filter_monikers: Optional[Collection[str]] = None,
    ) -> list[tuple[str, float]]:
        """
        Return up to k (moniker, cosine_similarity) pairs, best first.

        query_vec: shape (dims,).  Raises ValueError on dims mismatch.
        """
