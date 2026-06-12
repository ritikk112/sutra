from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection, Optional

from sutra.core.retrieval.query import ParsedQuery
from sutra.core.retrieval.types import SearchResult


class Channel(ABC):
    """
    One retrieval channel: ParsedQuery → ranked candidates.

    Introduced at P15, with the second channel, per the design-pattern
    decisions (an ABC with one impl is premature; with two it's a seam).

    Contract
    --------
    - `name` is the channel's stable identifier; it keys the provenance
      dict on every SearchResult the channel emits and the per-channel
      rank map in RRF fusion (P17).
    - retrieve() returns up to top_k results sorted best-first by the
      channel's OWN score.  Scores are channel-local — fusion uses ranks.
    - `filter_monikers`, when given, restricts candidates BEFORE the
      top-k cut (the P16 kind filter pre-restriction).
    """

    #: stable channel identifier — override in every subclass.
    name: str = ""

    @abstractmethod
    def retrieve(
        self,
        query: ParsedQuery,
        top_k: int = 50,
        filter_monikers: Optional[Collection[str]] = None,
    ) -> list[SearchResult]:
        """Return up to top_k candidates, best first."""
