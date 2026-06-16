from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sutra.core.extractor.base import Relationship, Symbol


@dataclass(slots=True)
class ResolutionStats:
    """
    Outcome of one resolver pass — the acceptance evidence for P20.

    `matchable` counts unresolved CALLS whose callee short-name exists as
    a symbol name in the repo: the only edges ANY in-repo resolver
    (heuristic or LSP) could possibly flip.  Builtins and third-party
    callees (logger.error, isinstance, …) are structurally out of reach
    and excluded from the denominator.
    """

    total_calls: int = 0
    unresolved_before: int = 0
    matchable: int = 0
    resolved: int = 0
    by_rule: dict[str, int] = field(default_factory=dict)

    @property
    def resolution_rate(self) -> float:
        """resolved / matchable — the P20-lite acceptance metric (≥0.60)."""
        return self.resolved / self.matchable if self.matchable else 0.0


class Resolver(ABC):
    """
    Upgrades unresolved relationships IN PLACE.

    Runs inside Indexer.index() after aggregation, before chunk building
    and export.  Implementations must only set `target_id`,
    `is_resolved`, and metadata — `target_name` stays untouched so
    embedding chunks (built from target_name) are byte-identical with
    and without a resolver.  Vectors stay valid across resolver changes.
    """

    @abstractmethod
    def resolve(
        self,
        symbols: list[Symbol],
        relationships: list[Relationship],
    ) -> ResolutionStats:
        """Mutate `relationships`; return the stats."""
