from __future__ import annotations

from typing import Collection, Optional

from sutra.core.retrieval.channels.base import Channel
from sutra.core.retrieval.query import ParsedQuery
from sutra.core.retrieval.text import split_identifier
from sutra.core.retrieval.types import SearchResult

# Tier scores — channel-local, only their ORDER matters (RRF uses ranks).
_EXACT_NAME = 1.0
_CI_NAME = 0.8
_QUALIFIED_SUFFIX = 0.6
_NAME_PART = 0.4


class MonikerChannel(Channel):
    """
    Exact + pattern lookup over the in-memory symbol index — the third
    channel in the architecture diagram (vector / bm25 / moniker).

    Fires ONLY on the analyzer's extracted entities (identifier-morphology
    tokens like `PostgresDALWrapper`, `find_one`, `models.meeting`).  A
    query with no code-looking tokens contributes nothing here — that's
    the design: this channel is surgical precision for "I know the name"
    queries, the other channels carry everything else.

    Tiers per entity, deduped best-tier-wins:
      1.0  exact `name` match
      0.8  case-insensitive `name` match
      0.6  qualified_name suffix match ("models.meeting.Meeting" ⊇
           "meeting.Meeting"); also dotted-entity exact match
      0.4  entity equals one identifier-part of the name
           ("retry" → lazyRetry) — weakest, still name-anchored
    """

    name = "moniker"

    def __init__(self, snapshot_symbols: dict[str, dict]) -> None:
        self._symbols = snapshot_symbols

        self._by_name: dict[str, list[str]] = {}
        self._by_ci_name: dict[str, list[str]] = {}
        self._by_part: dict[str, list[str]] = {}
        for moniker in sorted(snapshot_symbols):   # deterministic order
            sym = snapshot_symbols[moniker]
            name = sym.get("name") or ""
            if not name:
                continue
            self._by_name.setdefault(name, []).append(moniker)
            self._by_ci_name.setdefault(name.lower(), []).append(moniker)
            for part in set(split_identifier(name)):
                self._by_part.setdefault(part, []).append(moniker)

    def retrieve(
        self,
        query: ParsedQuery,
        top_k: int = 50,
        filter_monikers: Optional[Collection[str]] = None,
    ) -> list[SearchResult]:
        if not query.entities:
            return []

        best: dict[str, float] = {}

        def offer(moniker: str, score: float) -> None:
            if score > best.get(moniker, 0.0):
                best[moniker] = score

        for entity in query.entities:
            for m in self._by_name.get(entity, ()):
                offer(m, _EXACT_NAME)
            for m in self._by_ci_name.get(entity.lower(), ()):
                offer(m, _CI_NAME)
            if "." in entity:
                suffix = entity.lower()
                for moniker, sym in self._symbols.items():
                    qn = (sym.get("qualified_name") or "").lower()
                    if qn == suffix or qn.endswith("." + suffix):
                        offer(moniker, _QUALIFIED_SUFFIX)
            else:
                for m in self._by_part.get(entity.lower(), ()):
                    offer(m, _NAME_PART)

        if filter_monikers is not None:
            allowed = set(filter_monikers)
            best = {m: s for m, s in best.items() if m in allowed}

        ordered = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [
            SearchResult(moniker=m, score=s, provenance={self.name: s})
            for m, s in ordered
        ]
