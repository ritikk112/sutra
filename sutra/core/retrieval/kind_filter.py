from __future__ import annotations

from typing import Mapping, Optional

from sutra.core.retrieval.query import ParsedQuery
from sutra.core.retrieval.types import SearchResult


def allowed_monikers(
    query: ParsedQuery,
    symbols: Mapping[str, dict],
) -> Optional[set[str]]:
    """
    Pre-restriction set for VectorStore.filter_monikers / Channel.retrieve.

    Returns the monikers whose kind matches the query's hint, or None when
    the query carries no confident hint (None = "don't restrict anything").
    Preferred over post-filtering: dropping 80% of candidates must not also
    shrink the top-k.
    """
    if query.kind_hint is None:
        return None
    return {
        moniker
        for moniker, sym in symbols.items()
        if sym.get("kind") in query.kind_hint
    }


def apply_kind_filter(
    results: list[SearchResult],
    query: ParsedQuery,
    symbols: Mapping[str, dict],
) -> list[SearchResult]:
    """
    Post-hoc variant: drop candidates whose kind contradicts the hint.

    Conservative by construction — when kind_hint is None (no explicit
    nouns, ambiguous nouns, or no verbs) nothing is dropped.  Unknown
    monikers (not in `symbols`) are kept: filtering must never lose a
    result over a lookup gap.
    """
    if query.kind_hint is None:
        return results
    kept = []
    for r in results:
        sym = symbols.get(r.moniker)
        if sym is None or sym.get("kind") in query.kind_hint:
            kept.append(r)
    return kept
