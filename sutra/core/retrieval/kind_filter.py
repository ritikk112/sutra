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


def boost_kinds(
    results: list[SearchResult],
    query: ParsedQuery,
    symbols: Mapping[str, dict],
    weight: float = 1.3,
    verb_weight: Optional[float] = None,
) -> list[SearchResult]:
    """
    Soft alternative to the hard pre-filter: multiply the fused score of
    hint-matching candidates by `weight` and re-sort, instead of deleting
    non-matching candidates from the pool.

    Rationale (KIND_FILTER_AB.md): the kind hint is *inferred* from one noun
    in the query, and on corpora where kind nouns are domain vocabulary
    ("model" in ML code, "schema" in web code) the inference inverts.  A hard
    filter then has unbounded downside — the gold is erased from every
    channel before ranking and no downstream stage can recover it — while its
    upside is only a reordering of an already-decent list.  A multiplicative
    boost keeps the upside (hint-matching symbols rise) and bounds the
    downside (a wrongly-unboosted gold keeps its fused rank and merely gets
    out-nudged by ~`weight`).

    Keep `weight` modest: at ~3x the boost overwhelms the fused ordering and
    degenerates into the hard filter's behavior (measured on the 12-query
    sutra set: x1.2-1.5 recovers the hard filter's losses at no cost to the
    explicit-kind queries; x3.0 regresses to hard-filter numbers).

    When the query carries no hint, results are returned unchanged.  Unknown
    monikers keep their score: boosting must never lose a result over a
    lookup gap.

    `verb_weight`, when given, replaces `weight` for hints derived from the
    behavioral-verb fallback (query.kind_hint_source == "verb") — a verb is
    weaker evidence of the wanted kind than an explicit noun.
    """
    if query.kind_hint is None:
        return results
    if verb_weight is not None and query.kind_hint_source == "verb":
        weight = verb_weight
    if weight == 1.0:
        return results
    boosted = [
        SearchResult(
            moniker=r.moniker,
            score=r.score * weight,
            provenance={**r.provenance, "kind_boost": weight},
        )
        if (sym := symbols.get(r.moniker)) is not None
        and sym.get("kind") in query.kind_hint
        else r
        for r in results
    ]
    # Same deterministic ordering contract as rrf_fuse: score desc,
    # moniker asc on ties.
    boosted.sort(key=lambda r: (-r.score, r.moniker))
    return boosted


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
