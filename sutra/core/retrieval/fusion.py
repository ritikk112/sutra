from __future__ import annotations

from typing import Mapping, Optional, Sequence

from sutra.core.retrieval.types import SearchResult

# The standard RRF constant.  k=60 dampens the head so one channel's #1
# can't drown out consistent mid-rank agreement from the others.
DEFAULT_RRF_K = 60


def rrf_fuse(
    channel_results: Mapping[str, Sequence[SearchResult]],
    k: int = DEFAULT_RRF_K,
    top_k: Optional[int] = None,
) -> list[SearchResult]:
    """
    Reciprocal Rank Fusion across channels:

        score(d) = Σ_channels 1 / (k + rank_channel(d))

    FUSION assigns the ranks itself by enumerating each channel's list —
    channels return sorted lists and that is their whole contract; rank
    assignment lives here so the contract can't be silently violated by a
    channel emitting its own rank numbers.

    Raw channel scores are never compared across channels (BM25 ~11.2 vs
    cosine ~0.83 are different universes) — only ranks enter the formula.
    Provenance survives: the fused result carries the union of every
    channel's provenance plus {"rrf": fused_score}.

    Ties break on moniker (ascending) for full determinism.
    """
    fused_scores: dict[str, float] = {}
    merged_provenance: dict[str, dict[str, float]] = {}

    for _, results in sorted(channel_results.items()):
        for rank, result in enumerate(results, start=1):
            fused_scores[result.moniker] = (
                fused_scores.get(result.moniker, 0.0) + 1.0 / (k + rank)
            )
            merged_provenance.setdefault(result.moniker, {}).update(
                result.provenance
            )

    ordered = sorted(fused_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        ordered = ordered[:top_k]

    return [
        SearchResult(
            moniker=moniker,
            score=score,
            provenance={**merged_provenance.get(moniker, {}), "rrf": score},
        )
        for moniker, score in ordered
    ]
