from __future__ import annotations

from typing import Iterable, Optional, Sequence

from sutra.core.retrieval.types import SearchResult


def first_hit_rank(
    expected: Iterable[str],
    results: Sequence[SearchResult],
) -> Optional[int]:
    """
    1-based rank of the first result whose moniker is in `expected`.
    None if no expected moniker appears at all.
    """
    expected_set = set(expected)
    for rank, r in enumerate(results, start=1):
        if r.moniker in expected_set:
            return rank
    return None


def recall_at_k(
    expected: Iterable[str],
    results: Sequence[SearchResult],
    k: int,
) -> float:
    """
    Per-query hit indicator: 1.0 if ANY expected moniker is in the top k,
    else 0.0.  `expected` is an any-of equivalence set (several symbols can
    answer one query), so this is hit@k semantics; the aggregate over a
    dataset (mean) is the recall@k we report.
    """
    rank = first_hit_rank(expected, results[:k])
    return 1.0 if rank is not None else 0.0


def reciprocal_rank(
    expected: Iterable[str],
    results: Sequence[SearchResult],
) -> float:
    """1/rank of the first expected hit; 0.0 if absent.  Mean over a dataset = MRR."""
    rank = first_hit_rank(expected, results)
    return 0.0 if rank is None else 1.0 / rank


def must_include_coverage(
    must_include: Iterable[str],
    results: Sequence[SearchResult],
    k: int,
) -> Optional[float]:
    """
    Fraction of `must_include` monikers present in the top k.
    None when the case declares no must_include set (not aggregated).
    """
    required = list(must_include)
    if not required:
        return None
    present = {r.moniker for r in results[:k]}
    return sum(1 for m in required if m in present) / len(required)
