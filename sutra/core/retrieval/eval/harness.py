from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, Sequence, Union

from sutra.core.retrieval.eval.dataset import EvalCase
from sutra.core.retrieval.eval.metrics import (
    first_hit_rank,
    must_include_coverage,
    reciprocal_rank,
)
from sutra.core.retrieval.types import SearchResult

DEFAULT_KS: tuple[int, ...] = (1, 5, 10)


class Searcher(Protocol):
    """Structural contract every retriever satisfies — no ABC needed yet."""

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]: ...


@dataclass(frozen=True, slots=True)
class QueryEval:
    """Per-query outcome — kept whole so drift reports can name queries."""

    case: EvalCase
    results: tuple[SearchResult, ...]
    first_hit_rank: Optional[int]
    reciprocal_rank: float
    hit_at: dict[int, bool]                    # k → any expected in top-k
    must_include_coverage: Optional[float]     # at max(ks); None if not declared


@dataclass(slots=True)
class EvalReport:
    """Aggregate + per-query eval outcome for one retriever over a dataset."""

    per_query: list[QueryEval]
    ks: tuple[int, ...] = DEFAULT_KS

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self, category: Optional[str] = None) -> dict[str, float]:
        rows = [
            q for q in self.per_query
            if category is None or q.case.category == category
        ]
        if not rows:
            return {"n": 0}
        out: dict[str, float] = {"n": len(rows)}
        for k in self.ks:
            out[f"recall@{k}"] = sum(q.hit_at[k] for q in rows) / len(rows)
        out["mrr"] = sum(q.reciprocal_rank for q in rows) / len(rows)
        return out

    def by_category(self) -> dict[str, dict[str, float]]:
        cats = sorted({q.case.category for q in self.per_query})
        return {c: self.aggregate(c) for c in cats}

    def to_dict(self) -> dict:
        """JSON-serializable snapshot — what tests/eval/baselines/ stores."""
        return {
            "ks": list(self.ks),
            "overall": self.aggregate(),
            "by_category": self.by_category(),
            "per_query": [
                {
                    "query": q.case.query,
                    "repo": q.case.repo,
                    "category": q.case.category,
                    "first_hit_rank": q.first_hit_rank,
                    "top": [r.moniker for r in q.results[:3]],
                }
                for q in self.per_query
            ],
        }

    def summary(self) -> str:
        """Human-readable table for terminal output / PROGRESS.md."""
        lines = [
            f"{'category':<16} {'n':>3} "
            + " ".join(f"{'r@' + str(k):>6}" for k in self.ks)
            + f" {'mrr':>6}"
        ]

        def fmt(label: str, agg: dict[str, float]) -> str:
            cells = " ".join(f"{agg.get(f'recall@{k}', 0.0):>6.3f}" for k in self.ks)
            return f"{label:<16} {agg['n']:>3} {cells} {agg.get('mrr', 0.0):>6.3f}"

        for cat, agg in self.by_category().items():
            lines.append(fmt(cat, agg))
        lines.append(fmt("OVERALL", self.aggregate()))
        return "\n".join(lines)


def run_eval(
    retriever: Union[Searcher, Mapping[str, Searcher]],
    cases: Sequence[EvalCase],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> EvalReport:
    """
    Run every case through the retriever and collect metrics.

    `retriever` is either a single Searcher (single-repo eval) or a mapping
    {repo_name → Searcher} when cases span multiple reference repos.
    Results are fetched once at top_k = max(ks); all metrics derive from
    that single ranked list.
    """
    max_k = max(ks)
    per_query: list[QueryEval] = []

    for case in cases:
        if isinstance(retriever, Mapping):
            searcher = retriever.get(case.repo)
            if searcher is None:
                raise KeyError(
                    f"No retriever supplied for repo {case.repo!r} "
                    f"(have: {sorted(retriever)})."
                )
        else:
            searcher = retriever

        results = tuple(searcher.search(case.query, top_k=max_k))
        rank = first_hit_rank(case.expected, results)
        per_query.append(QueryEval(
            case=case,
            results=results,
            first_hit_rank=rank,
            reciprocal_rank=reciprocal_rank(case.expected, results),
            hit_at={k: (rank is not None and rank <= k) for k in ks},
            must_include_coverage=must_include_coverage(
                case.must_include, results, max_k
            ),
        ))

    return EvalReport(per_query=per_query, ks=ks)


@dataclass(frozen=True, slots=True)
class QueryDelta:
    query: str
    repo: str
    category: str
    rank_before: Optional[int]
    rank_after: Optional[int]


@dataclass(slots=True)
class Comparison:
    """A/B between two EvalReports over the same case list."""

    overall_delta: dict[str, float]
    category_delta: dict[str, dict[str, float]]
    improved: list[QueryDelta] = field(default_factory=list)
    regressed: list[QueryDelta] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["metric deltas (candidate − baseline):"]
        for metric, d in self.overall_delta.items():
            lines.append(f"  {metric:<10} {d:+.3f}")
        if self.improved:
            lines.append(f"improved ({len(self.improved)}):")
            for q in self.improved:
                lines.append(
                    f"  [{q.category}] {q.query!r}: rank {q.rank_before} → {q.rank_after}"
                )
        if self.regressed:
            lines.append(f"regressed ({len(self.regressed)}):")
            for q in self.regressed:
                lines.append(
                    f"  [{q.category}] {q.query!r}: rank {q.rank_before} → {q.rank_after}"
                )
        return "\n".join(lines)


def compare(baseline: EvalReport, candidate: EvalReport) -> Comparison:
    """
    A/B two reports.  Requires identical case lists (same queries, same
    order) — comparing different datasets is meaningless and raises.
    """
    base_keys = [(q.case.repo, q.case.query) for q in baseline.per_query]
    cand_keys = [(q.case.repo, q.case.query) for q in candidate.per_query]
    if base_keys != cand_keys:
        raise ValueError(
            "compare() requires both reports to cover the same cases in the "
            "same order — run both retrievers over one dataset."
        )

    base_overall = baseline.aggregate()
    cand_overall = candidate.aggregate()
    overall_delta = {
        m: cand_overall[m] - base_overall[m]
        for m in base_overall
        if m != "n"
    }

    base_cats = baseline.by_category()
    cand_cats = candidate.by_category()
    category_delta = {
        c: {
            m: cand_cats[c][m] - base_cats[c][m]
            for m in base_cats[c]
            if m != "n"
        }
        for c in base_cats
    }

    improved: list[QueryDelta] = []
    regressed: list[QueryDelta] = []
    for qb, qc in zip(baseline.per_query, candidate.per_query):
        delta = QueryDelta(
            query=qb.case.query,
            repo=qb.case.repo,
            category=qb.case.category,
            rank_before=qb.first_hit_rank,
            rank_after=qc.first_hit_rank,
        )
        rb = qb.first_hit_rank or 10**9
        rc = qc.first_hit_rank or 10**9
        if rc < rb:
            improved.append(delta)
        elif rc > rb:
            regressed.append(delta)

    return Comparison(
        overall_delta=overall_delta,
        category_delta=category_delta,
        improved=improved,
        regressed=regressed,
    )
