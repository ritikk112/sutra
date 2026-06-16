from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SearchResult:
    """
    A single retrieval hit.

    Introduced at P12 (eval harness) so the contract between channels,
    fusion, reranker, harness, and MCP is fixed from day one and does not
    churn through P15–P18.

    Attributes
    ----------
    moniker : str
        The symbol's SCIP-style moniker (globally unique id).
    score : float
        Channel- or stage-specific relevance score.  Higher is better.
        Scores are NOT comparable across channels — RRF fusion (P17)
        deliberately uses ranks, not raw scores.
    provenance : dict[str, float]
        Which stage(s) produced/boosted this result and with what score,
        e.g. {"vector": 0.83} from the cosine channel, later merged to
        {"vector": 0.83, "bm25": 11.2, "rrf": 0.031}.  The eval harness
        and MCP surface this for debuggability; fusion populates it.
    """

    moniker: str
    score: float
    provenance: dict[str, float] = field(default_factory=dict)
