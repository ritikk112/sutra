from __future__ import annotations

from typing import Optional

import rustworkx as rx

from sutra.core.artifact.loader import ArtifactSnapshot
from sutra.core.retrieval.types import SearchResult

# Relationship kinds worth walking for retrieval context (SUTRA_PHASE2.md
# stage [5]).  CONTAINS/IMPORTS are structural noise at query time.
EXPAND_KINDS = frozenset({"calls", "extends", "implements", "references"})

DEFAULT_DISCOUNT = 0.5
DEFAULT_MAX_SEEDS = 10


class GraphExpander:
    """
    One-hop neighborhood expansion via an in-memory rustworkx digraph.

    Built once per snapshot from RESOLVED relationships only — this is why
    P20-lite precedes P17: before the heuristic resolver, 100% of CALLS
    edges were unresolved and this stage would walk an empty graph.

    Plain neighbor lookups (one hop = successors + predecessors of a
    node), NOT recursive CTEs, NOT k-hop traversals — per the locked
    architecture.  Neighbors join the result list at
    seed_fused_score × discount, so they can surface but never outrank
    the directly-retrieved evidence that pulled them in.
    """

    def __init__(
        self,
        snapshot: ArtifactSnapshot,
        discount: float = DEFAULT_DISCOUNT,
    ) -> None:
        self._discount = discount
        self._graph: rx.PyDiGraph = rx.PyDiGraph()
        self._node_of: dict[str, int] = {}

        for moniker in snapshot.symbols:
            self._node_of[moniker] = self._graph.add_node(moniker)

        for rel in snapshot.relationships:
            if rel.get("kind") not in EXPAND_KINDS:
                continue
            if not rel.get("is_resolved") or not rel.get("target_id"):
                continue
            src = self._node_of.get(rel["source_id"])
            dst = self._node_of.get(rel["target_id"])
            # Resolved edges to non-indexed targets are valid in the data
            # model but can't join an in-graph expansion.
            if src is None or dst is None or src == dst:
                continue
            self._graph.add_edge(src, dst, rel["kind"])

    @property
    def edge_count(self) -> int:
        return self._graph.num_edges()

    def expand(
        self,
        fused: list[SearchResult],
        top_k: Optional[int] = None,
        max_seeds: int = DEFAULT_MAX_SEEDS,
    ) -> list[SearchResult]:
        """
        Append one-hop neighbors of the top `max_seeds` fused results.

        Existing results are never displaced or rescored — expansion only
        ADDS context candidates below the evidence that seeded them.
        """
        present = {r.moniker for r in fused}
        additions: dict[str, float] = {}

        for seed in fused[:max_seeds]:
            node = self._node_of.get(seed.moniker)
            if node is None:
                continue
            neighbor_nodes = set(self._graph.successor_indices(node)) | set(
                self._graph.predecessor_indices(node)
            )
            for n in neighbor_nodes:
                moniker = self._graph[n]
                if moniker in present:
                    continue
                score = seed.score * self._discount
                if score > additions.get(moniker, 0.0):
                    additions[moniker] = score

        expanded = list(fused) + [
            SearchResult(
                moniker=m, score=s, provenance={"expansion": s}
            )
            for m, s in sorted(additions.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        expanded.sort(key=lambda r: (-r.score, r.moniker))
        return expanded[:top_k] if top_k is not None else expanded
