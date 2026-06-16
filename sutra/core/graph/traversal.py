"""
Consumer-side graph traversal — the OTHER half of the split that
sutra/core/graph/base.py documents.

`IncrementalReader` (indexer-side, SQL, per-file diff) and
`GraphTraversal` (consumer-side, rustworkx, in-memory pointer chase) are
deliberately separate interfaces: their cost models differ by ~500×, and
one shared ABC would invite calling a "1ms verb" that table-scans SQL.

GraphTraversal is built from an ArtifactSnapshot at MCP boot — no
database exists on the consumer side, ever.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Optional

import rustworkx as rx

from sutra.core.artifact.loader import ArtifactSnapshot

# All relationship kinds present in graph.json (RelationKind values).
ALL_EDGE_KINDS = frozenset({
    "calls", "extends", "implements", "imports",
    "contains", "references", "returns_type", "parameter_type",
})


@dataclass(frozen=True, slots=True)
class Neighbor:
    """One traversal hit: which symbol, via what edge, in which direction."""

    moniker: str
    edge_kind: str
    direction: str        # "out" (moniker is the target) | "in" (the source)
    depth: int            # 1 = direct neighbor


class GraphTraversal(ABC):
    """MCP-side traversal verbs over the in-memory relationship graph."""

    @abstractmethod
    def get_callers(self, moniker: str) -> list[Neighbor]:
        """Symbols with a resolved CALLS edge INTO `moniker`."""

    @abstractmethod
    def get_callees(self, moniker: str) -> list[Neighbor]:
        """Symbols `moniker` has a resolved CALLS edge to."""

    @abstractmethod
    def expand_neighbors(
        self,
        moniker: str,
        depth: int = 1,
        kinds: Optional[Collection[str]] = None,
    ) -> list[Neighbor]:
        """BFS neighborhood up to `depth` hops over the given edge kinds
        (default: all), both directions."""


class RustworkxTraversal(GraphTraversal):
    """rustworkx-backed implementation, built once per snapshot.

    Only RESOLVED relationships become edges — an unresolved edge has no
    target node to walk to.  Edge payload is the relationship kind.
    """

    def __init__(self, snapshot: ArtifactSnapshot) -> None:
        self._graph: rx.PyDiGraph = rx.PyDiGraph()
        self._node_of: dict[str, int] = {}

        for moniker in snapshot.symbols:
            self._node_of[moniker] = self._graph.add_node(moniker)

        for rel in snapshot.relationships:
            if not rel.get("is_resolved") or not rel.get("target_id"):
                continue
            src = self._node_of.get(rel["source_id"])
            dst = self._node_of.get(rel["target_id"])
            if src is None or dst is None:
                continue
            self._graph.add_edge(src, dst, rel["kind"])

    # ------------------------------------------------------------------

    def get_callers(self, moniker: str) -> list[Neighbor]:
        return self._directed(moniker, "calls", incoming=True)

    def get_callees(self, moniker: str) -> list[Neighbor]:
        return self._directed(moniker, "calls", incoming=False)

    def expand_neighbors(
        self,
        moniker: str,
        depth: int = 1,
        kinds: Optional[Collection[str]] = None,
    ) -> list[Neighbor]:
        start = self._node_of.get(moniker)
        if start is None:
            return []
        kind_set = frozenset(kinds) if kinds is not None else ALL_EDGE_KINDS

        seen: dict[int, int] = {start: 0}
        frontier = [start]
        out: list[Neighbor] = []

        for level in range(1, max(1, depth) + 1):
            next_frontier: list[int] = []
            for node in frontier:
                for _, dst, kind in self._graph.out_edges(node):
                    if kind in kind_set and dst not in seen:
                        seen[dst] = level
                        next_frontier.append(dst)
                        out.append(Neighbor(self._graph[dst], kind, "out", level))
                for src, _, kind in self._graph.in_edges(node):
                    if kind in kind_set and src not in seen:
                        seen[src] = level
                        next_frontier.append(src)
                        out.append(Neighbor(self._graph[src], kind, "in", level))
            frontier = next_frontier
            if not frontier:
                break

        return out

    # ------------------------------------------------------------------

    def _directed(self, moniker: str, kind: str, incoming: bool) -> list[Neighbor]:
        node = self._node_of.get(moniker)
        if node is None:
            return []
        edges = self._graph.in_edges(node) if incoming else self._graph.out_edges(node)
        hits = []
        for src, dst, edge_kind in edges:
            if edge_kind != kind:
                continue
            other = src if incoming else dst
            hits.append(Neighbor(
                self._graph[other], edge_kind, "in" if incoming else "out", 1
            ))
        # Deterministic order for stable tool output.
        hits.sort(key=lambda n: n.moniker)
        return hits
