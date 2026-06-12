"""
CALLS resolution — the seam DESIGN.md promised but Phase 1 never built.

Phase 1 leaves every cross-file CALLS relationship unresolved
(target_id=None, target_name=the literal callee).  P20-lite adds the
`Resolver` ABC and a heuristic first implementation that flips
is_resolved in place for intra-repo calls; P20-full adds an LSP-backed
implementation behind the same seam.
"""
from sutra.core.resolver.base import ResolutionStats, Resolver
from sutra.core.resolver.heuristic import HeuristicResolver

__all__ = ["HeuristicResolver", "ResolutionStats", "Resolver"]
