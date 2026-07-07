from __future__ import annotations

import re
from collections import defaultdict

# Sentinel returned by _resolve_local when the innermost visible scope that has
# the requested name is AMBIGUOUS (>1 hit).  Distinct from None ("not found"),
# so the caller can avoid falling through to the cross-file by_name pool.
_AMBIGUOUS_LOCAL: object = object()

from sutra.core.extractor.base import (
    ClassSymbol,
    FunctionSymbol,
    MethodSymbol,
    RelationKind,
    Relationship,
    Symbol,
)
from sutra.core.extractor.moniker import parse_moniker
from sutra.core.resolver.base import ResolutionStats, Resolver

# Callables + classes (constructor calls).  MethodSymbol ⊂ FunctionSymbol.
_CALLABLE_TYPES = (FunctionSymbol, ClassSymbol)

_SEGMENT_SPLIT = re.compile(r"[./\\]+")


def _module_path_of_file(file_path: str) -> str:
    """
    Repo-relative file path → import-style dotted module path.

      a/b/c.py            → a.b.c
      database/__init__.py → database
      src/lib/api-client.ts → src.lib.api-client
      binding/binding.go  → binding.binding
    """
    path = re.sub(r"\.(py|ts|tsx|go)$", "", file_path)
    if path.endswith("/__init__"):
        path = path[: -len("/__init__")]
    return path.replace("/", ".")


def _segments(path: str) -> list[str]:
    """Normalize an import source / module path into comparable segments.

    Strips relative-import markers (leading dots, ./ ../) and frontend
    alias prefixes (@/...), then splits on . / \\ separators.
    """
    cleaned = path.lstrip("@")
    parts = [p for p in _SEGMENT_SPLIT.split(cleaned) if p and p != ".."]
    return parts


def _module_matches(import_source: str, module_path: str) -> bool:
    """
    True when `import_source` plausibly names the module at `module_path`.

    Symmetric suffix match: the shorter segment list must equal the tail
    of the longer one.  Handles Python absolute imports ("database" ↔
    "database"), TS aliases ("@/lib/api-client" ↔ "src.lib.api-client"),
    relative imports ("./utils" ↔ "src.lib.utils"), and Go package paths
    ("github.com/org/repo/binding" ↔ "binding.binding" last-segment-wise).
    """
    a, b = _segments(import_source), _segments(module_path)
    if not a or not b:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return longer[-len(shorter):] == shorter


class HeuristicResolver(Resolver):
    """
    P20-lite: resolve intra-repo CALLS without type inference.

    Per unresolved CALLS edge with callee name N, in precedence order:

      rule "local"   — exactly one symbol named N defined in the CALLER'S
                       OWN FILE → locals shadow everything.
      rule "import"  — the caller's file imports N (alias-aware) from
                       module M, and exactly one candidate named N lives
                       in a module matching M.
      rule "unique"  — exactly one call-form-compatible candidate named N
                       in the whole repo.

    Anything else stays unresolved — ambiguity is left for P20-full (LSP)
    rather than guessed at.  Precision over recall: a wrong edge poisons
    graph expansion (P17); a missing edge just doesn't help it.

    Call-form compatibility (used by "unique"): a `direct` call prefers
    free functions/classes over methods; a `method` call prefers methods.
    Preference only narrows when the preferred set is non-empty.
    """

    def resolve(
        self,
        symbols: list[Symbol],
        relationships: list[Relationship],
    ) -> ResolutionStats:
        stats = ResolutionStats()

        # name -> NON-LOCAL candidate symbols (locals resolve via scope, below)
        by_name: dict[str, list[Symbol]] = defaultdict(list)
        for sym in symbols:
            if isinstance(sym, _CALLABLE_TYPES) and not sym.is_local:
                by_name[sym.name].append(sym)

        sym_by_id: dict[str, Symbol] = {s.id: s for s in symbols}
        # (enclosing_moniker, name) -> local callable candidates
        local_by_scope: dict[tuple[str, str], list[Symbol]] = defaultdict(list)
        for sym in symbols:
            if isinstance(sym, _CALLABLE_TYPES) and sym.is_local and sym.enclosing_moniker:
                local_by_scope[(sym.enclosing_moniker, sym.name)].append(sym)

        # caller file → {local callee name → import_source}
        imports_of_file: dict[str, dict[str, str]] = defaultdict(dict)
        for rel in relationships:
            if rel.kind != RelationKind.IMPORTS or not rel.target_name:
                continue
            source_path = rel.metadata.get("import_source")
            if not source_path:
                continue
            local_name = rel.metadata.get("alias") or rel.target_name
            caller_file = parse_moniker(rel.source_id).file_path
            imports_of_file[caller_file][local_name] = source_path

        for rel in relationships:
            if rel.kind != RelationKind.CALLS:
                continue
            stats.total_calls += 1
            if rel.is_resolved:
                continue
            stats.unresolved_before += 1

            name = rel.target_name
            if not name:
                continue

            # Highest precedence: scope-gated local resolution (Python lexical scoping).
            local_target = self._resolve_local(rel, name, sym_by_id, local_by_scope)
            if local_target is _AMBIGUOUS_LOCAL:
                # A visible scope has >1 match — the name is bound (ambiguously) in that
                # scope, so it shadows outer/global names.  Leave the call unresolved
                # rather than wrongly falling through to the cross-file by_name pool.
                continue
            if local_target is not None:
                stats.matchable += 1
                rel.target_id = local_target.id
                rel.is_resolved = True
                rel.metadata["resolved_by"] = "local-scope"
                stats.resolved += 1
                stats.by_rule["local-scope"] = stats.by_rule.get("local-scope", 0) + 1
                continue

            candidates = by_name.get(name)
            if not candidates:
                continue
            stats.matchable += 1

            caller_file = parse_moniker(rel.source_id).file_path
            target = self._pick(rel, candidates, caller_file, imports_of_file)
            if target is None:
                continue

            symbol, rule = target
            rel.target_id = symbol.id
            rel.is_resolved = True
            rel.metadata["resolved_by"] = rule
            stats.resolved += 1
            stats.by_rule[rule] = stats.by_rule.get(rule, 0) + 1

        return stats

    # ------------------------------------------------------------------
    # Local-scope helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scope_chain_of(caller_id: str, sym_by_id: dict[str, Symbol]) -> list[str]:
        """Caller's own scope + enclosing scopes, innermost first."""
        chain = [caller_id]
        cur = sym_by_id.get(caller_id)
        seen = {caller_id}
        while cur is not None and cur.enclosing_moniker and cur.enclosing_moniker not in seen:
            chain.append(cur.enclosing_moniker)
            seen.add(cur.enclosing_moniker)
            cur = sym_by_id.get(cur.enclosing_moniker)
        return chain

    def _resolve_local(
        self,
        rel: Relationship,
        name: str,
        sym_by_id: dict[str, Symbol],
        local_by_scope: dict[tuple[str, str], list[Symbol]],
    ) -> "Symbol | object | None":
        """Innermost visible local named `name`, walking the caller's scope chain.

        Returns:
            Symbol         — unambiguous match found in the nearest scope that has the name.
            _AMBIGUOUS_LOCAL — the nearest scope with the name has >1 match; Python lexical
                               scoping means the name is bound here (even ambiguously), so it
                               SHADOWS outer/global — never fall through to by_name.
            None           — no scope on the chain has the name at all.
        """
        for scope in self._scope_chain_of(rel.source_id, sym_by_id):
            hits = self._prefer_call_form(rel, local_by_scope.get((scope, name), []))
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                # Ambiguous at the innermost visible scope that has the name.
                # Stop the walk — the name is shadowed here; do NOT fall through.
                return _AMBIGUOUS_LOCAL
        return None

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def _pick(
        self,
        rel: Relationship,
        candidates: list[Symbol],
        caller_file: str,
        imports_of_file: dict[str, dict[str, str]],
    ) -> tuple[Symbol, str] | None:
        # rule "local" — same file wins, but only when unambiguous there.
        local = [c for c in candidates if c.file_path == caller_file]
        local = self._prefer_call_form(rel, local)
        if len(local) == 1:
            return local[0], "local"

        # rule "import" — alias-aware import map of the caller's file.
        import_source = imports_of_file.get(caller_file, {}).get(rel.target_name or "")
        if import_source:
            imported = [
                c for c in candidates
                if _module_matches(import_source, _module_path_of_file(c.file_path))
            ]
            imported = self._prefer_call_form(rel, imported)
            if len(imported) == 1:
                return imported[0], "import"

        # rule "unique" — one compatible candidate repo-wide.
        unique = self._prefer_call_form(rel, candidates)
        if len(unique) == 1:
            return unique[0], "unique"

        return None

    @staticmethod
    def _prefer_call_form(
        rel: Relationship, candidates: list[Symbol]
    ) -> list[Symbol]:
        """Narrow by call form when that leaves something; never widen."""
        form = rel.metadata.get("call_form")
        if form == "method":
            preferred = [c for c in candidates if isinstance(c, MethodSymbol)]
        elif form == "direct":
            preferred = [c for c in candidates if not isinstance(c, MethodSymbol)]
        else:
            preferred = []
        return preferred if preferred else candidates
