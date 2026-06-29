# Tier-2 Local Symbols — Design Spec

**Date:** 2026-06-29
**Status:** Approved (brainstorm), pre-plan
**Author:** Sutra team (with two elder-agent design reviews)

## Goal

Index **function-local symbols** (nested functions, function-local classes, and
their methods) as first-class **graph nodes** so an agent can trace a complete
call stack — including hops through local helpers — when debugging or tracing a
call. Today these symbols are *skipped*, which (a) leaves the call graph
incomplete through local scopes and (b) was the source of the odin
duplicate-moniker abort.

## Motivating use case (verbatim)

> "I need tier-2 to cover the whole call stack when debugging a function or
> tracing a call; the agent needs detailed code-flow information."

This is a **graph-traversal** need (the agent reaches locals by walking
`get_callees`/`expand_neighbors` from a known entry point), **not** a search
need. That single fact shapes every decision below.

## Background: why this is needed

The moniker is Sutra's primary key: `sutra <lang> <repo> <file_path> <descriptor>`,
descriptors stacking with self-delimiting suffixes (`().` callable, `#` class,
`.` variable, `/` module). It identifies symbols in `embeddings_index.json`,
keys every graph edge, and is the string the MCP server returns to the agent.

Today the adapter (`sutra/core/extractor/adapters/python.py`):
- **skips function-local functions** (`python.py` ~line 612: `if
  _find_enclosing_function(func_node) is not None: continue # nested function`),
- **skips function-local classes** (added 2026-06-29 odin fix, same guard in the
  class loop),

so calls made inside those scopes never enter the graph. The class moniker only
encodes the enclosing-*class* chain, so two same-named function-local classes
collide — which the indexer treats as a fatal `AssertionError` that aborts the
whole repo (`sutra/core/indexer.py` ~line 190).

## Non-goals

- **Not embedded.** Local symbols are graph nodes only; they are excluded from
  the vector/BM25 retrieval index (`chunk_builder._EMBEDDABLE` gate). This keeps
  throwaway stubs out of search results — the elder's primary objection to
  indexing locals. The agent reaches them only by traversal.
- **No lambdas, comprehensions, or local variables.** Only *named* `def`/`class`
  local scopes. Lambdas/comprehensions are indirection-heavy noise; local
  variables are not callable nodes.
- **Not cross-repo / not searchable.** Locals are document-scoped, exactly per
  SCIP's local-symbol constraint ("MUST only be used for entities local to a
  Document, and cannot be accessed from outside the Document").
- **Python only** for this spec. TypeScript/Go adapters are unchanged; the
  identity/model machinery is language-agnostic so they can opt in later.

## Identity model — descriptor (display) + structured field (machine) + flag (safety)

Three coordinated pieces. The descriptor serves the agent reading the graph; the
structured field serves the resolver that must *compare* scopes; the flag is the
explicit, shape-independent safety signal.

### 1. Self-describing descriptor (stacks enclosing-function scope)

Reuse the existing self-delimiting grammar. The enclosing **function** scope
becomes a `name().` segment, exactly like the enclosing **class** scope is a
`name#` segment today:

| Symbol | Descriptor |
|---|---|
| nested function `inner` in `outer` | `outer().inner().` |
| function-local class `NoCause` in `test_no_cause` | `test_no_cause().NoCause#` |
| method `run` of a local class `Helper` in `outer` | `outer().Helper#run().` |

The elder verified this tokenizes unambiguously: SCIP suffix sentinels
(`().`, `#`, `.`, `/`) are self-delimiting, so a left-to-right parse yields
`outer().` → `Helper#` → `run().` with no ambiguity.

### 2. Structured `enclosing_moniker` field (the resolver compares this, never the string)

A new optional field on every symbol holding the **immediate enclosing scope's
moniker** (the containing function or class). This is what the resolver walks to
build a caller's scope chain — it never parses substrings out of the descriptor
(that would be the shape-coupling we are eliminating).

- A nested function's `enclosing_moniker` = the enclosing function's moniker.
- A function-local class's `enclosing_moniker` = the enclosing function's moniker.
- A method of a local class: its existing `enclosing_class_id` already points to
  the local class; `enclosing_moniker` is set to the same value for uniformity.
- Top-level symbols: `enclosing_moniker = None` (scope = module).

### 3. Explicit `is_local: bool` flag (safety)

`True` for any function-local symbol. The resolver checks this flag to exclude
locals from the cross-file candidate pool — safety does **not** depend on
descriptor shape. (Confirmed safe: `heuristic.py:156` and the LSP resolver
already key on `file_path`/location, never on descriptor shape.)

### 4. Residual same-scope collision → deterministic disambiguator

Note this is the **rare** case. The odin trigger (two `NoCause` in *different*
functions) is already disambiguated by scope-encoding alone
(`test_a().NoCause#` ≠ `test_b().NoCause#`). The disambiguator is only for the
truly degenerate case of the *same name redefined in the same scope* (legal but
near-pathological, e.g. `class A` twice in one function).

Scheme — a deterministic counter `N` (0-indexed appearance order in tree order)
applied to the 2nd+ occurrence, placed in SCIP's method-disambiguator parens so
it stays grammar-faithful **and** preserves each segment's trailing sentinel:

| Kind | 1st occurrence | 2nd occurrence |
|---|---|---|
| function | `inner().` | `inner(1).` |
| method | `Helper#run().` | `Helper#run(1).` |
| class | `NoCause#` | `NoCause(1)#` |

This requires a small, SCIP-faithful generalization of `descriptor_kind`
(below): the callable test changes from `endswith("().")` to `endswith(").")`,
so `inner(1).` still classifies as a callable rather than a variable. Class
disambiguation (`NoCause(1)#`) is a documented Sutra extension — SCIP has no
class-disambiguator slot — and is tokenization-safe (still ends in `#`).

Ordinality is owned in **one layer (the adapter)**, in tree order. The indexer
keep-first backstop remains the ultimate net if anything still collides.

## Component changes

### Symbol model — `sutra/core/extractor/base.py`

Add two defaulted fields to `SymbolBase` (defaults keep every existing
construction site and subclass field-ordering valid):

```python
is_local: bool = False
enclosing_moniker: Optional[str] = None   # immediate enclosing scope's moniker; None = module scope
```

### MonikerBuilder — `sutra/core/extractor/moniker.py`

- Generalize the `enclosing` chain in `for_function`/`for_class`/`for_method`
  from "outer class names" to an ordered chain of **scope segments**, where a
  function segment renders `name().` and a class segment renders `name#`.
  (Today `for_class`/`for_method` take `enclosing: tuple[str, ...]` of class
  names only; the chain must now carry segment *kind* so it can render `().`
  vs `#`.)
- **Harden `parse_moniker`:** turn the load-bearing comment at line ~173
  ("file_path and descriptor cannot contain spaces by construction") into a
  build-time assertion in `MonikerBuilder._build` — assert the descriptor
  contains no space, so a future adapter that emits a spaced local name fails
  loudly at construction instead of silently corrupting `split(" ", 4)`.
- `descriptor_kind` (line ~215) handles the *stacked* descriptors unchanged
  (verified: `outer().Helper#run().` → "method", `outer().inner().` →
  "function", `test_no_cause().NoCause#` → "class"). It needs **one** small
  change for the disambiguator: generalize the callable test from
  `endswith("().")` to `endswith(").")` so `inner(1).` classifies as a callable,
  not a variable. Existing monikers (all `().`) are a subset of `).` and
  classify identically — no behavior change for them. **Do not** add an
  "is this local?" check here via descriptor shape — read `is_local`.

### Python adapter — `sutra/core/extractor/adapters/python.py`

- **Stop skipping** function-local functions and function-local classes. Instead,
  build them as symbols with `is_local=True`, the scope-stacked descriptor, and
  `enclosing_moniker` set to the immediate enclosing scope.
- Track the **enclosing scope chain** while walking (extend the existing
  `_enclosing_class_names`/`_find_enclosing_function` helpers into one helper that
  returns the ordered `(kind, name, moniker)` chain up to the module).
- **Emit CONTAINS edges in the new directions** — `function → nested-function`,
  `function → function-local-class` — `is_resolved=True` with `target_id` set
  (mirroring the module→class CONTAINS at `python.py:528`). Methods of a local
  class keep the normal `class → method` CONTAINS.
- Stack only the **raw identifier text** for each scope segment — never a
  synthesized qualname (no `<locals>`/`<lambda>`).
- Own the residual-collision **disambiguator counter** here (tree order).
- Local `CALLS` are already correctly scoped: `_extract_calls` stops at function
  boundaries, so each (now-emitted) nested function gets its own call extraction
  with no double-counting.

### Resolver — `sutra/core/resolver/heuristic.py`

- **Exclude `is_local` symbols from `by_name`** (the cross-file candidate pool).
  This makes the existing `local`/`import`/`unique` rules byte-identical for
  non-local symbols — *no regression* on currently-resolving top-level calls
  (e.g. a top-level `inner` plus a local `inner` in one file no longer makes the
  top-level call ambiguous).
- **Add a new highest-precedence rule "local-scope":** for an unresolved CALLS
  edge from caller `C` to name `N`, walk `C`'s scope chain (via
  `enclosing_moniker`, innermost-first). If a unique local symbol named `N` is
  defined in `C`'s own scope or an enclosing scope on its chain, resolve to the
  innermost one (Python lexical scoping — locals shadow outer/global names).
  Locals are candidates **only** for callers on their scope chain — this both
  fixes the sibling-scope mis-fire and keeps locals from polluting top-level
  resolution.
- The LSP resolver (`lsp_resolver.py`) already maps by location interval and only
  indexes Function/Class symbols; verify it does not regress with locals present
  (it resolves the *ambiguous residue* and should now see fewer ambiguities).

### Exporter / loader

- `json_graph_exporter._symbol_to_dict` serializes `is_local` and
  `enclosing_moniker`; `artifact/loader.py` reads them back into the snapshot
  (for MCP `get_symbol` display and any future MCP-side use). Not required for
  traversal correctness (edges are resolved at index time), but kept for
  completeness and observability.

### Embeddings — `sutra/core/embedder/chunk_builder.py`

- Exclude locals from the embeddable set: change the filter at line ~45 to
  `isinstance(s, _EMBEDDABLE) and not s.is_local`. Locals never become chunks, so
  they never enter vector/BM25. (A local may still appear by *name* in a
  non-local caller's "Calls:" line — correct and harmless.)

### Indexer backstop — `sutra/core/indexer.py`

- Replace the fatal duplicate-moniker `AssertionError` (~line 190) with a
  **deterministic keep-first** policy: first occurrence in the globally-sorted
  walk wins; later duplicates are dropped, **counted**, and logged as a warning.
  This is the safety net for genuinely unexpected collisions; with scope-aware
  monikers + the adapter-owned disambiguator, it should *never fire for locals*.
- Surface the drop count (and existing skipped/failed counts) so a degenerate
  repo is observable, not silent. (Lightweight: extend the result/log, not a new
  subsystem.)

### MCP / traversal — `sutra/core/graph/traversal.py`

- No code change expected, but **verify** the resolved-only edge filter
  (`traversal.py:77`: drops edges where `is_resolved` is false or `target_id`
  missing) passes the new CONTAINS edges — i.e. they must be emitted resolved
  with `target_id`, or `expand_neighbors` silently won't reach the local.
- Optionally surface `is_local` in `sutra_get_symbol` output (nice-to-have).

## Determinism & incremental stability

- **Insert an earlier local in the same function:** descriptor-based ids are
  **stable** (the enclosing function name doesn't change; only a rare residual
  disambiguator counter could shift). This is strictly better than an opaque
  `local N` counter id, which renumbers on insert.
- **Rename the enclosing function:** every contained local's descriptor churns —
  but a structured `enclosing_moniker` would churn identically. A and B are
  **equally** unstable on rename; the win is only on the insert case. (Stated
  precisely to avoid overclaiming.)
- All ids are deterministic given file content + the global file-sort +
  tree-order disambiguator. No randomness, no line numbers.

## Testing strategy (real instances, no mocks)

Adapter (`tests/test_python_adapter.py`):
- odin reproduction: two same-named function-local classes → **no abort**, both
  present as `is_local` graph nodes with distinct monikers.
- nested function emitted with `outer().inner().` descriptor, `is_local=True`,
  `enclosing_moniker` = `outer`'s moniker.
- method of a local class → `outer().Helper#run().`.
- class-in-class still works (`Outer#Config#`) — unchanged regression guard.
- residual same-scope collision (two `class NoCause` in one function) →
  deterministic distinct disambiguated monikers (`NoCause#`, `NoCause(1)#`),
  tree-order stable.
- build-time space assertion fires on a descriptor containing a space.

Moniker (`tests/test_moniker.py`):
- `descriptor_kind` classifies disambiguated descriptors correctly:
  `inner(1).` → "function", `Helper#run(1).` → "method", `NoCause(1)#` →
  "class"; and a real variable `x.` still → "variable" (not caught by the
  generalized `).` callable test).

Resolver (`tests/test_resolver.py`):
- **sibling-scope**: a top-level `inner` and a function-local `inner` in one file
  — a top-level call resolves to the top-level one (no regression); a call from
  inside `outer` resolves to its local `inner` (local-scope rule, shadows).
- a local is **never** resolved as a cross-file `unique` match.

Traversal (`tests/test_mcp_*` / graph tests):
- `expand_neighbors(outer, kinds={"contains"})` **reaches** `outer().inner()`
  (proves the new CONTAINS edges are resolved-with-target and walkable).
- `get_callees(outer)` includes the local `inner` when `outer` calls it.

Indexer (`tests/test_indexer_sink.py` or sibling):
- duplicate-moniker no longer aborts: keep-first keeps the first, drops + counts
  the rest, emits a warning.

Embeddings:
- a repo with locals produces the **same** embeddings_index as before the feature
  for non-locals (locals excluded) — proves no retrieval-surface change.

## Risks / elder-flagged hazards (all addressed above)

1. **Resolver sibling-scope mis-fire** → fixed by excluding locals from `by_name`
   + scope-gated local-scope rule. *(Highest risk.)*
2. **Indexed ≠ reachable** (CONTAINS must be resolved-with-target) → explicit
   requirement + traversal test.
3. **Backstop vs disambiguator ordinality disagreement** → ordinality owned in
   one layer (adapter); indexer backstop is a pure keep-first that never fires
   for locals.
4. **`parse_moniker` space-split corruption** → build-time space assertion +
   raw-identifier-only rule.
5. **Descriptor-shape temptation** in `descriptor_kind` and future code → read
   `is_local`, never infer from shape (documented).

## Decomposition for the implementation plan

Suggested task order (each independently testable, TDD):
1. Symbol model fields (`is_local`, `enclosing_moniker`).
2. MonikerBuilder scope-segment chain + build-time space assert.
3. Adapter: emit nested functions/classes as locals + scope chain + new CONTAINS
   + disambiguator.
4. chunk_builder local exclusion.
5. Resolver: exclude locals from `by_name` + local-scope rule.
6. Exporter/loader serialization of the new fields.
7. Indexer keep-first backstop + drop count.
8. Traversal/MCP verification + reachability test (+ optional `is_local` in
   `get_symbol`).
