# Sutra — Phase 2 Roadmap & Retrieval Architecture

> **Last updated:** 2026-06-07 — revised after a multi-round architecture review (see [§ What changed since the first draft](#what-changed-since-the-first-draft)). The storage layer, product framing, and priority order all moved. If you read the original version of this file, re-read it — the deltas are load-bearing.
>
> **Companion documents:**
> - `DESIGN.md` — original locked architecture, data model, moniker format, Phase 1 scope. **Superseded on the storage layer:** DESIGN.md describes Apache AGE for graph storage; AGE was removed in Phase 2 P0 (see PROGRESS.md). Treat DESIGN.md's data model / moniker / embedding-strategy sections as authoritative, but read its AGE / single-Postgres sections as historical.
> - `project_memory.md` — locked design decisions. Same caveat: the AGE references are historical.
> - `PROGRESS.md` — implementation log. Phase 1 (priorities 1–11) + Phase 2 P0 (drop AGE) + P0.5 (nested-class moniker fix) are recorded there. Every priority below appends to that log as it ships.

---

## TL;DR

Phase 1 indexing is done end-to-end. The first Phase 2 work — **dropping Apache AGE for plain Postgres + an in-memory consumer architecture (P0)** and a **nested-class moniker fix (P0.5)** — has shipped. The next substantive workstream is **retrieval quality**, gated behind a measurement harness.

Two framing shifts since the original draft:

1. **The MCP server is THE product, not a reference consumer.** Nobody on a small team writes a second consumer of the artifact format, so MCP is what gets optimized for. The artifact (`graph.json` + `embeddings.npy`) stays clean because we control both producer and consumer — but we stop pretending the format is the surface.
2. **The consumer runs entirely in-memory with zero infrastructure.** A friend runs MCP with `pip install` + a directory of artifacts — no Postgres, no AGE, no DuckDB. MCP loads the artifact into `rustworkx` (graph) + NumPy (vectors, brute-force cosine) at boot. Postgres lives only on the indexer side, for incremental-update bookkeeping and producing the artifact.

The retrieval thesis is unchanged in spirit: embeddings are demoted from THE signal to one channel in an RRF-fused, multi-channel pipeline. **Behavioral fingerprinting is no longer the centerpiece** — it's a gated, speculative bet to attempt only if measurement shows the cheaper channels (BM25 + kind filter + reranker) don't close the gap. See [§ Retrieval Thesis](#retrieval-thesis).

**Start with Priority 12 (eval harness).** Without measurement, every retrieval change after it is a vibe check.

---

## What changed since the first draft

The original SUTRA_PHASE2.md proposed an embedding-centric retrieval rebuild with behavioral fingerprinting as the differentiating feature, running on the Phase 1 AGE + pgvector backend, with MCP as a "reference consumer." A multi-round review (elder + backend-tech-lead consultations) and direct measurement against a real indexed repo (`booth`) changed the following:

| Area | Original | Now | Why |
|---|---|---|---|
| **Graph storage** | Apache AGE (openCypher) | Plain Postgres SQL tables + recursive CTEs (indexer side); `rustworkx` in-memory (consumer side) | Phase 2 graph queries are shallow (one-hop, callers/callees). AGE's thin ecosystem + Postgres-version coupling + Cypher param-injection workarounds weren't paying for themselves. **Shipped as P0.** |
| **Consumer runtime** | MCP queries Postgres + pgvector | MCP loads `graph.json` + `embeddings.npy` in-memory (rustworkx + NumPy), zero infra | A friend must run MCP with no database. In-memory brute-force cosine over ~150k vectors is <100ms — faster than an indexed SQL round-trip at our scale. |
| **Product framing** | Artifact is the product; MCP is a reference consumer | MCP is the product; artifact serves MCP | No second consumer will exist for an internal team tool. Optimize for the real consumer; keep the artifact clean by controlling both ends. |
| **Behavioral fingerprinting** | The differentiator; built early (P13) | A gated bet; built only if measurement demands it (after the MEASURE gate) | The outgoing-call signal it provides is *already in the embedding chunk* (`chunk_builder.py` emits a `Calls:` line). Its only novel contribution is side-effect categories — speculative and lexicon-brittle. Rerankers usually dominate the lift; test that first. |
| **Multi-view embeddings** | Mid-plan schema migration (P14) | Gated; only if fingerprinting proves out | 4× storage + a breaking migration + incremental-updater rework, justified by an unvalidated thesis. Defer behind the MEASURE gate. |
| **DuckDB (considered)** | — | Rejected | Its vector (VSS/HNSW) extension is still flagged experimental in 2026. Risk-averse: in-memory NumPy is boring and deterministic. |
| **Embedder selection** | — | Out of scope; embedder is fungible | The embedder slot is already plug-and-play. Default to local sentence-transformers for hermetic eval. We are not optimizing model choice in Phase 2. |
| **LLM in indexing** | Forbidden | Forbidden (re-confirmed, locked) | Re-examined explicitly given the small scale; the principle holds. No LLM enrichment in the indexing pipeline, ever. |
| **Priority order** | 12 → 21 sequential | Reordered: eval → cheap channels → resolver → fusion → MEASURE gate → MCP → (gated speculative work) | Cheap, high-certainty interventions first; the expensive speculative ones behind a measurement gate. |
| **Resolver dependency** | P20 (LSP) late & independent | A heuristic resolver (P20-lite) must precede graph expansion (P17) | Measured: **100% of `CALLS` edges are unresolved** on a real Python repo. One-hop graph expansion over zero usable edges is dead weight. |

---

## What exists today

**Phase 1 (priorities 1–11), shipped:** dataclasses, JSON exporter, end-to-end indexer, Python/TypeScript/Go adapters, embedder (Fixture/OpenAI/Local + factory), pgvector store, incremental update pipeline. See PROGRESS.md.

**Phase 2 foundation, shipped:**
- **P0 — Drop AGE → plain Postgres SQL backend + in-memory MCP architecture.** `GraphWriter` / `IncrementalReader` ABCs; `SqlGraphWriter` / `SqlIncrementalReader` / `SqlIndexStateStore`; three migration tables (`sutra_repositories`, `sutra_symbols`, `sutra_relationships`). AGE files deleted; Docker stripped of AGE + flex/bison. pgvector and the JSON+npy artifact format are unchanged.
- **P0.5 — Nested-class moniker uniqueness.** `for_class` / `for_method` gained a backward-compatible `enclosing` param so same-named nested classes (e.g. Pydantic `class Config`) get distinct, descriptor-stacked monikers.

**Outside the formal priority log but in the repo:** a React + FastAPI UI under `frontend/` (uvicorn on `127.0.0.1:8000`; SSE log streaming; SQLite job history; FIFO queue).

---

## Storage & runtime architecture (post-P0)

```
INDEXER SIDE (write path — Postgres)
    Tree-sitter adapters → Symbols + Relationships (+ pgvector embeddings)
        │
        ├── sutra_symbols / sutra_relationships / sutra_repositories
        │     (plain SQL tables; recursive CTEs when traversal is needed;
        │      durable state for the incremental updater's per-file diff)
        ├── pgvector (HNSW) — embeddings, indexer-side only
        └── JSON exporter → graph.json + embeddings.npy   ← THE ARTIFACT

ARTIFACT (the handoff — a directory of files, shared manually for now)
    graph.json          symbols + relationships + metadata (versioned schema)
    embeddings.npy      float32 matrix, row N ↔ embeddings_index.json[N]
    embeddings_index.json
    .ready              sentinel written last (atomic-export contract)

CONSUMER SIDE (read path — MCP, in-memory, zero infra)
    Boot: load graph.json → rustworkx DiGraph
          load embeddings.npy → NumPy float32 matrix
          build in-memory indices (moniker→node, moniker→row, BM25 corpus)
          validate: embedding_model_id + dims match; schema_version compatible
    Serve: retrieval pipeline runs entirely in-process (no DB at query time)
```

Key consequences for Phase 2:
- **The retrieval pipeline runs in-memory, built from the artifact at MCP boot.** The vector channel is NumPy brute-force cosine (not pgvector at query time). The BM25 channel is an in-memory corpus (`rank-bm25` / `bm25s`) built at boot from identifiers + docstrings (not a Postgres `tsvector`). Graph expansion is `rustworkx` (not AGE / SQL CTE).
- **The eval harness (P12) and MCP share that in-memory stack.** The baseline is single-channel in-memory cosine over `embeddings.npy` — hermetic, no DB required.
- **Postgres remains indexer-side only.** It backs the incremental updater's body-hash diff. (Open question, not for now: could incremental diffing read the previous artifact instead, dropping Postgres entirely? Rejected for Phase 2 — would mean rewriting the working incremental updater.)

---

## Retrieval Thesis

### The failure mode

Behavioral queries fail. Concrete examples that motivated the rethink:

- *"a global function that attaches token"* → returns the `Token` interface or the cookie-storage code, not the function that attaches tokens to outgoing requests.
- *"which function saves the listing in db"* → returns the `Listing` request model, not the function that writes to the DB.

This is **kind/type pollution**: cosine similarity reliably picks the lexically densest match. Type definitions, request/response models, and interfaces share vocabulary with the implementing function and drown it out in the top results. Embeddings tell you which symbol *talks like* the thing — not which symbol *does* the thing.

### The reframe

**Embeddings are demoted, not removed.** They still earn their keep on genuine paraphrase queries (*"authentication"* → *"auth flow"*). But they stop being THE signal and become one input to RRF fusion alongside lexical (BM25) and exact (moniker) channels.

The recommended retrieval pipeline (all stages run in-memory in the MCP process):

```
Query
  │
  ▼
[1] Query analyzer    extracts: kind filter, behavioral verbs,
  │                   entity nouns, query embedding                       (P16)
  ▼
[2] Multi-channel retrieve (top ~50 candidates each)
  │   ├── Vector   — NumPy brute-force cosine over embeddings.npy
  │   ├── BM25     — in-memory corpus over identifiers + docstrings       (P15)
  │   └── Moniker  — exact + pattern lookup over the in-memory index
  ▼
[3] Kind filter       drop kinds that contradict query intent            (P16)
  ▼
[4] RRF fusion        across channels, k=60                              (P17)
  ▼
[5] Graph expansion   one hop via rustworkx in-memory graph              (P17)
  │                   (CALLS / EXTENDS / IMPLEMENTS / REFERENCES)
  │                   — requires a resolved CALLS graph: see P20-lite
  ▼
[6] Cross-encoder     BGE-reranker-v2-m3, opt-in per query               (P18)
  ▼
Final top-K
```

### Behavioral fingerprinting — demoted to a gated bet

The original plan made behavioral fingerprinting the centerpiece: statically extract each function's outgoing call signatures + side-effect categories (`db_write`, `network`, `filesystem`, …) from per-language lexicons, embed as a dedicated view, and let type definitions (empty fingerprints) drop out of behavioral queries.

Why it was demoted:

1. **Its main signal already exists.** `chunk_builder.py` already emits a `Calls:` line listing outgoing call names into the embedding text. The "outgoing calls" half of the fingerprint is already in the vector. The only *novel* contribution is the side-effect categorization.
2. **Side-effect lexicons are lossy and brittle.** A function that calls `_internal_helper()` which does the DB write won't be tagged. Real service-layered codebases wrap everything. The honest hit rate is the empirical question — and it's unproven; no production code-retrieval system ships this.
3. **Rerankers are unreasonably effective and cheap to bolt on.** The likely outcome is that BM25 + kind filter + cross-encoder reranker close most of the gap, leaving fingerprinting a marginal contributor that doubled storage and write cost.

So: build the cheap, high-certainty channels first, measure at the MEASURE gate, and attempt fingerprinting (P13) + multi-view embeddings (P14) **only if the gap remains**.

**If/when fingerprinting is built, extract it as a first-class symbol attribute, not just an embedded view.** As a queryable symbol property it's useful regardless of any retrieval thesis; conflating it with a vector view would leave a half-used schema field after a retrieval pivot.

### Out-of-scope retrieval ideas

- **No LLM in the retrieval pipeline at indexing time** — same constraint as indexing. The reranker is allowed because it runs at query time and is opt-in per query.
- **No agentic / iterative retrieval in v1.** RANGER-style MCTS over the graph is overscoped. RRF + one-hop expansion + reranker is the v1 stack.
- **No PageRank / centrality scoring** in v1.
- **No embedder-model bake-off.** The embedder is fungible (plug-and-play already); default to local sentence-transformers for hermetic eval. Choosing a code-specialized model is a future knob, not a Phase 2 priority.

---

## Vision

### What Sutra is

Sutra produces a **portable artifact** (`graph.json` + `embeddings.npy`) that a **shared MCP server** loads to power LLM-assisted querying over a team's repos. **The MCP server is the product.** The artifact format is the clean boundary between indexer and consumer — both of which we own.

### In scope

- The in-memory retrieval pipeline (priorities 12–18 below)
- The MCP server as the product (priority 19) — in-memory loader, zero consumer-side infra
- Heuristic + LSP cross-file `CALLS` resolution (P20-lite, then P20-full pyright) — upgrades unresolved relationships in place
- Markdown generator (priority 21, lowest — build only on concrete demand)
- More languages once the initial three are battle-tested

### Out of scope, full stop

- Not an IDE plugin, editor, or agent
- Not a SaaS — no `api.sutra.dev`, ever
- Not a paid product
- Not a Sourcegraph replacement
- Not a wiki generator, dead-code detector, or refactoring tool
- Not multi-modal — code only
- **No LLM enrichment in the indexing pipeline** — locked. Re-examined at this scale; the principle holds.
- **No third-party / package indexing in v1** — first-party repo code only. (This is why the moniker package block stays deferred — see locked decisions.)

---

## Locked decisions (Phase 2)

These were settled across the architecture review and should not be re-litigated without explicit discussion:

1. **No LLM in indexing. Ever.** Pure tree-sitter analysis.
2. **Embedder is fungible.** Likely local sentence-transformers (384 dims) by default. Not a tuning target in Phase 2.
3. **MCP server is THE product**, not a reference consumer.
4. **First-party-only indexing** for the foreseeable future → moniker package block deferred (current format `sutra <lang> <repo> <file_path> <descriptor>` stays).
5. **Graph storage = plain Postgres SQL (indexer) + rustworkx in-memory (consumer).** AGE removed. **Shipped (P0).**
6. **Zero infra for friends.** MCP loads JSON+npy in-memory. No DuckDB, no consumer-side Postgres.
7. **No DuckDB** — its vector index extension is still experimental in 2026.
8. **Manual artifact sharing** (sneakernet) for now — no distribution channel built yet.

---

## Design-pattern decisions (carry these into the priorities)

Settled with the backend-tech-lead review. Recorded so future sessions don't re-decide:

**ABCs that exist or are planned:**
- `GraphWriter` (ABC) — shipped (`SqlGraphWriter`).
- `IncrementalReader` (ABC) — shipped (`SqlIncrementalReader`). Narrow indexer-side API; **deliberately split** from the consumer-side traversal interface.
- `GraphTraversal` (ABC, MCP-side) — to build in P19. `rustworkx`-backed. Carries the k-hop / callers / callees verbs. Split from `IncrementalReader` because the cost models differ sharply (in-memory pointer chase vs SQL IO).
- `VectorStore` (ABC) — generalize the existing `PGVectorStore` (indexer-side write) and add `InMemoryVectorStore` (MCP-side NumPy brute-force). Add a `filter_monikers` arg so the kind filter can pre-restrict the candidate set.
- `Channel` (ABC) — introduce at **P15** (when the second channel lands), not at P12.

**Deliberately NOT ABCs:**
- `SqlIndexStateStore` — concrete; one impl + a test fake is enough.
- `Reranker` — a plain function `rerank(query, candidates, model) → candidates`. Promote to ABC only if a second reranker appears.
- `Retriever` — concrete `BaselineRetriever` at P12; promote to ABC only when a second top-level retriever exists (≈P14). **But introduce the `SearchResult` dataclass with a `provenance` field at P12** so the eval contract is right from day one and doesn't churn through P15–P18.

**Load-bearing non-ABC components to build:**
- `SearchResult` dataclass (`moniker`, `score`, `provenance: dict[str, float]`) — frozen, slots.
- `AtomicArtifactWriter` — bundle-commit utility: write `graph.json.tmp` + `embeddings.npy.tmp`, fsync, atomic-rename **as a unit**, write `.ready` sentinel last, retain one `.prev` generation. Both files commit together because MCP loads them together.
- MCP loader split into three: `ArtifactLoader` (pure: path → snapshot), `SnapshotRegistry` (ref-counted atomic swap so in-flight queries finish on the old snapshot), `ArtifactWatcher` (watches the `.ready` sentinel — never the data files — debounced). Lock-free reads; lock only the swap.

**Anti-patterns to avoid:** factory pattern for single-impl ABCs; Pydantic on hot-path dataclasses (use `@dataclass(frozen=True, slots=True)`); asyncio for CPU-bound rustworkx/NumPy tool bodies; SQLAlchemy ORM for three tables; a "GraphFacade" god-object.

**Artifact-integrity invariants (bake in as MCP boot-time checks):**
- `embeddings.npy.shape[0] == len(graph.json.symbols)` — primary torn-artifact check.
- `embedding_model_id` + `dims` recorded in artifact metadata; MCP refuses to mix a query embedder that doesn't match.
- `schema_version` (and a `moniker_format_version`) in metadata; MCP refuses incompatible artifacts at boot rather than silently mis-loading.

---

## Phase 2 Priorities

Numbering continues from Phase 1. Foundation work (P0, P0.5) is already in PROGRESS.md. Below are listed in **execution order**, with original-doc priority numbers preserved for traceability. Each ships as a self-contained PR, logs to PROGRESS.md, and gates behind a passing test suite (and, from P12 onward, the eval harness).

---

### Priority 12 — Retrieval Eval Harness  ·  NEXT

**Status:** NOT STARTED — start here.

**Why first:** every retrieval change in the priorities after it is unfalsifiable without it.

**What to build**
- `sutra/core/retrieval/eval/` — `dataset.py` (load query → expected-monikers cases from YAML/JSON), `metrics.py` (`recall_at_k`, `mrr`, per-query breakdown), `harness.py` (runs queries through any `Retriever`, reports aggregate + per-query, supports A/B between two retrievers).
- `SearchResult` dataclass (`moniker`, `score`, `provenance`) — introduce now (see design decisions).
- `BaselineRetriever` — **single-channel in-memory cosine over `embeddings.npy`** (not pgvector — the consumer stack is in-memory). Concrete class; `Retriever` ABC deferred.
- `tests/eval/datasets/` — checked-in datasets across ≥3 indexed reference repos (Python web backend, TS frontend, Go service; pin commit SHAs). ≥30 hand-written queries across: behavioral, entity, paraphrase, exact-name, known-failures.
- `tests/eval/baselines/` — snapshot of baseline metrics so every later priority can A/B against it.

**Acceptance**
- `pytest tests/test_eval_harness.py` runs the harness against `BaselineRetriever` and asserts metrics within ±5% of the snapshot; drift names the regressed queries.

**Notes**
- Hermetic: use `FixtureEmbedder` or a pinned local model. Have a teammate write half the queries blind to the known failures, to avoid over-fitting the architecture to failures we already know.
- Dataset format: `{query, expected: [moniker,...], must_include: [moniker,...] | None, kind_filter: str | None}`.

---

### Priority 15 — BM25 Channel

**Status:** NOT STARTED.

**What to build**
- `sutra/core/retrieval/channels/base.py` — `Channel` ABC (introduce here, with the second channel).
- `sutra/core/retrieval/channels/bm25_channel.py` — in-memory BM25 (`rank-bm25` / `bm25s`) over a weighted concatenation of `name` (A), `qualified_name` (A), `signature` (B), `docstring` (C). **Built at MCP boot from the artifact** — no Postgres `tsvector` (the consumer never touches a DB). Cache the derived index to `~/.cache/sutra/` keyed by artifact content hash if boot time ever matters.

**Acceptance**
- Eval harness with a BM25-only retriever: behavioral queries still poor (expected), exact-name queries near-perfect. Document the deltas.

---

### Priority 16 — Query Analyzer + Kind Filter

**Status:** NOT STARTED.

**What to build**
- `sutra/core/retrieval/query_analyzer.py` — `parse_query(s) → ParsedQuery` returning: `embedding`, `kind_hint: set[Kind] | None`, `verbs: set[str]`, `entities: list[str]`. Pure-Python keyword/regex; checked-in YAML lexicons. **One analyzer**, shared by all channels — don't let each channel re-parse the query.
- Kind filter applied to candidates before fusion: drop candidates whose `kind` contradicts the hint (conservative — only drop when confident; uses `VectorStore.filter_monikers` to pre-restrict where possible).

**Acceptance**
- Eval harness with kind filter on baseline results: behavioral queries return request models / interfaces less often. Quantify the lift.

---

### Priority 18 — Cross-Encoder Reranker

**Status:** NOT STARTED.

**What to build**
- `sutra/core/retrieval/reranker.py` — a function `rerank(query, candidates, model) → candidates` wrapping `BAAI/bge-reranker-v2-m3`. Reads chunk text per candidate (recomputed via `chunk_builder` at query time, cached per query).
- Add `sentence-transformers` / `FlagEmbedding` to an optional requirements file — don't force-install.
- Wired behind a `rerank: bool = False` flag (opt-in per query).

**Acceptance**
- Eval harness with reranker on top of the fused pipeline: clear recall@10 + MRR lift across categories. **Log the latency budget** (CPU cross-encoder over ~50 candidates is real wall-clock — single-digit QPS tolerates it, but record it).

---

### Priority 20-lite — Heuristic In-Repo Resolver

**Status:** NOT STARTED. **Must precede P17.**

**Why before P17:** measured against a real Python repo, **100% of `CALLS` edges are unresolved today**. One-hop graph expansion (P17) over an unresolved graph adds nothing. A heuristic resolver gets most intra-repo calls resolved cheaply, before LSP.

**What to build**
- A resolver that walks unresolved `CALLS` relationships and matches the callee short-name against the in-graph `{qualified_name → moniker}` map, disambiguated by the file's resolved imports. Flips `is_resolved=False → True` and sets `target_id` in place.
- Note: the `Resolver` seam described in DESIGN.md does **not** yet exist — this priority builds it (an ABC + this first impl), it is not "purely additive" as the original doc claimed.

**Acceptance**
- ≥60% of previously-unresolved intra-repo `CALLS` resolved on the booth fixture. If <30% short-name matches exist, skip straight to P20-full (LSP).

---

### Priority 17 — RRF Fusion + One-Hop Graph Expansion

**Status:** NOT STARTED. Depends on P15, P16, P20-lite.

**What to build**
- `sutra/core/retrieval/fusion.py` — Reciprocal Rank Fusion across channels. `score(d) = Σ 1/(k + rank_i(d))`, `k=60`. Fusion assigns ranks itself (channels return sorted `list[SearchResult]`; the fusion step controls rank assignment so the contract can't be silently violated).
- `sutra/core/retrieval/expander.py` — one-hop neighbors via the `rustworkx` in-memory graph (`CALLS` / `EXTENDS` / `IMPLEMENTS` / `REFERENCES`), added with a configurable score discount. **Use plain rustworkx traversals (one hop = one neighbor lookup), not recursive CTEs.**
- `sutra/core/retrieval/pipeline.py` — orchestrates analyzer → channels → kind filter → fusion → expansion. No reranker here (that's P18, composed on top).

**Acceptance**
- Eval harness: full pipeline (no reranker) vs baseline. Meaningful recall@10 lift on behavioral queries; no regression on paraphrase. Expansion must add ≥5% recall@10 over fused-without-expansion or it gets dropped.

---

### MEASURE GATE  ·  decision point

**Status:** NOT A PRIORITY — a checkpoint.

Run the full eval. If behavioral recall@10 is acceptable (target >0.6), **ship the MCP server (P19) and consider Phase 2 retrieval done.** The speculative work below (P13, P14) is attempted **only if this gate fails.** Don't let the roadmap drag past acceptable retrieval — diminishing returns are real and this is a solo/small-team tool.

---

### Priority 19 — MCP Server  ·  the product

**Status:** NOT STARTED. Can begin in parallel once P15/P16/P18 exist; ships after the MEASURE gate.

**What to build**
- `sutra/mcp/` — FastMCP mounted into the existing FastAPI process (confirm against current MCP SDK guidance).
- **In-memory loader** (the load-bearing component): `ArtifactLoader` + `SnapshotRegistry` + `ArtifactWatcher` (see design decisions). Boot-time validation (shape, embedding-model identity, schema version). Hot-reload on `.ready` sentinel change with ref-counted snapshot swap.
- `GraphTraversal` (rustworkx) + `InMemoryVectorStore` (NumPy) built from the snapshot.
- Tools: `sutra.search(query, repos?, top_k=10, rerank=True)`, `sutra.get_symbol(moniker)`, `sutra.expand_neighbors(moniker, depth=1, kinds?)`, `sutra.list_repos()`, `sutra.get_callers(moniker)` / `sutra.get_callees(moniker)`.
- Auth: shared bearer token via env (`SUTRA_MCP_TOKEN`), checked per request.
- Logging: every tool call → SQLite audit log (reuse the UI's job-history SQLite).

**Acceptance**
- An MCP-aware client (Claude Desktop, Cursor, etc.) connects, lists tools, runs a search, gets useful results — **with no Postgres running**, against a directory of artifacts.
- In-process integration test booting the app and exercising every tool, plus loader tests (torn-artifact rejection, version-mismatch rejection, hot-reload swap).

---

### Priority 13 — Behavioral Fingerprinting Extractor  ·  GATED

**Status:** NOT STARTED. Attempt **only if the MEASURE gate fails** on behavioral queries.

**What to build (if pursued)**
- `sutra/core/extractor/fingerprint/` — per-language YAML lexicons mapping import paths / call patterns to side-effect categories; `extractor.py` emitting `BehavioralFingerprint(outgoing_calls, side_effects)`; `category.py` enum.
- **Extract as a first-class optional field on `FunctionSymbol` / `MethodSymbol`** (default `None`), serialized in `graph.json`. Behind a feature flag. The "fingerprint view" (if P14 happens) is a separate consumer of this attribute — don't conflate them.

**Acceptance**
- Hand-crafted fixtures: `requests.post` → `{network}`; `session.add` + `session.commit` → `{db_write}`; pure helper → empty fingerprint; dataclass → `None`. Same for TS/Go. Eval harness shows fingerprinting moves behavioral recall **beyond** what reranker + kind filter already achieved — otherwise stop.

---

### Priority 14 — Multi-View Embeddings  ·  GATED

**Status:** NOT STARTED. Attempt **only if P13 proves out** that view-separation helps.

**What to build (if pursued)**
- Up to four chunks per embeddable symbol (signature / body / docstring / fingerprint); empty views skipped. Storage keyed by `(moniker, view)`. Indexer + incremental updater write all relevant views.
- **Incremental-updater blast radius is real:** a body change re-embeds up to 4 views; needs per-view dirty-tracking. Budget this as part of the priority, not a footnote.
- Add the regression test for the historical dim-mismatch bug (`vector(384)` vs `vector(1536)`) while in this area.

**Acceptance**
- Existing tests pass after migration. View-filtered retrieval works. Eval shows multi-view beats reranker-on-rich-single-chunk by enough to justify the 4× cost + migration. If not, don't ship it.

---

### Priority 20-full — LSP Resolver (pyright)  ·  GATED / deferred

**Status:** NOT STARTED. Build only when a real consumer needs >80% `CALLS` resolution beyond what P20-lite delivers.

**What to build**
- `sutra/core/resolver/lsp_resolver.py` implementing the resolver ABC (created in P20-lite). Run pyright as a subprocess, parse output, patch unresolved `CALLS` `target_id` + flip `is_resolved`. `--resolver lsp` CLI flag (default stays the heuristic). Warm one process per language per run; cache within a run.

**Acceptance**
- E2E on a Python fixture with cross-file calls: heuristic leaves some unresolved; `--resolver lsp` resolves them. Resolved relationships appear in the SQL store with `is_resolved=true`.

---

### Priority 21 — Markdown Generator  ·  lowest

**Status:** NOT STARTED. Build only if a concrete consumer asks. MCP makes this largely redundant.

**What to build**
- `sutra/core/output/markdown_generator.py` implementing the `OutputGenerator` interface. Walk the graph in dependency order, emit a dense per-repo doc. Configurable: include/exclude private symbols, bodies, max depth.

**Acceptance**
- Byte-stable output across runs for `sample_python_repo`.

---

## Implementation Conventions for Phase 2

These match the Phase 1 patterns that are working — keep doing them:

- **Per-priority test gate.** Don't merge until tests pass green and (from P12) the eval harness shows the expected delta.
- **Append to PROGRESS.md** at the end of each priority. Same format: Status / What was built / Files created / Files modified / Implementation decisions. Future sessions depend on this log.
- **Hermetic fixtures over real-world smoke tests.** Real-repo smoke tests are gated behind `SUTRA_PG_URL` (indexer) and skip cleanly without it; the in-memory MCP/retrieval tests need no DB at all.
- **No silent feature drift.** New behavior goes behind a flag until the eval harness confirms no regression; default flips on only after.
- **Incremental updater compatibility.** Every priority that touches indexer storage must update `IncrementalUpdater` in the same PR; its test suite is the canary.
- **Plug-and-play seams stay clean.** New modules go behind the ABCs named in the design-pattern decisions. No direct imports from `pipeline.py` into specific channel/reranker implementations.
- **No baggage.** When a real-world repo surfaces a latent bug (as booth did in P0.5), fix the root cause with a failing test first, scope-check the other adapters, and log it — don't paper over it.

---

## Recommended starting point

**Build Priority 12 first.** Concrete first-session goal:

1. Pick three reference repos and pin their commit SHAs.
2. Index them locally with the existing pipeline (FixtureEmbedder or a pinned local model — eval must be hermetic).
3. Hand-write the first ~10 queries with expected monikers, focused on the known failures (*"function that attaches token"*, *"which function saves the listing in db"*).
4. Implement `SearchResult`, `BaselineRetriever` (in-memory single-channel cosine), the harness, and metrics.
5. Snapshot the baseline. It'll be bad on behavioral queries — that's the point. Now every priority has a number to beat.

Then proceed in execution order: P15 → P16 → P18 → P20-lite → P17 → MEASURE gate. Ship P19 (MCP) once retrieval clears the gate. Treat P13/P14/P20-full as gated work that may never be needed.
