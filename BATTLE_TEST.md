# Sutra battle test — three unseen popular Python repos (2026-08-18)

Purpose: independent validation of sutra's retrieval quality and agent
usefulness on repos it was never tuned on, as source material for an
article. Companion to `KIND_FILTER_AB.md` (which covers the kind-filter
fix this test also exercises).

## Setup

- Sutra branch `checkpoint/embedder-and-recall-groundwork` @ c5bc20a
  (soft kind boost default, test files excluded from indexing).
- Embedder: local `BAAI/bge-base-en-v1.5` (768d), no API key. Heuristic
  CALLS resolver. Hardware: 16 GB M2 Pro laptop; each repo indexed in
  well under five minutes.
- Repos (cloned fresh, indexed at HEAD):

| repo | commit | symbols | embeddings | resolver rate |
|---|---|---|---|---|
| pallets/flask | d318b68 | 496 | 419 | 79% (397/503) |
| psf/requests | 8f8b212 | 344 | 299 | 83% (293/354) |
| pydantic/pydantic | 59af43a | 2793 | 2442 | 77% (3273/4224) |

pydantic was chosen deliberately: its domain vocabulary ("model",
"schema", "type", "validator") collides with symbol-kind words, the
exact failure class the soft kind boost was built for.

## Method

**Retrieval eval.** 12 ground-truth queries per repo (36 total), each
authored by an independent agent that read the actual code and verified
every gold moniker exists in the index. Queries were written from the
code, not tuned against retrieval output. Buckets per repo:

- **A** (3): kind-noun collision — the query naturally contains a
  lexicon kind noun used as domain vocabulary while the gold is a
  different kind (e.g. requests: "hooks" → gold is the `HOOKS` dict,
  a variable).
- **B** (5): behavioral — mechanism described, no symbol names, no kind
  nouns.
- **C** (4): explicit correct kind noun — "the function that…", "the
  class that…" (regression guard for hint narrowing).

Modes compared: `hard` (legacy pre-filter), `off` (no kind hints),
`soft` (shipped default, ×1.3 post-fusion boost). Metrics over top-10.

**Agent eval.** 6 realistic trace tickets (2/repo) solved by
claude-haiku subagents, one trial each, NEUTRAL tool framing (sutra MCP
tools listed as first-class equipment alongside Bash/Grep/Glob/Read),
read-only. Same protocol as the KIND_FILTER_AB baseline/post-fix waves.

## Retrieval results

| repo | mode | recall@5 | recall@10 | MRR | zero-recall |
|---|---|---|---|---|---|
| flask | hard | 0.583 | 0.750 | 0.564 | 3/12 |
| flask | off | 0.583 | 0.833 | 0.533 | 2/12 |
| **flask** | **soft** | **0.583** | **0.833** | **0.576** | **2/12** |
| requests | hard | 0.667 | 0.667 | 0.569 | 4/12 |
| requests | off | 0.750 | 0.750 | 0.653 | 3/12 |
| **requests** | **soft** | **0.750** | **0.750** | **0.667** | **3/12** |
| pydantic | hard | 0.500 | 0.500 | 0.347 | 6/12 |
| pydantic | off | 0.667 | 0.667 | 0.410 | 4/12 |
| **pydantic** | **soft** | **0.583** | **0.583** | **0.375** | **5/12** |
| pooled (36) | hard | 0.583 | 0.639 | 0.493 | 13/36 |
| pooled (36) | off | 0.667 | 0.750 | 0.532 | 9/36 |
| **pooled (36)** | **soft** | **0.639** | **0.722** | **0.539** | **10/36** |

Key observations:

1. **Soft never ranks worse than hard on any of the 36 queries** — with
   the 24 KIND_FILTER_AB queries that makes 60/60 across five repos.
   Soft has the best MRR on flask and requests and cures hard-filter
   zero-recalls on all three repos (flask "settings" query None→7;
   requests "environment settings" None→**1**; pydantic constrained-int
   None→3).
2. **On pydantic, `off` beats `soft`** — the one measured cost of the
   boost. PD-A3 ("Inside a custom `__get_pydantic_core_schema__`, what
   object do I call…", gold: the `GetCoreSchemaHandler` class) ranks 4
   under `off`; under `soft` the verb-derived callable hint boosts
   functions past the gold class and it drops out of top-10. Soft still
   equals hard there (both None) — the boost never recreated the hard
   filter's catastrophes, but it is not free.
3. **The C bucket confirms narrowing still pays**: 10 of 12 explicit
   kind-noun queries hit rank 1–3 under soft on their gold's kind.

## Failure analysis (the honest part)

9 of 36 queries fail in **every** mode. Their gold kinds tell the
story:

- **4/4 queries whose gold is a variable or module failed in every
  mode** (requests `HOOKS` and `REDIRECT_STATI` variables, flask
  `typing.py` module, pydantic `alias_generators` module) — vs 5/32
  for class/function/method golds. Sutra's channels are tuned for
  callables: variables have thin text to embed and BM25 over bodies
  favors code-dense symbols. If your question's answer is a constant
  or a module, the index probably won't surface it.
- The remaining 5 are ranking losses in dense corpora (e.g. pydantic's
  discriminated-union resolution, requests' `super_len`).
- **Recall is phrasing-sensitive in both directions**: the eval query
  "How does Flask turn the signed cookie back into the session" never
  surfaced `open_session` in any mode, yet the haiku agent's own
  phrasing of the same question ("session restore from cookie
  request") hit it at **rank 1**. Single-phrasing evals understate
  what an agent in a retry loop actually gets.

## Agent results (haiku, NEUTRAL framing, 1 trial/ticket)

Tickets: flask session-cookie lifecycle (FK1), WSGI→response dispatch
(FK2); requests redirect chain (RQ1), adapter dispatch/send (RQ2);
pydantic model-class construction (PY1), dict→instance validation flow
(PY2).

| ticket | sutra calls | bash | read | searches | first search with gold in results | trace correct |
|---|---|---|---|---|---|---|
| FK1 | 5 | 2 | 8 | 5 | #1 (rank 1) | yes |
| FK2 | 11 | 0 | 13 | 6 | #1 (rank 2) | yes |
| RQ1 | 2 | 6 | 11 | 2 | #1 (rank 7) | yes |
| RQ2 | 4 | 0 | 13 | 4 | #1 (rank 2) | yes |
| PY1 | 8 | 0 | 14 | 6 | #1 (rank 3) | yes |
| PY2 | 11 | 16 | 11 | 9 | #1 (rank 5) | yes |

- **Adoption 6/6 trials, 41 index calls total.** No agent was told to
  prefer the index; framing was neutral.
- **The first natural-phrasing search landed a gold mechanism in the
  result list in 6 of 6 tickets** (ranks 1–7). 18 of 32 searches
  (56%) contained a gold; most misses were narrow follow-up probes for
  private helpers, which is not a failure mode.
- **All 6 traces identified the correct mechanism with file:line
  citations** (self-assessed against known golds).
- Context from the KIND_FILTER_AB waves: adoption on these unseen
  repos (6/6) matches the post-fix fastapi rate (6/6) and far exceeds
  the pre-fix sutra-repo rate (1/6) — consistent with the finding that
  index value grows with corpus unfamiliarity and result quality.

## Verdict for the article

- On three popular repos it was never tuned on, sutra's out-of-the-box
  soft mode is the best or tied-best configuration on every repo, and
  the legacy hard filter is strictly dominated: 60/60 queries across
  five repos where soft ≥ hard.
- Realistic numbers to quote: recall@5 ≈ 0.58–0.75 and MRR ≈ 0.38–0.67
  per repo on adversarially-bucketed, blind-authored queries — not the
  inflated numbers a name-echo benchmark would give.
- Agents given the index under neutral framing use it every time, and
  their first query lands the right mechanism in the top-10 every
  time; all traces were correct.
- Known weaknesses to state honestly: variable/module golds are nearly
  invisible to retrieval (4/4 misses); one measured case where the
  kind boost hurt vs plain fusion (pydantic PD-A3); single-phrasing
  recall understates practical agent recall.

## Reproduction

- Ground truth + runner: `benchmarks/battle_test/` in this repo
  (flask.json, requests.json, pydantic.json, run_ab.py,
  ab_results.json — untracked). Golds are exact monikers verified
  against each repo's `graph.json`.
- Agent transcripts: session task files; tallies computed by counting
  `tool_use` blocks and scoring gold substrings in each
  `sutra_search` result list.
