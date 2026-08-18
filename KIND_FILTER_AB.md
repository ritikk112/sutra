# Kind-filter A/B — hard pre-filter vs soft boost (2026-08-18)

**Question.** `allowed_monikers` (pipeline.py) turns an *inferred* kind hint into a
hard pre-filter that deletes non-matching symbols from every channel's candidate
pool before ranking. A wrong hint therefore zeroes recall (unbounded downside).
Commit `1b4638b` validated it (+0.089 MRR, 0 regressions) on a 32-query eval over
three CRUD repos — where kind nouns ("model", "function") coincide with actual
kinds. On corpora where those nouns are domain vocabulary (ML infra: "model";
web frameworks: "response model", "schema", "types") the hint inverts and erases
the gold. This log measures the failure and the fix on two such repos.

**Fix under test.**
1. *Lexicon*: drop `model`/`models` and `module`/`modules` as kind nouns
   (same reasoning as the pre-existing `file`/`files` exclusion in the lexicon).
2. *Mechanism*: replace the hard pre-filter with a post-fusion score boost
   (×`kind_boost`, default 1.3) on hint-matching kinds — hint-correct queries
   keep their narrowing benefit; hint-wrong queries keep the gold in the pool.
   Modes: `hard` (old behavior) / `soft` (new default) / `off`.

## Setup

- Tree: branch `checkpoint/embedder-and-recall-groundwork` (test files excluded
  from indexing; local embedder `BAAI/bge-base-en-v1.5`, 768d).
- Repos indexed: `ritikk112/sutra` @ 432bde7 (672 symbols) — heuristic+lsp
  resolver; `fastapi/fastapi` @ 66b2c5a (2055 symbols, incl. 461 docs_src
  example files as realistic noise) — heuristic resolver.
- Retrieval metric set: 12 ground-truth queries per repo, each gold verified to
  exist in the index. Buckets: **A** kind-noun collision (noun is domain vocab,
  gold is another kind — predicts hard-filter failure), **B** behavioral,
  **C** explicit correct kind noun (regression guard — these are the queries the
  filter was built for). fastapi has 4 natural A-clashes ("model"×2, "schema",
  "types"); sutra has 2 ("model", "module").
- Agent metric set: 6 trace tickets per repo (sutra: the rigorous_sutra
  benchmark's own SL1-SS3; fastapi: FT1-FT6 below), solver = claude-haiku,
  NEUTRAL tool framing, 1 trial each. Measured: sutra-tool adoption, trace
  correctness (self-assessed against known gold mechanisms, not blind-graded).

### fastapi trace tickets (FT1-FT6)

- FT1: how does FastAPI decide whether an endpoint runs in the event loop or a
  threadpool (gold: routing.py run_endpoint_function → run_in_threadpool)
- FT2: how is an endpoint's return value filtered through response_model and
  where do response validation errors raise (gold: routing.py
  serialize_response → ResponseValidationError)
- FT3: how are query/path parameters validated and turned into endpoint kwargs
  (gold: dependencies/utils.py solve_dependencies / request_params_to_args)
- FT4: where does a request validation failure become the 422 JSON response
  (gold: exception_handlers.py request_validation_exception_handler)
- FT5: how does OAuth2PasswordBearer extract the bearer token and behave when
  the header is missing (gold: security/oauth2.py OAuth2PasswordBearer#__call__)
- FT6: how is the OpenAPI schema generated and cached on first access
  (gold: applications.py FastAPI#openapi → openapi/utils.py get_openapi)

## Baseline (shipped code: hard filter)

| repo | mode | recall@5 | recall@10 | MRR | zero-recall |
|---|---|---|---|---|---|
| sutra | hard (shipped) | 0.667 | 0.833 | 0.564 | 2/12 |
| sutra | off | 0.917 | 1.000 | 0.695 | 0/12 |
| fastapi | hard (shipped) | 0.417 | 0.500 | 0.225 | 6/12 |
| fastapi | off | 0.333 | 0.667 | 0.224 | 4/12 |

Per-query (fastapi, hard→off ranks): the three A-clashes go None→8, None→4,
None→None; the "types" clash stays None→None; but C's "exception class raised
when request validation fails" goes 2→7 — **the filter genuinely helps that
query**. This is the soft-boost argument in one table: `off` fixes the
catastrophes but gives back the narrowing benefit; neither extreme wins both.

Three fastapi queries fail in *every* mode (solve_dependencies,
request_params_to_args, analyze_param) — pure ranking failures on a
2055-symbol corpus with docs_src noise. Out of scope for this fix; noted so
the post-fix table is read honestly.

## Baseline agent runs (haiku, NEUTRAL framing, pre-fix server)

Tool calls per trial (sutra = mcp__sutra__* calls; agents used Bash for
grepping rather than the Grep tool):

| ticket | sutra | bash | read |   | ticket | sutra | bash | read |
|---|---|---|---|---|---|---|---|---|
| sutra SL1 | 2 | 12 | 11 |   | fastapi FT1 | 6 | 8 | 13 |
| sutra SL2 | 0 | 18 | 16 |   | fastapi FT2 | 7 | 9 | 15 |
| sutra SL3 | 0 | 10 | 10 |   | fastapi FT3 | 8 | 9 | 12 |
| sutra SS1 | 0 | 6 | 16 |   | fastapi FT4 | 0 | 13 | 16 |
| sutra SS2 | 0 | 12 | 14 |   | fastapi FT5 | 3 | 8 | 4 |
| sutra SS3 | 0 | 8 | 10 |   | fastapi FT6 | 7 | 4 | 13 |

**Adoption: sutra 1/6 trials (2 calls) vs fastapi 5/6 trials (31 calls).**
The index is barely touched on the small familiar-shaped repo and heavily
used on the larger unfamiliar one — consistent with the localization
literature (index value grows with corpus size). All 12 traces found the
correct gold mechanisms (self-assessed).

Caveat: the fix landed in the working tree mid-wave, so agents' file READS
may show post-fix code; the two measured quantities (tool adoption; search
results, served by the still-running pre-fix server process) are unaffected.

## The fix (implemented on this branch)

- `query_lexicon.yaml`: `model`/`models` and `module`/`modules` removed as
  kind nouns, with NOTE comments mirroring the file/files precedent.
- `kind_filter.py`: new `boost_kinds()` — post-fusion multiplicative boost
  (default x1.3) on hint-matching kinds, provenance-marked `kind_boost`.
- `pipeline.py`: new `kind_mode` param — `soft` (default) / `hard` (legacy,
  kept for A/B) / `off` — and `kind_boost`.
- Tests: 4 new pipeline tests (soft never erases; off ignores hints; hard
  still restricts; invalid mode rejected); analyzer tests updated for the
  lexicon change. Full suite: **814 passed, 0 failed**.

## Post-fix retrieval results (soft = new default; lexicon change included)

| repo | mode | recall@5 | recall@10 | MRR | zero-recall |
|---|---|---|---|---|---|
| sutra | hard (old) | 0.667 | 0.833 | 0.564 | 2/12 |
| sutra | off | 0.917 | 1.000 | 0.695 | 0/12 |
| **sutra** | **soft (new)** | **0.833** | **1.000** | **0.731** | **0/12** |
| fastapi | hard (old) | 0.417 | 0.500 | 0.225 | 6/12 |
| fastapi | off | 0.333 | 0.667 | 0.224 | 4/12 |
| **fastapi** | **soft (new)** | **0.583** | **0.667** | **0.263** | **4/12** |

Per-query: **soft never loses to hard on any of the 24 queries**, and it
keeps the narrowing benefit that `off` gives away — fastapi's "the exception
class raised when request validation fails" holds rank 2 under soft vs 7
under off. Both sutra zero-recalls are cured (the "module" query lands at
rank 1: with the noun gone, the verb fallback boosts the gold function).
fastapi's remaining 4 zero-recalls are the pre-existing ranking failures
that fail in every mode (docs_src noise), out of scope for this fix.

MRR is the highest of all three modes on both repos. On sutra, soft gives
back one query vs `off` at recall@5 (a verb-fallback boost nudges other
functions past one gold, 5->8) — the price of keeping the C-bucket wins;
the zero-recall column, the metric that motivated the fix, is clean.


## Post-fix agent wave — deferred to next session

The registered MCP server is a stdio child of the Claude Code session and
loaded the pre-fix code at session start. Killing it to force a code reload
drops the session's connection permanently — the harness does not respawn a
dead stdio server mid-session (verified by probe). The post-fix haiku wave
(same 12 tickets, NEUTRAL framing) therefore needs a fresh session, where
the server starts on the fixed branch automatically. The baseline wave's
numbers above are the comparison point; adoption is not expected to change
(framing and tools are identical) — the interesting post-fix question is
whether the FT-ticket agents' sutra_search calls return the gold symbols in
fewer attempts.

## Post-fix agent wave — results (run 2026-08-18, fresh session)

Setup: fresh Claude Code session, MCP server auto-started on the fixed
branch (soft mode confirmed live via `kind_boost: 1.3` in search
provenance). Prompts recovered **verbatim** from the baseline session's
transcript (byte-identical wording, one uniform NEUTRAL template), same
solver (haiku), same 12 tickets, 1 trial each, read-only. Gold-rank
analysis scripted over both waves' task transcripts; oversized tool
results (5 calls) recovered from their persisted files and scored too.

### Tool calls per trial (compare baseline table above)

| ticket | sutra | bash | read |   | ticket | sutra | bash | read |
|---|---|---|---|---|---|---|---|---|
| sutra SL1 | 7 | 0 | 6 |   | fastapi FT1 | 6 | 4 | 10 |
| sutra SL2 | 3 | 10 | 14 |   | fastapi FT2 | 11 | 1 | 13 |
| sutra SL3 | 3 | 0 | 7 |   | fastapi FT3 | 5 | 16 | 16 |
| sutra SS1 | 1 | 5 | 12 |   | fastapi FT4 | 8 | 10 | 18 |
| sutra SS2 | 0 | 4 | 15 |   | fastapi FT5 | 5 | 3 | 3 |
| sutra SS3 | 3 | 5 | 12 |   | fastapi FT6 | 8 | 2 | 11 |

**Adoption: sutra 5/6 trials (17 calls) vs baseline 1/6 (2 calls);
fastapi 6/6 (43 calls) vs baseline 5/6 (31 calls).** Framing was
byte-identical, so this shift is not presentation. With 1 trial/ticket it
may partly be run-to-run variance, but the direction is consistent with a
feedback effect: searches that return the gold get followed by more
searches. All 12 traces again found the correct gold mechanisms
(self-assessed) — correctness unchanged at 12/12.

### Searches-until-gold (sutra_search calls until the gold symbol appears
anywhere in a result list; rank = gold's position in that first hit)

| ticket | baseline | post-fix |   | ticket | baseline | post-fix |
|---|---|---|---|---|---|---|
| sutra SL1 | #1 (r1) | #1 (r1) |   | fastapi FT1 | #1 (r7) | #1 (r2) |
| sutra SL2 | no searches | #1 (r3) |   | fastapi FT2 | #2 (r1) | #1 (r10) |
| sutra SL3 | no searches | #1 (r1) |   | fastapi FT3 | #1 (r6) | #3 (r8) |
| sutra SS1 | no searches | #1 (r1) |   | fastapi FT4 | no searches | #1 (r3) |
| sutra SS2 | no searches | no searches |   | fastapi FT5 | #1 (r1) | #1 (r1) |
| sutra SS3 | no searches | #3 (r3) |   | fastapi FT6 | #3 (r1) | #1 (r3) |

The cleanest single observation is a natural paired A-clash query on FT2:
baseline query #1 `response_model validation serialization filtering` →
gold **absent** from all 15 results (hard filter, "model" kind noun); the
post-fix agent's near-identical first query `response_model validation
filtering serialization` → gold (serialize_response) present at rank 10,
and its trace pivoted from there. Same phrasing, hard-erased vs
soft-retained — the fix's mechanism observed in the wild, not just in the
retrieval eval.

Reading the table honestly: first-query gold hits on fastapi went 2/5
ticket-trials-that-searched (FT1, FT5) to 5/6 (all but FT3), and FT6's
first natural phrasing (`openapi schema generation`) now hits at #1 vs #3.
But FT3 got *worse* on first-hit (#1→#3) — its golds are the pre-existing
every-mode ranking failures (docs_src noise), so its hits in both waves
are luck-of-phrasing, not filter behavior. Per-call hit rate across all
fastapi searches is flat (~52% baseline, ~51% post-fix): post-fix agents
issued more narrow follow-up probes (exact private-symbol lookups at
top_k=5) whose "misses" are not failures. FT4 is not comparable on this
metric (baseline agent never searched).

### Agent-wave verdict

The wave shows the fix where it was predicted to show: ticket phrasings
that collide with kind nouns (FT2's "response_model") no longer erase the
gold, and first-query gold-landing on the fastapi tickets improved from
2/5 to 5/6. Index adoption rose on both repos under byte-identical
framing — consistent with better results reinforcing use, though 1
trial/ticket leaves the adoption delta suggestive rather than proven.
Correctness stayed 12/12. Nothing in the wave contradicts the retrieval
eval; the soft-boost default stands.

## Verdict

The hard kind pre-filter is replaced as default. Measured across 24
ground-truth queries on two repos, soft boost (x1.3) + the lexicon fix:
- never ranks worse than the hard filter on any single query,
- cures all hard-filter zero-recalls (2 on sutra, 2 of 6 on fastapi; the
  other 4 fail in every mode and are unrelated ranking failures),
- keeps the explicit-kind narrowing that plain removal (`off`) loses,
- gives the best MRR of the three modes on both repos.

`kind_mode="hard"` remains available for A/B. The original P16 evidence
(+0.089 MRR on booth/outreach/gin) has not been re-run — those artifacts
require repos on the other machine; scripts/build_eval_artifacts.py is also
missing from the repo (doc/code drift). Re-running that eval under soft
mode is the remaining validation step.
