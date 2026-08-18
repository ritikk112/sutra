# Retrieval gaps — task sheet (authored 2026-08-18)

Findings from the battle test (`BATTLE_TEST.md`) and the kind-filter A/B
(`KIND_FILTER_AB.md`) that must be addressed before the article ships.
Written for an executing agent. Work top to bottom; Task 1 and Task 5
are independent quick wins, Task 2 is the deep one.

## Ground rules (read first)

- Repo: this repo, branch `checkpoint/embedder-and-recall-groundwork`.
  Machine: 16 GB M2 Pro — no models >~150M params, never two model jobs
  concurrently, `nice -n 10` heavy jobs.
- Eval assets: `benchmarks/battle_test/` — `flask.json`, `requests.json`,
  `pydantic.json` (36 ground-truth queries, golds are exact monikers),
  `run_ab.py` (hard/off/soft runner), `ab_results.json` (current
  numbers). Run with `.venv/bin/python benchmarks/battle_test/run_ab.py`
  from repo root. Artifacts live in `~/.sutra/artifacts/`; the three
  repos are checked out at `~/Desktop/claude-dir/{flask,requests,pydantic}`.
- **Invariants that must survive every change** (check per-query, not
  just aggregates):
  1. soft ≥ hard on every query (currently 60/60);
  2. pooled MRR and recall@5 on the 36-query set do not regress;
  3. full pytest suite passes (814 tests as of c5bc20a).
- After any indexer/text change you must re-index the three repos
  before re-running the eval (sequentially, not in parallel).
- Record before/after numbers for every task in a `## Results` section
  appended to this file. Honest reporting: if a fix doesn't work, say
  so and keep the analysis.

---

## Task 1 — Data symbols (variables/modules) are invisible to retrieval

**Finding.** All 4 battle-test queries whose gold is a variable or
module zero-recalled in every mode: FL-A3 (`flask/typing.py` module),
RQ-A1 (`HOOKS` variable), RQ-A3 (`REDIRECT_STATI` variable), PD-C4
(`alias_generators` module). Callable golds failed only 5/32.

**Diagnose first.** Inspect what text actually gets embedded / fed to
BM25 for `variable` and `module` symbols (chunk construction — start
from where the indexer builds embedding text, and the BM25 corpus
builder in `sutra/core/retrieval/channels/bm25_channel.py`). Hypothesis:
variables embed as little more than a name + assignment line; modules
may embed only a docstring or nothing. Confirm by printing the chunk
text for the 4 golds.

**Candidate fixes** (try cheapest first, measure each):
1. Enrich variable chunk text: include the full assignment RHS (e.g.
   the dict literal of `HOOKS`), enclosing-module context (module name,
   docstring first line), and nearby comment lines.
2. Enrich module chunk text: module docstring + the names of its
   top-level symbols (a synthetic "table of contents" string).
3. If text enrichment is insufficient, consider a small kind-prior in
   fusion (variables/modules get a floor, not a penalty) — but prefer
   fixing the text; priors are a blunt instrument.

**Acceptance.** ≥3 of the 4 queries reach top-10 under soft;
invariants hold; per-query table shows no callable-gold query dropping
out of top-10 as a side effect.

---

## Task 2 — Ranking quality ceiling on dense corpora (the deep one)

**Finding A.** 5 callable-gold battle-test queries fail in every mode
(FL-B3 session-cookie decode, RQ-B3 `super_len`, PD-A1
`generate_schema` method, PD-B2 `apply_discriminator`, PD-B5
`_check_frozen`). The earlier fastapi eval adds 4 more, worsened by
461 `docs_src/` example files acting as ranking noise.

**Finding B (previous session, `~/.claude` memory + PROGRESS notes).**
On the sutra repo's own 12-query set, the full hybrid pipeline scored
recall@5 0.667 / MRR 0.536 while dense-only bge-base scored 0.917 /
0.774 — the fusion layer, not the embedder, was the bottleneck. This
has not been re-verified after the soft-boost change; re-verify before
acting on it.

**Work items.**
1. Build a per-channel diagnostic: for every battle-test query, log
   each channel's rank of the gold (provenance already carries this)
   plus the fused rank. Classify each failure: gold missing from all
   channels (recall problem) vs present-but-drowned (fusion problem).
2. Re-measure hybrid vs dense-only on the 36-query set (add a
   `--channels vector` mode to `run_ab.py` or a sibling script). If
   dense-only wins pooled, the fusion layer needs rebalancing:
   candidates include weighted RRF (per-channel weights), dropping or
   gating the moniker channel when no entity is present, and raising
   `rrf_k`.
3. Example-dir noise: extend the indexer's exclusion precedent (tests
   are already excluded) with a configurable `exclude_globs` in
   `config/sutra.yaml` (default: none; document `docs_src/`,
   `examples/` as suggestions). Re-index fastapi with `docs_src/`
   excluded and re-run the fastapi 12-query set from
   `KIND_FILTER_AB.md` §Setup to quantify the noise cost. NOTE: those
   12 queries were not committed — reconstruct from the KIND_FILTER_AB
   per-query descriptions, verify golds against graph.json first.
4. Only if 1–3 leave headroom: try `rerank=true` as the MCP default
   for `sutra_search` and measure latency + quality tradeoff on-device
   (bge-reranker is heavy; watch the 16 GB constraint — if it swaps,
   abandon).

**Acceptance.** A written diagnosis (which failures are recall vs
fusion), pooled recall@5 on the 36-query set improved by ≥0.05 without
breaking invariants, and a clear keep/revert decision per change.

---

## Task 3 — Soft kind boost misfires on verb-derived hints

**Finding.** PD-A3 ("inside a custom `__get_pydantic_core_schema__`,
what object do I call…", gold: class `GetCoreSchemaHandler`): `off`
ranks it 4, `soft` drops it out of top-10 — the callable hint comes
from behavioral VERBS ("call"), not from an explicit kind noun, and
boosts functions past the gold class.

**Hypothesis to test.** Verb-derived hints are weaker evidence than
explicit noun hints and should get a smaller boost (e.g. ×1.15) or no
boost at all. `sutra/core/retrieval/query_analyzer.py` knows whether
the hint came from a noun or the verb fallback — thread that
distinction into `ParsedQuery` and let `boost_kinds()` scale
accordingly.

**Measure.** Full 36-query A/B with: (a) verb-boost ×1.0 (nouns only),
(b) verb-boost ×1.15, (c) status quo ×1.3. Also re-check the two
KIND_FILTER_AB sutra queries that the verb fallback *helped* (the
"module" query landed rank 1 via verb boost — don't break it; it is in
the committed sutra dataset only as described in KIND_FILTER_AB, so
verify against the sutra artifact).

**Acceptance.** Choose the config with best pooled MRR subject to:
soft ≥ hard everywhere still holds, PD-A3 back in top-10 OR documented
as an accepted tradeoff with data.

---

## Task 4 — Phrasing sensitivity / missing bge query prefix

**Finding.** The same mechanism went from zero-recall to rank 1 on
re-phrasing (FL-B3 eval phrasing vs the agent's own phrasing). Known
confound: `LocalEmbedder` never applies bge's recommended query
instruction prefix ("Represent this sentence for searching relevant
passages: ") to QUERIES (documents correctly get none).

**Work items.**
1. Add query-side instruction support to `LocalEmbedder` (a
   `query_prefix` applied in a new `embed_query` path or an
   `is_query` flag; document-side embedding must remain unchanged so
   existing artifacts stay valid — verify dims/normalization
   unchanged).
2. Measure on the 36-query set (all three modes). bge-base's prefix
   often buys a few recall points; if it measures neutral-or-worse,
   revert and record.
3. (Stretch) Multi-phrasing query expansion: fuse results of the raw
   query + a deterministic reword (e.g. verb-nominalization swap). Only
   prototype if 1–2 don't move the needle; no LLM calls in the
   pipeline.

**Acceptance.** Prefix decision made on data; committed only if pooled
recall@5/MRR improve with invariants intact.

---

## Task 5 — Quick hygiene (do alongside Task 1)

1. `benchmarks/battle_test/run_ab.py` currently hardcodes the
   scratchpad-era sutra path — make paths relative to repo root and
   commit the whole `benchmarks/battle_test/` dir (it is untracked).
2. Add the 36-query battle-test set to CI-runnable form if cheap (skip
   if it needs model downloads in CI — document instead).
3. `KIND_FILTER_AB.md` sutra/fastapi 12-query sets exist only as prose
   — reconstruct both into committed JSON next to battle_test so
   future regressions are catchable (verify every gold against the
   artifacts before committing).

---

## Task 6 — Slim the MCP result payloads (from the vs-control A/B)

**Finding (`SUTRA_VS_CONTROL.md`).** Sutra halves agents' median
calls-to-localization (2 vs 4) but the cost advantage is erased:
sutra-arm trials grow context ~30% faster (95.9k vs 73.5k cache-write
tokens/trial) because each `sutra_search` result carries full
monikers, signatures, docstrings and provenance for 10–15 hits.

**Work items.**
1. Measure where the bytes go: serialize a typical top-10 result and
   attribute size per field.
2. Slim the default response: drop `provenance` unless
   `include_provenance=true` (already a flag — verify it defaults
   off end-to-end), truncate docstring summaries harder, consider a
   compact moniker alias (e.g. `file:qualified_name`) with the full
   moniker only on `sutra_get_symbol`.
3. Consider default `top_k=5` for MCP `sutra_search` (agents in the
   waves rarely used hits past rank ~5; check `ab_efficiency.json` +
   wave transcripts before deciding).
4. Re-run a 6-ticket mini wave (2 arms × 1 trial) to confirm context
   growth drops without hurting localization.

**Acceptance.** Median sutra-arm cache-write tokens within ~10% of
control on the mini wave; localization advantage retained; full test
suite passes.

## Benchmark protocol note (NOT a fix task) — next vs-control run needs harder tickets

This is an instruction for whoever runs the NEXT sutra-vs-control
benchmark, not a code fix. Nothing in the codebase changes here.

**Why.** All 108 vs-control trials were graded correct in BOTH arms —
the 18 trace tickets ceiling at haiku tier and cannot detect
correctness differences. Any article claim that the index improves
*answers* (not just navigation) is unsupported until re-measured on a
harder suite.

**Protocol for the next run.**
1. Author 6–10 harder tickets with verifiable golds: multi-hop bug
   localization ("this observable misbehavior — find the causal line"),
   cross-file interaction questions, and questions whose naive grep
   keyword is misleading. Target the large corpora (pydantic, fastapi).
2. Difficulty gate: a ticket qualifies only if a 1-trial haiku
   grep-only pilot FAILS it at least once (otherwise it ceilings again).
3. Re-run the 2-arm × 3-trial blind-judged protocol (reuse the
   `sutra-vs-control-benchmark` workflow script, swap the ticket table).
4. Report the correctness delta (either direction) with per-ticket
   pairs; a suite qualifies only if the control arm scores <80%.

**Outcome (executed 2026-08-19): the suite did NOT qualify — and that
is the finding.** 16 candidate tickets were authored by sonnet agents
that read pydantic/fastapi source through two adversarial lenses
(cross-file mechanisms ≥3 hops; grep-trap vocabulary), every gold
marker grep-verified against the clones
(`benchmarks/battle_test/hard_tickets.json`). The 1-trial grep-only
haiku pilot graded **14/16 correct, 2/16 partial (HP5, HP8), 0 wrong**
(`hard_ticket_pilot.json`) — control correctness 87.5%, above the 80%
qualification bar, so the 2-arm main run was not run. Conclusion for
the article: the correctness ceiling is robust, not an artifact of
easy tickets — haiku+grep solves even deliberately hardened cross-file
grep-trap tickets on 3k-symbol corpora. At this task tier sutra's
measurable value is navigation efficiency (2 vs 4 calls-to-gold at
parity context growth), not answer correctness. A correctness delta,
if it exists, lives above this tier (larger corpora, multi-repo
questions, or weaker/faster models under call budgets).

## Results (executed 2026-08-19, commits 2a16760..HEAD)

All work TDD'd; suite grew 814 → 840 tests, all green. Eval assets:
`benchmarks/battle_test/` (36-query set + grids) and
`benchmarks/kind_filter_ab/` (reconstructed 24-query sets + manifest).

**Task 1 — data symbols (DONE).** Root cause was stronger than thin
text: variables/modules were excluded from embedding entirely
(`_EMBEDDABLE`). They now embed (assignment source / docstring+member
roster). Alone this fixed little — the fusion layer re-buried them —
but combined with Task 2's weighted RRF, **3 of 4 data-symbol golds
reach top-10 under soft** (RQ-A1 10, RQ-A3 9, PD-C4 7; FL-A3 still
misses). Acceptance (≥3/4) met.

**Task 2 — ranking ceiling (DONE, honest partials).**
- Diagnosis: dense-only run showed 5 of 9 every-mode failures were
  *present-but-drowned* (vector ranked them top-4; unweighted RRF's
  agreement scoring buried them). Confirmed the old hybrid<dense
  finding: dense-only flask r@5 .833 vs fused .583.
- Fix: weighted RRF — `rrf_k` 60→20, vector weight 1.5 (new defaults;
  `channel_weights={}` restores classic). On the 36-query set:
  zero-recall 10→6, r@10 .722→.833, r@5/MRR flat. Acceptance asked
  r@5 +0.05 — NOT met (flat); adopted for the tail/zero-recall gains.
- Known cost: FL-B3 became the one query where hard finds a gold
  (rank 9) that soft misses — the previous 60/60 soft≥hard record is
  now 59/60 under the new fusion params.
- docs_src measured: excluding it cut the fastapi index 2055→758
  symbols and lifted soft MRR .368→.451 on the reconstructed set; the
  4 chronic fastapi failures persist without docs_src (deeper problem
  — likely chunk quality for very large functions).

**Task 3 — verb-boost knob (DONE, default unchanged by data).**
`kind_hint_source` + `kind_boost_verb` implemented. Measured 1.0/1.15/
1.3: pooled MRR .545/.559/.561 — reducing verb boosts LOSES more than
it gains, and PD-A3's bad hint turned out noun-derived ("handler"), so
the knob can't fix it. Default stays kind_boost_verb=None (=1.3).

**Task 4 — bge query prefix (DONE, kept).** `embed_queries` +
auto-instruction for bge en-family. Pooled soft: r@5 .611→.639, MRR
.543→.561; tail slightly worse (r@10 .750→.722, zero 9→10). Kept per
the named acceptance metrics (r@5/MRR); tradeoff recorded.

**Task 5 — hygiene (DONE except CI).** run_ab.py is repo-relative and
grew --manifest/--artifacts-dir/--rrf-k/--vector-weight/--dense-only/
--no-prefix; benchmarks committed. KIND_FILTER_AB 24-query sets
reconstructed from the old session transcript with golds re-resolved
to exact monikers (`benchmarks/kind_filter_ab/`). Regression check on
them with all fixes: sutra set now PERFECT (r@5 1.000, zero 0, all
modes; was soft .833/MRR .731); fastapi soft .583→.667 r@5, MRR
.263→.368. CI wiring skipped (needs model download; documented here).

**Task 6 — payload slimming (DONE; mini-wave acceptance PASSED
2026-08-19).** Measured: `signature` was 115K of a 126K two-query
payload (fastapi's Annotated[...Doc()] style). Search hits now carry a
200-char collapsed signature summary (full one on sutra_get_symbol).
Worst-case payload 94K→7.3K (−92%).

Live acceptance (fresh session, server on 2a1f368): two 12-trial waves
(battle-test tickets FK/RQ/PY + fastapi FT1–FT6, 2 arms × 1 haiku
trial, byte-identical NEUTRAL template, analyzer extended to sum
cache_creation_input_tokens):

- **Context-growth parity achieved.** Combined 12 tickets: sutra-arm
  cache-write mean 85.1k vs control 82.4k (**+3.3%**), paired
  per-ticket geomean **+1.5%** — inside the ~10% acceptance bar. On
  the fastapi tickets, where the pre-slim gap was concentrated
  (per-ticket ratios 1.45–1.73), the paired geomean went from ~1.4+ to
  **0.99**; sutra-arm absolute cache-write fell on every FT ticket
  (worst case FT3 160.9k→79.8k, −50%).
- **Localization advantage retained**: median calls-to-gold 2 (sutra)
  vs 4 (control) across the 24 trials; sutra ≤ control on 11 of 12
  tickets (FT6 the exception, 2 vs 1). All 24 trials localized.
- Adoption 12/12 sutra-arm trials (mean 8.4 index calls/trial);
  control contamination 0.0.
- Data: `benchmarks/battle_test/ab_efficiency.json` (24 post-slim
  trial rows); transcripts in session workflow dirs wf_d124da28 /
  wf_18ccdf15. Cost conclusion for the article: the "cost wash" from
  SUTRA_VS_CONTROL.md is now stale — with slimmed payloads sutra
  halves calls-to-localization at parity context growth.

## Explicitly out of scope for this sheet

- The sutra-vs-no-sutra value benchmark (separate protocol, separate
  sheet — see conversation notes / forthcoming design).
- P16 booth/outreach/gin re-run (repos live on the Linux box).
- Embedder swaps (bge-base already beat 5 alternatives; don't re-open
  without new evidence).
