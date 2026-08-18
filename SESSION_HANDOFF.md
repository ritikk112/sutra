# Session handoff — sutra benchmarking + gap fixes (updated 2026-08-19)

Read this, then `RETRIEVAL_GAPS_TASKS.md` (§Results) for what was fixed
and measured. Reports: `BATTLE_TEST.md`, `SUTRA_VS_CONTROL.md`,
`KIND_FILTER_AB.md`. Goal context: the user is writing an article on
sutra; all numbers are article source material.

## Where things stand

- Repo: `/Users/ritikshukla/Desktop/claude-dir/sutra`, branch
  **`checkpoint/embedder-and-recall-groundwork`**, clean tree, HEAD
  `b7802d3`. Full suite 840 passed.
- Fix commits today (all TDD'd): 2a16760 variables/modules embed;
  1fbe612 verb-hint source + kind_boost_verb; 78a66a4 bge query
  prefix; 62dd160 weighted RRF (k=20, vector 1.5 defaults); 5208550
  indexing.exclude_globs; 2a1f368 MCP signature summary (payload −92%).
- Eval assets committed: `benchmarks/battle_test/` (36-query set,
  grids, runner with --manifest) and `benchmarks/kind_filter_ab/`
  (reconstructed 24-query sets).
- Artifacts in `~/.sutra/artifacts/` all re-indexed on current code:
  sutra 687 sym @2a1f368, fastapi 2055 (docs_src INCLUDED — the
  excluded variant lives only in the old scratchpad, gone after
  restart), flask 496, requests 344, pydantic 2793, toolz 193 (stale,
  old code — re-index if used).
- Headline numbers: battle-test soft zero-recall 10→6, r@10
  .722→.833; kind-filter sutra set r@5 1.000 all modes; fastapi soft
  MRR .263→.368 (.451 with docs_src excluded); soft≥hard now 59/60
  (FL-B3 the one flip, documented).

## Pending work (in order)

1. ~~Task 6 mini agent wave~~ **DONE 2026-08-19**: two 12-trial waves
   on the fresh server (battle-test + FT tickets). Cache-write gap
   +3.3% mean / +1.5% paired geomean (was +30% median pre-slim);
   fastapi tickets went from ratio ~1.4+ to 0.99; localization
   retained (2 vs 4). Results in RETRIEVAL_GAPS_TASKS.md §Results
   Task 6; SUTRA_VS_CONTROL.md cost section marked superseded;
   `benchmarks/battle_test/ab_efficiency.json` now holds the 24
   post-slim trial rows.
2. **Harder-ticket vs-control rerun** — see the protocol note in
   RETRIEVAL_GAPS_TASKS.md (correctness ceiling: 54/54 both arms).
   Workflow script to reuse: session workflows dir,
   `sutra-vs-control-benchmark` (also summarized in SUTRA_VS_CONTROL.md).
3. Decide: default-exclude docs_src-style dirs for fastapi in the
   served artifact (re-index with the config) — +.083 MRR measured.
4. P16 booth/outreach/gin re-run — still blocked (repos on Linux box).
5. Merge to main + article drafting (user decision).

## Gotchas (still true)

- 16GB M2 Pro: no models >~150M params, never two model jobs
  concurrently, `nice -n 10` heavy jobs.
- Killing the registered stdio MCP server mid-session drops it for the
  session; artifacts hot-reload fine (ArtifactWatcher), code does not.
- LocalEmbedder has no `trust_remote_code` — jina/nomic silently load
  random weights. (The bge query-prefix confound is FIXED: embed_queries
  applies it automatically for BAAI/bge-*-en* models.)
- sutra's own index now contains benchmarks/*.py — fine for serving,
  but remember when eval golds count symbols.
