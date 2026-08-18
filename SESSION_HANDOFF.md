# Session handoff — sutra kind-filter fix (written 2026-08-18)

Read this first, then `KIND_FILTER_AB.md` (same dir) for full data.
Auto-memory also covers most of this (memory file `sutra-setup.md`).

## Where things stand

- Repo: `/Users/ritikshukla/Desktop/claude-dir/sutra`, branch
  **`checkpoint/embedder-and-recall-groundwork`**, clean tree
  (only untracked: `config/sutra.yaml.openai.bak`, this file).
- Commits on top of main (432bde7):
  1. `99ab417` fix(mcp): registry passes artifact embedding dims (bug: any
     non-384-dim local embedder was unservable)
  2. `1d1f774` feat(indexer): test files excluded from indexing
  3. `2cf518f` chore(config): local embedder = BAAI/bge-base-en-v1.5, 768d
  4. `c5bc20a` feat(retrieval): **soft kind boost replaces hard pre-filter**
     (`kind_mode="soft"` default, ×1.3), `model`/`module` dropped from
     `query_lexicon.yaml`. Full suite 814 passed.
- Indexed artifacts in `~/.sutra/artifacts/`: `ritikk112__sutra` (672 sym),
  `fastapi__fastapi` (2055 sym, clone at `~/Desktop/claude-dir/fastapi` @
  66b2c5a), `pytoolz__toolz` (193 sym). All bge-base 768d. `.prev` rollback
  copies inside artifact dirs are normal, not a leak.
- MCP server `sutra` registered at user scope; it was killed mid-session
  (dead stdio servers don't respawn until a new session). **In this new
  session it starts fresh on the fixed code — verify with sutra_list_repos.**

## Measured results (details + per-query tables in KIND_FILTER_AB.md)

Retrieval, 12 ground-truth queries/repo (recall@5 / zero-recall):
- sutra:   hard 0.667 (2 zero) · off 0.917 (0) · **soft 0.833 (0), best MRR 0.731**
- fastapi: hard 0.417 (6 zero) · off 0.333 (4) · **soft 0.583 (4), best MRR 0.263**
- Soft never lost to hard on any of 24 queries; 4 fastapi queries fail in
  every mode (docs_src ranking noise — unrelated to this fix).

Agent adoption (haiku, NEUTRAL framing, pre-fix server): sutra 1/6 trials
used the index vs **fastapi 5/6 (31 calls)** — index value grows with corpus
size. Earlier sonnet experiment: original benchmark's 0/36 "agents never use
the index" replicated with its own wording, but neutral framing moved
adoption to 3/6 → the 0/36 was substantially a presentation artifact.

## Pending work (in order)

1. ~~**Post-fix agent wave**~~ **DONE 2026-08-18** — results + verdict
   appended to KIND_FILTER_AB.md §"Post-fix agent wave — results".
   Headline: adoption sutra 1/6→5/6 trials, fastapi 5/6→6/6; fastapi
   first-query gold hits 2/5→5/6; FT2's "response_model" phrasing no
   longer erases the gold (paired baseline/post observation); 12/12
   correct traces both waves. Prompts were recovered verbatim from the
   baseline session transcript (`~/.claude/projects/-Users-ritikshukla-Desktop-claude-dir/d81aa42f-*.jsonl`).
2. Re-run the P16 eval (booth/outreach/gin) under soft mode — blocked on
   this machine (repos live on the Linux box; `scripts/build_eval_artifacts.py`
   is referenced by tests but missing from the repo — doc/code drift).
3. Decide merge to main + whether to upstream `KIND_FILTER_AB.md` findings
   into the benchmark branch's reports (their 0/36 conclusion needs the
   framing-confound caveat; their cache-cost explanation overstates the
   1.25× premium — cached reads are 0.1×).

## Gotchas

- 16GB M2 Pro: no models >~150M params, never two model jobs concurrently
  (Qwen3-0.6B previously wedged the machine). `nice -n 10` heavy jobs.
- Eval scripts from the old session lived in its /tmp scratchpad — gone
  after restart. Recreate from KIND_FILTER_AB.md's query lists if needed
  (golds are exact monikers; verify against graph.json before use).
- `LocalEmbedder` has no `trust_remote_code` — jina/nomic models silently
  load with random weights (cosine ~1.0 for everything). Sanity-check any
  new embedder with similar/dissimilar snippet pairs.
- bge query-instruction prefix is NOT applied by LocalEmbedder — known
  confound in the embedder comparison, unfixed.
