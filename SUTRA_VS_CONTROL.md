# Sutra vs no-sutra — is it worth installing? (2026-08-18)

Controlled A/B measuring what the sutra index adds to a coding agent
over the tools every agent already has (Bash/Grep/Glob/Read).
Companion to `BATTLE_TEST.md` and `KIND_FILTER_AB.md`.

## Protocol

- 18 trace tickets across 5 repos: sutra (672 symbols), fastapi
  (2055), flask (496), requests (344), pydantic (2793). Ticket IDs
  match the earlier waves (SL/SS, FT, FK/RQ/PY).
- 2 arms × 3 trials each = 108 solver runs, claude-haiku, identical
  NEUTRAL prompts; the CONTROL arm's prompt simply omits the sutra
  tools and says "use only these tools". Sutra-repo tickets told both
  arms to ignore the repo's markdown docs (they contain answer keys).
- Grading: every trace blind-judged by a separate sonnet judge against
  pre-authored gold mechanisms (judge never sees which arm produced a
  trace). No self-assessment.
- Efficiency: per-transcript tool-call counts, calls-to-localization
  (tool calls until the gold mechanism first appears in an input or
  result), token usage, wall-clock.
- Contamination check: control-arm sutra-tool calls = **0.0** (clean).
- Scale: 216 agents, ~8.6M subagent tokens, 33 min wall-clock.

## Results

### Correctness: a ceiling — no difference

| arm | correct | partial | wrong |
|---|---|---|---|
| sutra | 54/54 | 0 | 0 |
| control | 54/54 | 0 | 0 |

Haiku solved every ticket in every trial with or without the index.
These trace tickets cannot differentiate correctness at this model
tier — an important negative result: **on tasks an agent can already
solve with grep, the index does not change the outcome.**

### Localization speed: sutra halves it

| metric (median/trial) | sutra | control |
|---|---|---|
| calls until gold mechanism first touched | **2** | 4 |
| total tool calls | **23** | 27 |
| output tokens | 6,404 | 6,963 |
| context growth (cache-write tokens) | 95.9k | **73.5k** |
| cost units (in + cw + 0.1·cr + 5·out) | 306k | 313k |
| wall-clock | 117s | 121s |

Per-ticket paired calls-to-gold (sutra | control): sutra **wins 12**,
ties 2, loses 4 of 18. The four losses are FT6 (3|1 — "openapi" greps
instantly by name) and three of the six sutra-repo tickets (SL2 4|3,
SS2 5|4, SS3 2|1). On the 12 tickets outside the small sutra repo,
sutra wins 11 and loses 1.

| ticket | s | c | | ticket | s | c | | ticket | s | c |
|---|---|---|---|---|---|---|---|---|---|---|
| SL1 | 4 | 4 | | FT1 | 2 | 4 | | FK1 | 2 | 3 |
| SL2 | 4 | 3 | | FT2 | 2 | 4 | | FK2 | 3 | 5 |
| SL3 | 2 | 4 | | FT3 | 5 | 8 | | RQ1 | 2 | 3 |
| SS1 | 3 | 3 | | FT4 | 2 | 5 | | RQ2 | 2 | 3 |
| SS2 | 5 | 4 | | FT5 | 2 | 3 | | PY1 | 2 | 3 |
| SS3 | 2 | 1 | | FT6 | 3 | 1 | | PY2 | 3 | 4 |

### Cost: a wash — and why

The localization win does not convert into cost savings: sutra trials
grow their context ~30% faster (95.9k vs 73.5k cache-write tokens)
because sutra_search result payloads are verbose (full monikers,
signatures, provenance for 10–15 hits per call, plus the one-time
ToolSearch schema load). Net cost lands within noise (−2%), wall-clock
within noise.

## Verdict (for the article, stated plainly)

1. **Correctness on solvable tasks: no effect.** With haiku on
   realistic trace tickets, grep-only agents matched the index arm
   54/54. Anyone claiming an index makes agents "smarter" on tasks
   like these is overclaiming.
2. **Navigation: real, repo-size-dependent effect.** The index halves
   the median path to the right mechanism (2 vs 4 tool calls) and wins
   the paired comparison 12–4. Wins concentrate on larger unfamiliar
   corpora; on a small greppable repo the index is a wash-to-slight
   loss. This matches the adoption data from earlier waves (1/6 usage
   on small repo, 6/6 on large ones).
3. **Cost today: neutral.** Faster localization is currently cancelled
   out by verbose result payloads. This is fixable (see follow-up
   tasks) — payload slimming would likely turn the draw into a win.
4. **Where the value should show but wasn't tested here:** tasks hard
   enough that mis-localization causes wrong answers (multi-hop bug
   hunts, cross-repo questions, corpora ≫3k symbols), and
   latency-sensitive interactive use where 4 fewer calls matter.

**Bottom line:** worth installing if your agents work in large or
unfamiliar codebases and you value navigation speed; not yet a cost or
correctness win on small repos or straightforward tickets — and the
report says so.

## Limitations

- One model tier (haiku), 3 trials/ticket. A sonnet replication is the
  obvious next step; sonnet greps better, so the control likely gets
  stronger.
- Correctness ceiling: harder tickets are needed before any
  correctness claim in either direction.
- Calls-to-gold counts a gold string appearing in a tool input/result,
  which slightly favors whichever arm's tool outputs are more verbose.
- Judges graded strictly but from a written gold; a gold-blind human
  spot-check has not been done.

## Reproduction

- Workflow script: `sutra-vs-control-benchmark` (session workflows dir);
  results journal + all 108 solver transcripts in the session's
  workflow transcript dir.
- Efficiency analyzer + raw rows: `benchmarks/battle_test/`
  neighbors (`analyze_ab_transcripts.py`, `ab_efficiency.json` — copy
  committed alongside this doc).
