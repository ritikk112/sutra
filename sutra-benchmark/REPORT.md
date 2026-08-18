# Sutra Token-Efficiency Benchmark

**Date:** 2026-07-08
**Overseer:** main session (Opus 4.8)
**Question:** How much does the Sutra code index actually save an AI agent on real
tracing/debugging work — in tokens and in steps — versus an agent that only has
grep/find/read?

---

## 1. Method

Two background agents were handed the **same 5 vague tracing/debugging tickets**
over the **same repository** (`/home/ritik/Desktop/sutra` — the Sutra codebase
itself, which is both checked out on disk *and* indexed in the Sutra MCP server as
repo `sutra`). The only difference between them was tool access:

| | Agent A | Agent B |
|---|---|---|
| **Sutra MCP tools** | ✅ primary navigation | ❌ forbidden |
| **grep / find / read** | ✅ (confirmation only) | ✅ (only option) |
| **Model** | claude-sonnet | claude-sonnet |
| **Repo** | sutra | sutra |
| **Ticket wording** | identical | identical |

The tickets were kept deliberately vague ("something's off with…", "figure out
how…", "seems to…") so the agent had to *locate* the relevant code before it could
reason — exactly the phase where a code index is supposed to help.

### The 5 tickets
- **T1** — trace where local embedding dimensions are decided/validated, and what happens on a bad config value.
- **T2** — what actually starts the LSP language server, and what could make it hang or die mid-resolve.
- **T3** — trace CALLS-edge creation → resolution → where an unresolved edge silently drops.
- **T4** — trace a search query into ranked results, and where scores from different channels get combined.
- **T5** — how an embedder is picked/cached at query time, and what breaks if two repos use different models.

### How the numbers were measured
This is **measured, not estimated**. Each background agent's completion emits an
authoritative telemetry record (`subagent_tokens`, `tool_uses`, `duration_ms`).
Each agent's full JSONL transcript was then parsed offline (never loaded into the
overseer's context) to break down tool calls by name, assistant turns, and the
input/output/cache token split.

### Validity controls
- **Same everything but tools.** Identical model tier, repo, and ticket text.
- **Control purity verified.** Agent B's transcript shows **zero** `sutra_*` calls (13 Bash greps + 15 Reads).
- **Treatment actually used the treatment.** Agent A's transcript shows **10 Sutra tool calls** (9 `sutra_search`, 1 `sutra_get_callers`).
- **Equal quality.** Both agents reached the same correct conclusions with the same
  file:line evidence (e.g. `traversal.py:76-83` for the silent-drop of unresolved
  edges, `fusion.py` RRF for score combination, `lsp_resolver.py:61` for the
  un-timed `initialize` hang). The token delta is therefore an efficiency delta,
  not a quality trade-off.

---

## 2. Results

| Metric | WITH Sutra (A) | WITHOUT Sutra (B) | Δ | Sutra savings |
|---|--:|--:|--:|--:|
| **Reported tokens** (`subagent_tokens`) | 73,480 | 79,538 | −6,058 | **7.6%** |
| **Fresh tokens** (input+output, non-cache) | 50,145 | 110,026 | −59,881 | **54.4%** |
| Total context processed (incl. cache) | 1,668,789 | 2,201,018 | −532,229 | 24.2% |
| Tool calls | 22 | 28 | −6 | 21.4% |
| Assistant turns | 32 | 39 | −7 | 17.9% |
| Files read | 11 | 15 | −4 | 26.7% |
| Wall-clock | 110.2 s | 100.1 s | +10.1 s | −10.1% (slower) |

### Tool composition — *how* each agent navigated

| Agent A (WITH Sutra) | count | | Agent B (WITHOUT Sutra) | count |
|---|--:|---|---|--:|
| `sutra_search` | 9 | | `Bash` (grep/find/cat) | 13 |
| `Read` | 11 | | `Read` | 15 |
| `sutra_get_callers` | 1 | | | |
| `ToolSearch` (one-time load) | 1 | | | |
| **Total** | **22** | | **Total** | **28** |

---

## 3. Interpretation

**There are two honest ways to read "token savings," and they tell different parts
of the same story.**

1. **Reported tokens (`subagent_tokens`): 7.6% cheaper.** This is the harness's
   single headline number. It is conservative because it appears to discount cached
   context heavily — and both agents benefited from large, similar cache reads.

2. **Fresh tokens (non-cached input + output): 54.4% cheaper.** This is the number
   that reflects the *actual work the model had to do*. Agent B poured whole files
   into its context via 13 grep/cat sweeps and 15 reads, spending **100,515** input
   tokens. Agent A asked the index 10 targeted questions that returned exact
   symbols, signatures, and file:line locations, and only spent **42,252** input
   tokens — less than half. This is the core mechanism: **an index returns the
   needle; grep returns the haystack and makes the model read it.**

3. **Fewer steps too: 21% fewer tool calls, 18% fewer turns, 27% fewer files read.**
   Sutra collapsed the "where is this even defined?" search loop. Agent B needed
   grep→read→grep→read cycles to walk the call graph by hand; Agent A walked it with
   one `sutra_get_callers` call.

### The one place Sutra lost: wall-clock (+10s, ~10% slower)
MCP round-trips and a one-time `ToolSearch` to load the tools cost real latency.
Fewer, heavier tool calls (semantic search + graph walk) each take longer than a
local grep. **So Sutra trades a little wall-clock latency for a large reduction in
token/context load and steps.** On metered token cost and on context-window
pressure — the things that actually limit long agent sessions — Sutra wins clearly;
on raw speed for a single small task, plain grep is marginally faster.

---

## 4. Caveats & honest limits

- **n = 1 run of 5 tickets.** This is a directional demonstration, not a statistical
  study. Re-running would shift the exact numbers (agent behavior is stochastic).
- **Self-referential codebase.** The target repo *is* Sutra. It's the fair choice
  (only repo both indexed and on disk here), but a larger unfamiliar codebase would
  likely *widen* Sutra's lead, because grep's haystack grows while an index lookup
  stays flat.
- **`subagent_tokens` formula is opaque.** It is the harness's reported figure; its
  internal weighting of cache vs. fresh tokens is not documented, so the 7.6% and
  54.4% numbers are two different real lenses rather than a contradiction.
- **Cache dynamics.** Both agents ran with warm-ish caches; the "total context"
  figure is dominated by cache reads and is the least comparable metric.
- **Equal-quality claim is qualitative.** Both answers were independently correct,
  but no separate grader scored them.

---

## 5. Bottom line

On five realistic "trace this / debug this" tickets, giving the agent Sutra:
- cut the **fresh reasoning tokens by ~54%**,
- cut **tool calls by ~21%** and **files read by ~27%**,
- at **equal answer quality**,
- for a **~10% wall-clock latency cost**.

The savings come from one thing: **Sutra hands the agent the exact symbol instead of
making it read files to find it.** The bigger and less familiar the codebase, the
more that gap should grow.

*See `dashboard.html` for the visual breakdown, and `data.json` for the raw
measured figures.*
