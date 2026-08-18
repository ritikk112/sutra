# Sutra Benchmark #2 — Large Repo (dify, 75k symbols)

**Date:** 2026-07-08
**Overseer:** main session (Opus 4.8)
**Target:** `langgenius/dify` @ `d67123e` (74,825 symbols, 47,816 embeddings) — the
on-disk checkout was fetched and aligned to the exact indexed commit before running,
so both agents saw identical code.

This is the follow-up to the first benchmark, which ran on the ~1.9k-symbol `sutra`
repo. The hypothesis going in: *the bigger the codebase, the more an index should
beat grep, because grep's search cost scales with the repo while an index lookup
stays flat.* The result is more nuanced — and more interesting — than that.

---

## 1. Method (identical to benchmark #1)

Two background agents, same model (claude-sonnet), same 5 vague tracing/debugging
tickets, same repo/commit. Only tool access differed:

- **Agent A (WITH Sutra):** Sutra MCP index as primary navigation (`repo=langgenius/dify`).
- **Agent B (WITHOUT Sutra):** grep / find / read only.

Telemetry is measured, not estimated: each agent's completion emits
`subagent_tokens`, `tool_uses`, `duration_ms`; transcripts were parsed offline for
the tool-by-tool breakdown and token split.

**Validity — verified from transcripts:**
- Agent B: **zero** `sutra_*` calls (56 Bash greps + 19 Reads). Clean control.
- Agent A: **26** Sutra calls (25 `sutra_search`, 1 `get_symbol`) + 13 Reads.
- **Equal quality:** both produced deep, correct findings on the same bugs — e.g.
  the uncaught `json.JSONDecodeError` in `fc_agent_runner.py:355` (agent turn dies
  on a malformed tool call), and the `APP_MAX_EXECUTION_TIME` force-stop that
  truncates SSE streams (`base_app_queue_manager.py:60-85`). Agent B independently
  surfaced an *additional* sharp bug (a NaN-skip that desyncs the embedding `zip`
  in `cached_embedding.py:72-90`). Neither agent was quality-starved.

---

## 2. Results

| Metric | WITH Sutra (A) | WITHOUT Sutra (B) | Sutra effect |
|---|--:|--:|--:|
| **Reported tokens** (`subagent_tokens`) | 126,881 | 115,465 | **+9.9% (more)** |
| **Fresh tokens** (input+output, non-cache) | 66,739 | 51,929 | **+28.5% (more)** |
| Total context processed (incl. cache) | 4,507,149 | 7,978,895 | **−43.5%** |
| Tool calls | 40 | 75 | **−46.7%** |
| Assistant turns | 56 | 109 | **−48.6%** |
| Files read | 13 | 19 | −31.6% |
| Wall-clock | 232.6 s | 429.6 s | **−45.9%** |

**Tool composition:** A = 25 `sutra_search` + 1 `get_symbol` + 13 Read + 1 ToolSearch.
B = 56 Bash (grep/find) + 19 Read.

---

## 3. The honest headline: it flipped, and that's the finding

On the small repo, Sutra saved **54% of fresh tokens**. On dify, Sutra used **28%
*more* fresh tokens** and ~10% more reported tokens. That is a real reversal, not
noise, and it's worth understanding rather than hiding:

- **Why Sutra's token count went *up*:** `sutra_search` returns dense payloads —
  each of 25 searches returned up to 10 ranked symbols with signatures, docstrings,
  and provenance. That is a lot of tokens injected per call. `grep` output is the
  opposite: compact `file:line: matching-text`. So *per call*, grep is cheaper on
  tokens than a semantic-search result.

- **Why Sutra still won everything else:** Agent B needed **75 tool calls across 109
  turns** to triangulate answers by hand (grep → read → grep → read). Agent A got
  there in **40 calls / 56 turns**. Every extra turn re-processes the whole growing
  conversation, which is why Agent B's **total context ballooned to 7.98M tokens
  (43% more than Sutra's 4.51M)** and its **wall-clock hit 7.2 minutes vs Sutra's
  2.3**.

So on a large repo the advantage *moved*: from "fewer tokens per query" to "far
fewer round-trips, far less cumulative compute, and half the wall-clock." The dense
per-call payload is the price Sutra pays to collapse the number of turns.

---

## 4. Scaling: benchmark #1 (1.9k symbols) vs #2 (75k symbols)

Sutra effect (% change vs the grep-only baseline; negative = Sutra used less):

| Metric | sutra repo (1.9k) | dify (75k) | Trend as repo grows |
|---|--:|--:|---|
| Tool calls | −21.4% | **−46.7%** | Sutra's lead **more than doubles** |
| Assistant turns | −17.9% | **−48.6%** | Lead **more than doubles** |
| Total context | −24.2% | **−43.5%** | Lead widens |
| Wall-clock | +10.1% (slower) | **−45.9% (faster)** | **Flips from cost to big win** |
| Fresh tokens | −54.4% (saved) | +28.5% (more) | **Flips the other way** |
| Reported tokens | −7.6% (saved) | +9.9% (more) | Flips |

**The consistent, monotonic signal is round-trips.** Sutra cut tool calls and turns
on *both* repos, and the cut **grew sharply with repo size** (−21% → −47% on calls,
−18% → −49% on turns). Wall-clock went from a 10% penalty to a 46% win. The token
axis is the one that flipped sign — because it depends on grep's file-reading
behavior (whole files on the small repo → huge input; compact greps + heavy caching
on the big repo → low fresh tokens), not on anything Sutra controls.

---

## 5. What to take away

- **Sutra's durable, scale-growing win is the agentic loop:** ~half the tool calls,
  ~half the turns, ~half the wall-clock, and 43% less total context on a real 75k-
  symbol codebase. That is the metric that governs whether a long agent session
  stays inside its context window and finishes in a reasonable time.
- **Sutra is not universally "fewer tokens."** On a large repo its rich search
  payloads cost more *fresh* tokens than terse grep output. If raw token-per-run is
  the only thing being optimized on a big repo, that's a real tradeoff to know about.
- **The wall-clock flip is the practical story:** the MCP round-trip overhead that
  made Sutra 10% slower on a trivial repo is dwarfed on a large one, where it comes
  out **46% faster** by avoiding dozens of grep-and-read cycles.

### Caveats
- **n = 1 per repo.** Directional, not statistical; agent behavior is stochastic.
- **Equal-quality is qualitative** — both answers were independently correct and
  deep, but no separate grader scored them. (Notably, the grep agent found one bug
  the index agent didn't, and vice-versa.)
- **`subagent_tokens` weighting is opaque** — treat "reported" and "fresh" as two
  real lenses, not a single truth.
- **Total-context is cache-dominated** — it best captures "total work done," but the
  cache accounting is the least directly comparable across runs.

*See `dashboard.html` for the visuals (dify results + the scaling comparison), and
`data.json` for raw measured figures. Benchmark #1 lives one directory up.*
