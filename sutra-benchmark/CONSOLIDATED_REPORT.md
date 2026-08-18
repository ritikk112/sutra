# Sutra Benchmark — Consolidated Report (3 repos)

**Date:** 2026-07-08
**Overseer:** main session (Opus 4.8)
**Design:** For each repo, two background agents (same model — claude-sonnet) were
handed the **same 5 vague tracing/debugging tickets**. Agent A navigated with the
Sutra MCP index; Agent B used only grep/find/read. Telemetry
(`subagent_tokens`, `tool_uses`, `duration_ms`) is measured from each agent's
completion; transcripts were parsed offline for tool breakdown and token split.

Three repos, spanning ~40× in size:

| Repo | Symbols | Commit | Role |
|---|--:|---|---|
| `sutra` | 1,896 | `6a1a33a` | tiny (self-repo) |
| `frappe/frappe` | 13,436 | `1eeb4a5` | mid |
| `langgenius/dify` | 74,825 | `d67123e` | large |

Every run: checkout aligned to the exact indexed commit; both agents verified from
transcripts (B never called a `sutra_*` tool; A actually used the index; on frappe,
neither agent touched the off-limits `first-bench` copy). Answer quality was
comparable in every run — both agents repeatedly found the same real bugs, and on
two runs each side independently found one the other missed.

---

## 1. Headline — read this first

**The first run (tiny self-repo) suggested Sutra saves ~54% of tokens. Two larger,
real-world repos did not reproduce that. The honest, corrected conclusion:**

> **Sutra is a speed-and-steps optimization, not a token optimization.** On real
> codebases it reliably cuts wall-clock time and the number of agentic turns — and
> that advantage *grows with repo size* — but it **costs more tokens**, because
> semantic-search results are token-dense compared to terse grep output.

The token "win" only appeared on the trivial 1.9k-symbol self-repo, where the
grep agent burned tokens reading whole files. On both real repos, Sutra used
**more** tokens, not fewer.

---

## 2. Raw results per repo

Sutra effect = % change vs the grep-only baseline. **Negative = Sutra used
less/fewer** (a win); positive = Sutra used more (a cost).

| Metric | sutra (1.9k) | frappe (13.4k) | dify (75k) |
|---|--:|--:|--:|
| Reported tokens | **−7.6%** | +25.0% | +9.9% |
| Fresh tokens (non-cache) | **−54.4%** | +55.9% | +28.5% |
| Total context (incl. cache) | −24.2% | −13.1% | −43.5% |
| Tool calls | −21.4% | −18.6% | −46.7% |
| Assistant turns | −17.9% | −30.0% | −48.6% |
| Files read | −26.7% | −13.0% | −31.6% |
| **Wall-clock** | **+10.1%** | **−31.1%** | **−45.9%** |

Absolute figures:

| | sutra A / B | frappe A / B | dify A / B |
|---|---|---|---|
| Reported tokens | 73,480 / 79,538 | 89,533 / 71,629 | 126,881 / 115,465 |
| Fresh tokens | 50,145 / 110,026 | 46,569 / 29,872 | 66,739 / 51,929 |
| Tool calls | 22 / 28 | 35 / 43 | 40 / 75 |
| Assistant turns | 32 / 39 | 49 / 70 | 56 / 109 |
| Wall-clock (s) | 110.2 / 100.1 | 170.3 / 247.0 | 232.6 / 429.6 |

(A = WITH Sutra, B = WITHOUT.)

---

## 3. What actually holds up across all three runs

### Robust (consistent on all 3 repos)
1. **Fewer agentic turns — monotonically stronger with size.** −17.9% → −30.0% →
   −48.6%. This is the cleanest signal in the whole dataset.
2. **Faster wall-clock, and the win grows — monotonically.** +10.1% (a *penalty* on
   the trivial repo, where MCP round-trip overhead dominates a quick task) → −31.1%
   → −45.9%. On real repos Sutra roughly cut wall-clock by a third to a half.
3. **Fewer tool calls** — true on all three (−21%, −19%, −47%), though the magnitude
   is not a clean line (frappe dipped).
4. **Less total context processed** — true on all three (−24%, −13%, −44%), also
   noisy in magnitude.
5. **Equal answer quality** — no run traded correctness for efficiency.

### Does NOT hold up
- **"Sutra saves tokens."** False on both real repos. Fresh tokens: −54% (tiny) but
  **+56% (frappe)** and **+28% (dify)**. Reported tokens flip positive too. The tiny
  repo was the outlier, not the rule.
- **"The advantage is cleanly monotonic across the board."** Only *turns* and
  *wall-clock* are strictly monotonic across the three sizes. Tool calls,
  total-context, and the token metrics are directionally consistent but bounce around
  in magnitude — expected at **n = 1 per repo**.

### Why the token axis flips
- `sutra_search` returns dense payloads — ranked symbols with signatures, docstrings,
  and provenance. `grep` returns compact `file:line: match` text. So per lookup, grep
  is cheaper on tokens.
- On the tiny self-repo the grep agent compensated by *reading whole files* (15 reads
  on a small tree → 110k input tokens), which is why Sutra looked token-cheap there.
- On bigger repos the grep agent stayed surgical (compact greps, heavy caching), so
  its fresh-token count dropped and Sutra's dense payloads came out higher.

### Why Sutra still wins time and turns
- The grep agent needs many more round-trips to triangulate by hand
  (grep → read → grep → read). Turns scale badly: 39 → 70 → 109 for grep vs
  32 → 49 → 56 for Sutra. Each extra turn re-processes the growing conversation and
  adds latency. Sutra collapses "where is this defined / who calls it" into one
  indexed lookup, so it finishes in fewer turns and less time.

---

## 4. Practical guidance

- **If you care about wall-clock latency or context-window pressure on real
  codebases → Sutra is a clear win**, and more so the bigger the repo.
- **If you are optimizing raw token spend per run → Sutra is not the lever** on
  non-trivial repos; its richer payloads cost more tokens than grep. (It may still be
  worth it for the time/turn savings, but go in with eyes open.)
- **On trivial repos** (a couple thousand symbols) the MCP overhead can make Sutra
  slightly slower and its token edge is repo-specific — the tooling shines as the
  codebase grows.

---

## 5. Caveats
- **n = 1 per repo.** Directional, not statistical. The non-monotonic magnitudes are
  the clearest sign that per-run variance is real; a defensible claim needs ≥3 trials
  per repo with medians.
- **Equal-quality is qualitative** — both answers were independently correct and
  deep, but no separate grader scored them.
- **`subagent_tokens` weighting is opaque** — "reported" and "fresh" are two real
  lenses, not one truth. Fresh tokens (non-cache input+output) best reflect model work.
- **Total context is cache-dominated** — best captures "total work," least directly
  comparable across runs.
- **frappe source** was a fresh shallow clone in the scratchpad (the on-disk copy was
  off-limits); it was aligned to the indexed commit, and first-bench was never touched.

*Per-repo detail: `dify/REPORT.md`, `REPORT.md` (sutra). Raw figures: each repo's
`data.json`. Visuals: `consolidated_dashboard.html`.*
