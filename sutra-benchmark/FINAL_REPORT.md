# Sutra Performance Benchmark — Final Report (with actual API cost)

**Date:** 2026-07-08
**Prepared for:** sharing / independent evaluation of Sutra
**One-line result:** On 15 fresh code-tracing tickets across three repos, giving an
AI agent the **Sutra** code index produced **equal-quality answers** but cost
**~67% more in actual API dollars** than a plain grep/read agent, while cutting
tool-calls/turns only on the mid-size repo. **Sutra is a latency/steps optimization
in some cases, not a token or cost saver.**

---

## 1. What was measured

For each of three repositories, two identical agents were run on the **same 5
fresh tracing/debugging tickets** (15 tickets total, all new — not reused from
earlier runs):

- **Agent A — WITH Sutra:** navigates via the Sutra MCP index (semantic search +
  call-graph tools), reading files only to confirm.
- **Agent B — WITHOUT Sutra:** navigates with grep / find / read only.

Everything else is held constant:

| Control | Value |
|---|---|
| Model | `claude-sonnet-5` (both agents, every run) |
| Repos & commits | sutra @ `6a1a33a` · frappe/frappe @ `1eeb4a5` · langgenius/dify @ `d67123e` |
| Checkout alignment | Each on-disk checkout is the **exact indexed commit** (sutra via a fresh git worktree; dify/frappe checked out to the indexed SHA) |
| Ticket wording | Identical between A and B |
| Parallelism | All 6 agents run concurrently |

**Validity (verified from transcripts):** the WITHOUT agents made **0** `sutra_*`
tool calls; the WITH agents made 10 / 13 / 16 sutra calls. Both sides reached
correct, deep answers on every ticket (spot-checked: both traced the frappe
scheduler's silent-skip paths, the `check_safe_sql_query` weak allowlist, and
dify's `RerankModelRunner` score-threshold + top-k cut identically).

### How cost is computed (measured, not estimated)

Each agent's full JSONL transcript is parsed for the exact token counts the API
reported per turn: `input_tokens`, `output_tokens`, `cache_read_input_tokens`,
`cache_creation_input_tokens`. Cost is those counts × published Anthropic rates:

| Token type | Rate (Sonnet 5, standard) | Rate (Sonnet 5, intro¹) |
|---|--:|--:|
| Input (uncached) | $3.00 / MTok | $2.00 / MTok |
| Output | $15.00 / MTok | $10.00 / MTok |
| Cache read (0.1×) | $0.30 / MTok | $0.20 / MTok |
| Cache write, 5-min TTL (1.25×) | $3.75 / MTok | $2.50 / MTok |

¹ Introductory Sonnet-5 pricing ($2/$10) is in effect through 2026-08-31 and is
what actually bills today; standard ($3/$15) is the durable rate. Multipliers
(cache read 0.1×, cache write 1.25×) confirmed from Anthropic's prompt-caching
docs. Cost = the subagent's own token usage; it counts against your Claude usage
(as dollars on API/console billing, or as rate-limit consumption on a
subscription).

---

## 2. The 15 tickets

**sutra** (the code-index tool itself, ~1.9k symbols):
- **S1** — trace how the indexer selects the output backend (Postgres graph vs JSON artifact).
- **S2** — trace how a moniker string is parsed back into parts; where a malformed moniker raises.
- **S3** — when a symbol's body changes but its signature doesn't, trace body-hash computation and where a stale embedding could be reused on re-index.
- **S4** — trace how the MCP server loads a repo's artifact at startup; where a missing/corrupt field fails.
- **S5** — trace BM25 tokenization/scoring; where an empty or stopword-only query is handled.

**frappe** (low-code framework behind ERPNext, ~13.4k symbols):
- **F1** — trace DocType field-metadata load/cache; where a schema change invalidates it.
- **F2** — a scheduled job isn't firing: trace scheduler tick → dispatch; where a job is silently skipped.
- **F3** — trace how a Query Report runs user filters/SQL; where SQL injection is guarded (or not).
- **F4** — trace how an uploaded file is stored and served; where private-file / path-traversal is checked.
- **F5** — trace how `frappe.cache()` wraps Redis; where invalidation happens on a document write.

**dify** (LLMOps platform, ~74.8k symbols):
- **D1** — trace how a workflow variable template (`{{#node.field#}}`) is parsed/resolved; where an unresolved ref fails.
- **D2** — trace how conversation history is loaded and truncated to the model context window.
- **D3** — trace how per-app API rate limiting/quota is enforced; where the counter is checked/decremented.
- **D4** — trace how retrieval applies reranking + top-k; where below-threshold results are dropped.
- **D5** — trace how a file/image input is validated and passed to a multimodal model; where an unsupported type is rejected.

---

## 3. Results

### Per repo (A = WITH Sutra, B = WITHOUT)

| Repo | Metric | WITH (A) | WITHOUT (B) | Sutra effect |
|---|---|--:|--:|--:|
| **sutra** (1.9k) | Fresh tokens | 49,839 | 36,454 | +36.7% |
| | Tool calls | 22 | 20 | +10.0% |
| | Assistant turns | 32 | 32 | 0.0% |
| | Wall-clock | 113.1 s | 109.0 s | +3.8% |
| | **Cost (std)** | **$2.094** | **$0.935** | **+124%** |
| | Cost (intro) | $1.396 | $0.623 | +124% |
| **frappe** (13.4k) | Fresh tokens | 53,172 | 31,705 | +67.7% |
| | Tool calls | 32 | 48 | −33.3% |
| | Assistant turns | 47 | 76 | −38.2% |
| | Wall-clock | 168.9 s | 233.6 s | −27.7% |
| | **Cost (std)** | **$2.149** | **$1.680** | **+27.9%** |
| | Cost (intro) | $1.433 | $1.120 | +27.9% |
| **dify** (74.8k) | Fresh tokens | 67,000 | 48,011 | +39.6% |
| | Tool calls | 44 | 48 | −8.3% |
| | Assistant turns | 71 | 75 | −5.3% |
| | Wall-clock | 290.1 s | 298.0 s | −2.7% |
| | **Cost (std)** | **$3.012** | **$1.733** | **+73.7%** |
| | Cost (intro) | $2.008 | $1.156 | +73.7% |

### Aggregate (all 3 repos, 15 tickets)

| | WITH Sutra | WITHOUT Sutra | Sutra effect |
|---|--:|--:|--:|
| **Total cost (standard $3/$15)** | **$7.26** | **$4.35** | **+66.8%** |
| **Total cost (intro $2/$10)** | **$4.84** | **$2.90** | **+66.8%** |

### Tool composition (how each agent navigated)

| Repo | WITH Sutra | WITHOUT Sutra |
|---|---|---|
| sutra | 10 sutra_search · 9 Read · 2 Bash · 1 ToolSearch | 10 Bash · 10 Read |
| frappe | 13 sutra_search · 17 Read · 1 Bash · 1 ToolSearch | 29 Bash · 19 Read |
| dify | 12 sutra_search · 3 get_symbol · 1 get_callers · 5 Read · 22 Bash · 1 ToolSearch | 44 Bash · 4 Read |

---

## 4. Reading the numbers

**Cost — the clearest, most consistent signal.** Sutra cost **more real money on
every repo** (+124%, +28%, +74%; +67% aggregate). Two things drive this:

1. **Denser context per query.** `sutra_search` returns ranked symbols with
   signatures, docstrings, and provenance — token-rich payloads that become
   `cache_creation` tokens billed at the **1.25× premium**. Grep returns compact
   `file:line: match` text.
2. **MCP tool-loading overhead.** The WITH agent loads the Sutra tool schemas
   (via ToolSearch) into its prompt, which is then re-read as `cache_read` on
   every subsequent turn. On the tiny sutra repo this fixed overhead dominates —
   hence the +124% there, where the grep baseline is only $0.94.

**Fresh tokens — sutra higher on all three** (+37% / +68% / +40%), for the same
reasons.

**Tool calls, turns, wall-clock — a real but repo-dependent win.** Sutra's
advantage here showed up **clearly only on frappe** (−33% tool calls, −38% turns,
−28% wall-clock). On dify it was marginal (−8% / −5% / −3%) and on the tiny sutra
repo it was absent (tools +10%, turns 0%, +4% slower). So the "fewer round-trips"
benefit is genuine on mid-size codebases but was **not universal in this run**.

**Quality — equal.** No run traded correctness for cost; both agents produced
deep, file:line-backed answers, and each side independently surfaced findings the
other did (e.g. the grep agent found an extra uncaught-`KeyError` on sutra's
artifact load).

### Bottom line for evaluators
- If you are optimizing **API cost / token spend**: Sutra is a **net negative** on
  these read-only tracing tasks — expect to pay more, roughly **1.3×–2.2×**.
- If you are optimizing **wall-clock latency or number of agent round-trips** on a
  **mid-to-large, unfamiliar** codebase: Sutra can help (best case here: frappe,
  ~28% faster in ~⅔ the turns), but the effect is **variable and shrinks on very
  small repos**, where MCP overhead dominates.
- **Answer quality** was equivalent in every case.

---

## 5. Caveats (please read before citing)
- **n = 1 per cell.** Six agent runs, one ticket-set per repo. Agent behavior is
  stochastic; the *cost* direction was consistent across all three repos, but the
  *magnitude* of the tool-call/turn savings varied a lot and should not be treated
  as precise. A defensible claim needs ≥3 trials per cell with medians.
- **Cost includes Sutra's MCP tool-loading overhead**, which is a real, inherent
  cost of using the integration — but it is a *fixed* overhead, so it penalizes
  small repos and short tasks disproportionately.
- **Read-only tracing tasks only.** This measures "find and explain code," not
  editing, refactoring, or multi-file changes, where the tradeoff may differ.
- **Cost = the subagent's own token usage** at Sonnet-5 rates, 5-minute cache TTL
  assumed (1.25× writes). Rates confirmed from Anthropic docs on 2026-07-08.
- **"Total context processed" is a work metric, not a bill** — it double-counts the
  cached prefix re-read each turn. The dollar figures above are the real cost.

*Raw measured data: `fresh_run_data.json` (in the session scratchpad, values
transcribed above). Visual comparison incl. billed cost: `consolidated_dashboard.html`.*
