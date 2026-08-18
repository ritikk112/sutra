# Does the Sutra code-index beat plain grep for an AI agent? — Rigorous single-repo A/B/C (repo = *sutra itself*)

**Pre-registered, blind-graded benchmark. 1 repo · 6 tickets · 3 arms · 6 trials/cell · 108 solves + 108 blind grades · 0 constraint violations · 0 fabrications.**

> **TL;DR.** Giving a `claude-sonnet-5` agent the Sutra MCP index produced **no quality gain** over plain grep on read-only code-tracing tasks, and **cost 75% more** (significant by sign test across 6/6 tasks). When the agent had *both* grep and the index and could choose (the realistic "power-user" setup), it used the index **0 times out of 36 trials** — so the real-world arm was indistinguishable from grep. These results **replicate the earlier 2-repo (frappe + dify) run** and are now sharper at n=6.

---

## 1. What was measured — and what was *not*

This measures one thing precisely: **on a repo already checked out on disk, does the index help an agent trace/debug faster, cheaper, or more correctly than grep + read?** It deliberately does **not** measure the index's theoretical strengths — cross-repo search, onboarding to an unfamiliar codebase, or discovery when you don't know the identifier names. Read the conclusions with that scope in mind (§7).

## 2. Method (pre-registered, identical protocol to the 2-repo run)

| Parameter | Value |
|---|---|
| Repo | **`sutra`** (the index engine's own codebase), indexed commit `6a1a33a`, 1896 symbols / 1715 embeddings |
| Checkout | clean **git worktree pinned to `6a1a33a`** so SUTRA_ONLY (index) and GREP_ONLY see byte-identical code |
| Arms | **SUTRA_ONLY** (Sutra MCP + Read; Bash/Grep/Glob forbidden) · **GREP_ONLY** (Bash/Grep/Glob + Read; `sutra_*` forbidden) · **BOTH** (all tools — the realistic user) |
| Model | `claude-sonnet-5` (solvers **and** blind grader) |
| Tickets | 6 = 3 **lexical** (naive keyword present in code & lands) + 3 **semantic** (symptom keyword absent — must be reasoned) |
| Trials | **6 per cell** → 18 cells → 108 solves |
| Grading | **blind** 0–4 vs overseer-authored gold; grader never told which arm produced the answer |
| Cost | each transcript's exact input/output/cache-read/cache-creation tokens × Sonnet-5 rates ($3 / $15 per MTok; cache read 0.1×, cache write 1.25×) |
| Integrity | arm constraints verified **post-hoc from the tool calls in every transcript**: **0 violations** (SUTRA_ONLY used grep 0/36; GREP_ONLY used the index 0/36) |

**Gold** was authored by an 8-agent overseer "scout" workflow that cross-checked every fact with **both** grep (over the pinned worktree) and the Sutra index; every cited `file:line` was re-verified at `6a1a33a`, and every gold fact was confirmed reachable in the index (no dead-end like dify's DL3, whose answer lived in an un-indexed external package).

### The 6 tickets

| ID | Class | Naive grep | Ticket → gold mechanism |
|---|---|---|---|
| **SL1** | lexical | `rerank` (13) | Rerank + top-k cutoff → `reranker.py:87` cross-encoder `model.predict` → sort + `rescored[:top_k]`; gated by `rerank=True`, fed top-50 |
| **SL2** | lexical | `moniker` (199) | Stable id → `MonikerBuilder._build` 5-field string; kind-suffix + scope-prefix; `_disambiguate` `(N)` + keep-first backstop |
| **SL3** | lexical | `incremental` (28) | Changed-file re-index → `git_differ.changed_files` `git diff --name-status --diff-filter=ACDM --no-renames`; A/M re-embed, D removed |
| **SS1** | semantic | combine/blend/interleave = **0** | Merge 3 channel rankings → `rrf_fuse` reciprocal rank fusion, Σ 1/(k+rank), k=60, ranks assigned by fusion |
| **SS2** | semantic | isolation/tenant/leak = **0** | Repo-A can't return repo-B → per-repo `ServingUnit` via `registry.get`; one `ArtifactSnapshot` per pipeline; physical partitioning, not a WHERE filter |
| **SS3** | semantic | launch/spawn/"language server" = **0** | Cross-file call resolution → pyright started at index time from `Indexer.index()→resolver.resolve()`; `subprocess.Popen(pyright-langserver)`; opt-in `--resolver lsp` |

---

## 3. Result — quality (blind 0–4)

**Identical across all three arms. No arm fabricated or found a wrong root cause in any of the 108 solves.**

| Arm | Mean | Median | Score distribution | Min |
|---|---|---|---|---|
| SUTRA_ONLY | **3.67** | 4.0 | `{3:12, 4:24}` | 3 |
| GREP_ONLY | **3.69** | 4.0 | `{2:1, 3:9, 4:26}` | 2 |
| BOTH | **3.69** | 4.0 | `{2:2, 3:7, 4:27}` | 2 |

- The lowest score anywhere was **2** (3 trials total, on SL2/SS2); there were **no 0s or 1s** — every arm reliably located the correct mechanism.
- Per-cell quality was tied on 4 of 6 tickets. The only divergences: **SS3** (LSP kickoff) SUTRA_ONLY median 3.0 vs 4.0 for grep/both, and **SL2** (moniker) 3.5 vs 4.0 — i.e. the index-only agent was slightly *less* likely to surface the bonus edge-case, never *more*. On **SS2** all three arms tied at 3.0 (the hardest ticket).

> **Conclusion Q (pre-registered):** *No measurable quality advantage from the index.* Confirmed — medians equal, means within 0.02, and the index-only arm was marginally weaker (not stronger) on 2 of 6 tickets.

## 4. Result — cost (Sonnet-5 $/run)

| Arm | Median $/run | Range [min–max] | Total (36 runs) | vs GREP |
|---|---|---|---|---|
| GREP_ONLY | **$0.269** | [0.201 – 0.467] | $10.24 | — (baseline) |
| BOTH | **$0.277** | [0.155 – 0.429] | $9.74 | +3% |
| SUTRA_ONLY | **$0.471** | [0.243 – 0.981] | $16.88 | **+75%** |

**Per-cell direction (the significance that matters):** SUTRA_ONLY cost more than GREP in **all 6 tickets (6/6)** → two-sided **sign test p = 0.031**. So while the *pooled* ranges overlap (a conservative "magnitude not significant"), the *direction* — "the index-only agent costs more" — is statistically significant, and stronger than the n=3 2-repo run could show.

| Ticket | SUTRA_ONLY | GREP_ONLY | BOTH | SUTRA vs GREP |
|---|---|---|---|---|
| SL1 | $0.320 | $0.235 | $0.191 | +37% |
| SL2 | $0.479 | $0.279 | $0.284 | +72% |
| SL3 | $0.325 | $0.264 | $0.260 | +23% |
| SS1 | $0.361 | $0.258 | $0.273 | +40% |
| SS2 | $0.520 | $0.390 | $0.310 | +33% |
| **SS3** | **$0.610** | $0.259 | $0.277 | **+136% (ranges disjoint)** |

The gap is largest on the semantic tickets, where the index-only agent had to *navigate the graph* (median 6–8 `sutra_*` calls, 18–23 turns) rather than land the answer with one grep. Per class: SUTRA_ONLY is **+37%** on lexical, **+76%** on semantic.

> **Conclusion C (pre-registered):** *SUTRA_ONLY is not cheaper.* Confirmed and strengthened — it is more expensive on every task (sign test p=0.031), driven by more turns / round-trips.

## 5. Result — the decisive finding: BOTH never used the index

When the agent had grep **and** the index and was free to pick (the realistic setup):

- **0 / 36** BOTH trials made a single `sutra_*` call. Every one went **pure grep** (36/36 used grep).
- Result: **BOTH ≈ GREP** — +3% cost (overlapping ranges, sign test 3/6, p=1.0), identical quality.

This is **more extreme than the 2-repo run** (there, 28/36 BOTH trials were pure grep; here it's 36/36) — and it happened on **Sutra's own source code**, where the index is a perfect match for the material. It is the direct, reproducible explanation for why earlier single-shot (n=1) comparisons contradicted each other: a "real user" agent with both tools behaves like a grep user, so whether the index "wins" is pure noise.

> **Conclusion B (pre-registered):** *The realistic BOTH arm is indistinguishable from grep.* Confirmed — because current agent behavior defaults to grep when both are available. (This is a statement about agent tool-selection behavior, not only about index quality; a differently-prompted agent *instructed* to prefer the index would be a separate experiment.)

## 6. Turns / tool-calls

| Arm | Median turns | Median tool-calls |
|---|---|---|
| SUTRA_ONLY | 16 | 10 |
| GREP_ONLY | 11 | 6 |
| BOTH | 11–12 | 7 |

Index-only navigation consistently takes ~50% more turns — the mechanism behind the cost gap. No arm had a turn advantage from the index.

## 7. Honest limitations (what could change the answer)

- **Scope.** Read-only tracing on an **already-checked-out** repo, where grep is at its strongest. This says nothing about cross-repo search, unfamiliar-codebase onboarding, or discovery when identifier names are unknown — the index's design strengths, none of which this harness exercises.
- **Agent behavior, not just tool quality.** The BOTH=0/36 result reflects how *this* agent, with *this* prompt, chooses tools. Explicitly instructing the agent to prefer the index would test a different question.
- **Single model / single repo / n=6.** One model (`sonnet-5`), one repo (its own), 6 trials/cell. The direction (cost↑, quality flat) now agrees across 3 repos, but effect *magnitudes* still carry per-cell variance (pooled ranges overlap).
- **Cost includes MCP overhead** — tool loading and multi-round-trip navigation are part of the measured cost, as they would be for a real user.
- **Grader.** Blind, with explicit gold facts, same model as solvers. A stronger grader could shift a few 3↔4 calls but not the equal-medians / zero-fabrication picture.

## 8. How this squares with the 2-repo run

| Finding | 2-repo (frappe+dify, n=3) | Single-repo (sutra, n=6) |
|---|---|---|
| Quality | equal, no fabrications (ex-DS2) | equal, **0 fabrications** |
| SUTRA_ONLY cost | +67% (directional, ranges overlap at n=3) | **+75%, 6/6 cells, sign test p=0.031** |
| BOTH vs GREP | ≈ (index barely used: 28/36 pure grep) | ≈ (**index never used: 36/36 pure grep**) |
| Turn advantage | none | none |

Three repositories now agree. The claim that survives scrutiny is narrow and consistent: **for an agent tracing code in a repo it already has on disk, the Sutra index does not improve answer quality and increases cost, and agents given the choice fall back to grep.**

## 9. Reproducibility

All artifacts in `sutra-benchmark/rigorous_sutra/` (the 2-repo run in `sutra-benchmark/rigorous/` was left untouched):

| File | What |
|---|---|
| `tickets.json`, `gold.json` | the 6 tickets + verified ground truth |
| `scout_gold.js`, `scout_results.json` | the 8-agent gold-authoring workflow + its output |
| `abc_workflow.js` | the solve+grade fleet (108 solves + 108 grades) |
| `full_parse.py` | transcript → per-cell cost/turns/tools/constraint/score |
| `final_agg.py` | aggregation + pre-registered conclusions |
| `results.json`, `final_analysis.json`, `report_data.json` | raw per-trial → aggregates |
| `dashboard.html` | the shareable visual summary |

Repo pinned at `6a1a33a`; worktree at `…/scratchpad/sutra_src`. Constraint compliance and grade coverage: **108/108 solves parsed, 108/108 graded, 0 violations, 0 fabrications.**
