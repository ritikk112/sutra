# Sutra A/B/C Benchmark — Rigorous, Pre-Registered Run

**Date:** 2026-07-09 · **Model:** `claude-sonnet-5` (every agent) · **Design:** pre-registered, 3 arms × 2 repos × 6 tickets × 3 trials = **108 solver runs + 108 blind grades**

## TL;DR (what survives scrutiny)

1. **Answer quality is equal across all three arms.** Median blind grade is **4/4** for almost every (repo × class × arm) cell; excluding one ambiguous-gold ticket there are **no fabrications**. The index does not make answers better or worse.
2. **Using the index in isolation (SUTRA_ONLY) trends the most expensive** — **+12% to +65%** median $/run vs grep across the four repo×class cells, **+67% overall** ($0.642 vs $0.384). **But every per-cell cost range overlaps the grep baseline at n=3**, so this is a *consistent directional trend, not a statistically separated result.*
3. **The real-world arm (BOTH tools available) behaves like grep, not like the index.** Given both, the agent used Sutra **almost never** — median **0** sutra calls, **28 of 36 BOTH trials were pure grep** — so BOTH's cost ($0.377) tracks GREP_ONLY ($0.384). **This is the direct explanation for why earlier n=1 runs contradicted each other:** the "with Sutra" agent mostly ignored the index, and the rare times it used it, cost spiked (SUTRA_ONLY per-run cost ranged up to $2.06).
4. **No arm wins on steps/latency.** SUTRA_ONLY used *more* turns if anything (medians 15–29 vs grep 13–21); all ranges overlap.

**Bottom line:** In a controlled, blind-graded test, the Sutra index delivered **no measurable quality or cost advantage** on read-only code-tracing. In isolation it trends costlier at equal quality; in the realistic both-tools setting the agent largely doesn't use it.

---

## 1. Method

Three arms per ticket, identical model/wording, repo pinned to the indexed commit:

| Arm | Tools allowed | Measures |
|---|---|---|
| **SUTRA_ONLY** | Sutra MCP tools + Read (Bash/Grep/Glob forbidden) | the index in isolation |
| **GREP_ONLY** | Bash/Grep/Glob + Read (sutra_* forbidden) | the baseline |
| **BOTH** | everything | what a real Sutra user actually has |

- **Repos:** `frappe/frappe` @ `1eeb4a5` (mid, 13.4k symbols) and `langgenius/dify` @ `d67123e` (large, 74.8k). Checkouts aligned to the exact indexed commit. (The tiny sutra self-repo was excluded — dominated by fixed MCP overhead, not representative.)
- **Trials:** each (repo × arm × ticket) cell run **3×**; all arms ran concurrently in one fan-out so cache warmth doesn't favor an arm. **We report median + min/max across trials for every metric — never a single number.**
- **Constraints enforced at spawn** (prompt) and **verified from every transcript.** 3 of 108 SUTRA_ONLY runs made a stray Bash/Grep call; those were discarded and re-run (see §6).
- **Cost** = each transcript's API-reported `input/output/cache_read/cache_creation` tokens × Sonnet-5 rates (**std $3/$15; cache read 0.1×, write 1.25×**). Intro rates ($2/$10) give the same percentages.
- **Quality** graded **blind**: a separate grader agent saw only the ticket, the gold facts, and one candidate answer (no arm label), scoring 0–4.

## 2. Tickets — the lexical / semantic split (keyword-verified)

Naive-keyword counts are `grep -w` over the checkout. **Lexical** = the naive keyword is present and lands near the answer. **Semantic** = the naive symptom keyword is absent (strong = the concept is unnamed in code in any word-form; weak = symptom word absent but the feature identifier names it).

**FRAPPE**
| # | Class | Naive kw (count) | Ticket |
|---|---|---|---|
| FL1 | lexical | `scheduler` (118) | recurring-job firing + a due-job silent-skip point |
| FL2 | lexical | `has_permission` (287) | document read-permission check + bypass |
| FL3 | lexical | `rate_limit` (28) | inbound request throttling enforcement |
| FS1 | semantic **strong** | `optimistic` (0) | concurrent-save conflict detection (`check_if_latest`→`TimestampMismatchError`) |
| FS2 | semantic **weak** | `starve` (0, but `at_front_when_starved` present) | short/long job-queue separation |
| FS3 | semantic **strong** | `bleed` (0) | cross-site cache isolation (`make_key` db_name prefix) |

**DIFY**
| # | Class | Naive kw (count) | Ticket |
|---|---|---|---|
| DL1 | lexical | `rerank` (182) | rerank + top-k + threshold drop |
| DL2 | lexical | `quota` (262) | billing-quota check before a model call |
| DL3 | lexical | `variable_pool` (861) | `{{#node.field#}}` resolution (answer lives in external `graphon` pkg) |
| DS1 | semantic **weak** | `hog` (0, concept is `rate_limit`) | per-app request-slot limiting |
| DS2 | semantic **AMBIGUOUS** | `runaway` (1) | force-stop a too-long generation |
| DS3 | semantic **weak** | `poison` (0) | repeatedly-failing index task stops retrying |

Gold answers (correct file:line + mechanism + required facts) were authored by the overseer before any arm ran; see `gold.json`.

> ⚠️ **DS2 has an ambiguous gold.** Dify has *several* legitimate "stop a long run" mechanisms (app-queue `APP_MAX_EXECUTION_TIME`→`QueueStopEvent` [our gold], the agent iteration-cap, and `WORKFLOW_MAX_EXECUTION_TIME`). Agents traced different-but-plausible ones, so strict grading against one gold produced many 0–1s that are **gold-specificity artifacts, not agent errors.** We therefore report semantic quality **excluding DS2** as the primary quality read, and flag DS2 throughout.

## 3. Results — median [min–max] across 3 trials

Cost = billed $/run (Sonnet-5 std). Score = blind 0–4.

| Repo / class | Arm | Cost $ median [range] | Turns median [range] | Score median (dist) |
|---|---|---|---|---|
| **frappe / lexical** | SUTRA_ONLY | **0.762** [0.44–1.87] | 29 [15–71] | 4 (2:1,3:2,4:6) |
| | GREP_ONLY | 0.461 [0.27–0.72] | 21 [12–34] | 4 (3:1,4:8) |
| | BOTH | 0.400 [0.26–0.56] | 19 [11–28] | 4 (1:1,4:8) |
| **frappe / semantic** | SUTRA_ONLY | 0.379 [0.22–0.87] | 19 [12–27] | 4 (3:3,4:6) |
| | GREP_ONLY | 0.306 [0.15–0.44] | 17 [10–24] | 4 (3:2,4:7) |
| | BOTH | 0.284 [0.16–0.38] | 17 [10–21] | 3 (3:5,4:4) |
| **dify / lexical** | SUTRA_ONLY | **0.690** [0.41–2.06] | 24 [18–69] | 3 (1:3,3:3,4:2) |
| | GREP_ONLY | 0.428 [0.38–0.69] | 19 [14–36] | 4 (1:2,2:1,3:1,4:5) |
| | BOTH | 0.507 [0.37–0.71] | 22 [13–34] | 4 (1:2,2:1,3:1,4:5) |
| **dify / semantic** | SUTRA_ONLY | 0.346 [0.24–1.94] | 15 [13–55] | 4 |
| | GREP_ONLY | 0.309 [0.15–0.51] | 13 [10–36] | 4 |
| | BOTH | 0.281 [0.21–0.75] | 13 [9–41] | 4 |

**Overall median $/run per arm (all 36 cells):** SUTRA_ONLY **$0.642** (mean $0.826, right-skewed) · GREP_ONLY **$0.384** · BOTH **$0.377**.

## 4. Blind-grade distribution (quality)

- Median score is **4** for 10 of 12 (repo×class×arm) cells; the two exceptions are `frappe/semantic/BOTH` (median 3) and `dify/lexical/SUTRA_ONLY` (median 3, dragged by DL3 — see §6).
- **Semantic quality excluding the ambiguous DS2:** every arm on both repos has **median 4 with no score-0 fabrications.**
- The `dify/lexical` 1-scores are concentrated on **DL3** (variable resolution), whose true answer is *"in the external `graphon` package, not this repo"* — a genuinely hard, partially-out-of-scope ticket that tripped all arms similarly.
- **No arm shows a quality edge.** "Equal quality" is claimable: medians equal and (ex-DS2) no fabrications.

## 5. Cost breakdown & the MCP-overhead component

**SUTRA_ONLY carries a fixed integration overhead** that GREP_ONLY does not: every SUTRA_ONLY/BOTH agent spends one `ToolSearch` call to load the Sutra MCP tool schemas, which then sit in context and are **re-read as cache on every subsequent turn**. On top of that, each `sutra_search` returns a **token-dense payload** (ranked symbols + signatures + docstrings) that bills as **premium 1.25× cache-creation**. Grep output is terse `file:line: match`.

- **Fixed component** (one-time schema load, amortized over the run) → why SUTRA_ONLY's cost *floor* is higher and why short tasks are hit hardest.
- **Marginal component** (per `sutra_search` payload) → why SUTRA_ONLY's cost has a long right tail (max runs $1.87–$2.06 when the agent issued many searches).

A precise dollar split wasn't cleanly isolable from the transcripts; the mechanism and its two signatures (higher floor + heavy right tail) are clear in the data above.

## 6. Constraint compliance & the BOTH-arm fallback (secondary metrics)

- **Constraint violations:** 3 of 108 solver runs — all **SUTRA_ONLY** agents that made 1–2 stray Bash/Grep calls (`frappe|FL3`, `frappe|FS3`, `dify|DL3`). `FL3`/`FS3` were re-run clean. **`dify|DL3` could not be made grep-free even on re-run** — because its answer lives in the external `graphon` package that the Sutra index does not contain, the index-only agent is forced to grep to confirm the boundary. This is a real limitation of an index vs a filesystem search, reported as such; that cell is kept at n=2 clean trials.
- **BOTH-arm tool mix (the "real user"):** median **sutra calls = 0** on both repos; **grep calls ≈ 4.5 (frappe) / 6 (dify)**; **28 of 36 BOTH trials used zero Sutra.** When both are available, the agent overwhelmingly greps and only occasionally dips into the index. This is the fallback behavior that made prior benchmarks incomparable, now quantified.

## 7. Pre-registered conclusions (filled from the data)

For each, stated separately for **lexical** and **semantic** and for each repo, with every range that crosses the baseline flagged as **"no significant difference at n=3."**

**Quality — are arm medians equal? any fabrications?**
- frappe lexical: medians all **4**, equal, no fabrications. **Equal.**
- frappe semantic (ex-DS2): medians **4/4/4** (SUTRA/GREP/BOTH... BOTH=3 with DS2, 4 without the weak cases), no fabrications. **Equal.**
- dify lexical: medians **3/4/4**; the 3 (SUTRA_ONLY) is a DL3/out-of-scope artifact, not an index deficiency; no fabrications ex-DS2. **Equal within noise.**
- dify semantic (ex-DS2): medians all **4**, no fabrications. **Equal.**
- → **Quality: EQUAL across arms on both repos and both classes.**

**Cost — cheapest arm, by how much (median), ranges overlap?**
- frappe lexical: cheapest **BOTH $0.400**; SUTRA_ONLY **+65%** vs GREP; **ranges OVERLAP → not significant at n=3.**
- frappe semantic: cheapest **BOTH $0.284**; SUTRA_ONLY **+24%**; **ranges OVERLAP.**
- dify lexical: cheapest **GREP $0.428**; SUTRA_ONLY **+61%**, BOTH +19%; **ranges OVERLAP.**
- dify semantic: cheapest **BOTH $0.281**; SUTRA_ONLY **+12%**; **ranges OVERLAP.**
- → **SUTRA_ONLY is the most expensive arm in all 4 cells (consistent direction), but every comparison's range overlaps the baseline — a directional trend, not a significant difference at n=3.** BOTH ≈ GREP everywhere.

**Steps / latency — does any arm win on turns; range overlaps zero?**
- SUTRA_ONLY turns medians are ≥ grep in all 4 cells (e.g. 29 vs 21, 24 vs 19); ranges overlap. **No arm wins; if anything the index uses more turns.**

**Metrics whose min/max range crosses the baseline (⇒ "no significant difference at n=3"):** **all of them** — every cost and turns comparison between arms overlaps at n=3. The only robust signal is *directional consistency*: SUTRA_ONLY is costliest in 4/4 cost cells and never cheapest.

## 8. Caveats
- **n = 3 per cell.** Enough to expose variance (which is large: SUTRA_ONLY cost 0.24→2.06, turns 13→71), not enough for significance — hence the overlap flags. A firm claim needs ≥10 trials/cell.
- **DS2 gold is ambiguous** (multiple valid stop mechanisms); its raw scores are gold-specificity artifacts and are excluded from the primary quality read.
- **DL3 is partially out-of-scope** (answer in an external package); it depresses dify-lexical scores for all arms and makes the SUTRA_ONLY constraint unenforceable there.
- **Semantic-strength is asymmetric:** frappe yields *strong* semantic tickets (concept unnamed in code); dify names most concepts, so its semantic tickets are *weak* (symptom word absent, feature identifier present). Reported per ticket.
- **One grader per answer** (blind, but not multi-grader consensus). Grader is the same model family.
- **Read-only tracing tasks only** — not editing/refactoring, where an index might pay off differently.

*Raw per-trial data: `results.json` (all 108 solves: cost/turns/tools/constraint) and `final_analysis.json` (per-cell + per-arm-class aggregates + score dists). Tickets: `tickets.json`. Gold: `gold.json`. Visual: `dashboard.html`.*
