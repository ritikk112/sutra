# Sutra vs no-sutra — a fresh controlled benchmark (2026-08-19)

A clean-slate A/B measuring what the sutra index adds to a coding agent over
the tools every agent already has (Bash/Grep/Glob/Read). Authored, run and
analysed independently of any earlier benchmark: new tickets, new golds, new
metrics, new ticket-id namespace. No prior result was used to set a parameter,
justify a prediction, or anchor a comparison.

**Bottom line: the index makes agents navigate in fewer steps and invent fewer
symbol names. It does not make them more correct, and it makes them slower.
Its most dangerous failure is that its coverage boundary silently becomes the
agent's answer boundary.**

## Protocol

- **20 tickets, 4 classes x 5**, across pydantic (2,793 symbols), fastapi
  (2,055), sutra (687), flask (496), requests (344).
  - **A — blast radius (`BR`)**: "what breaks if I change X?", graded on
    precision *and* recall against a hand-verified call-site list, with decoys.
  - **B — cross-repo (`XR`)**: one question whose answer spans 3+ repos.
  - **C — grep-favorable (`GF`)**: deliberately authored to make the index
    lose — verbatim identifier, string literal, non-code file, module constant,
    index staleness.
  - **D — neutral trace (`NT`)**: ordinary bug tickets, authored without regard
    to either toolset. The experiment's own control class.
- **120 solver runs** (20 x 2 arms x 3 trials), claude-haiku, plus a 21-agent
  calibration pilot and 20 blind judges. 161 agents total.
- **Byte-identical prompts.** The only difference between arms is the tool
  list; verified by diffing the rendered prompts. Neither arm is told it is
  expected to win.
- **Tool-call budget of 6**, identical in both arms, fixed by the pilot and
  never tuned afterwards.
- **Judges see only final answers**, never traces. A trace from the index arm
  contains `mcp__sutra__*` calls in plain sight, so trace-level judging cannot
  be blinded. One sonnet judge per ticket grades all 6 answers shuffled and
  unlabelled, against a pre-authored gold.
- **Golds were built without the index.** No author, verifier or judge had
  sutra tools, so a gold can never be derived from the system it scores.

### Run hygiene

| check | result |
|---|---|
| control-arm sutra calls (contamination) | **0.00** |
| budget violations | sutra 7/60, control 10/60 (max +3 both) |
| concurrency at solver start (contention) | sutra 6, control 7 |
| index adoption, sutra arm | BR 15/15, XR 15/15, NT 13/15, GF 9/15 |
| judge gold disputes upheld | 0 of 20 (1 cosmetic signature note on XR1) |

## Registered predictions vs outcome

Stated before the run so it could falsify me. **Three of four failed.**

| class | predicted | measured (sutra − control) | held? |
|---|---|---|---|
| A blast-radius | sutra wins | −0.033 | **no** — tie |
| B cross-repo | sutra wins | −0.100 | **no** — control |
| C grep-favorable | control wins | −0.033 | **no** — tie (both ~ceiling) |
| D neutral | tie | +0.033 | yes |

Class D holding is the load-bearing one: the arms tie where nothing was
engineered to favour either, which is what says the harness itself is not
tilted.

## Where sutra performed BETTER

### 1. Fewer tool calls to reach the answer — the one solid win

| metric | sutra | control | paired delta | p |
|---|---|---|---|---|
| calls-to-gold (median) | **1** | 2 | −1.1 | **0.005** |

Significant, and consistent: sutra reaches the gold mechanism first on 11 of
20 tickets. The gap is widest on cross-repo work (1 vs 3.5), which is the one
place the index does something grep structurally cannot — search every repo in
a single call.

### 2. It invents fewer symbol names

`false_localization` counts an answer *confidently naming a specific wrong
symbol or file*. It is scored separately from "wrong" because a confident
wrong pointer sends an engineer to read the wrong file, whereas "I could not
determine this" merely fails.

| class | sutra | control |
|---|---|---|
| A blast-radius | **1/15 (7%)** | 5/15 (33%) |
| B cross-repo | 7/15 (47%) | 8/15 (53%) |
| C, D | 0/15 | 0/15 |
| **overall** | **8/60 (13%)** | 13/60 (22%) |

The control arm fabricated plausible-but-nonexistent symbols under budget
pressure — `_get_solved_dependency`, `Mount._solve_dependencies`,
`_before_request_funcs`. All confidently asserted; none exist.

**But this is not statistically established.** Paired across 20 tickets,
p = 0.24; the blast-radius subset is 5 tickets, Fisher p = 0.169. The effect
is in the right direction with a clean mechanism behind it, and it is
underpowered. Calling it proven would be overclaiming.

### 3. Slightly better call-site precision, no decoys

| | precision | recall | decoys claimed |
|---|---|---|---|
| sutra | **0.960** | 0.873 | **0** |
| control | 0.947 | **0.982** | 1 |

## Where sutra performed WORSE

### 1. It is slower — the most robust negative result in the run

| metric | sutra | control | paired delta | p |
|---|---|---|---|---|
| wall clock (median s) | 43.3 | **38.5** | +5.9 | **<0.001** |
| model time | 36.2 | **31.4** | +4.1 | **<0.001** |
| tool time | 8.2 | **5.0** | +2.5 | **0.002** |
| **time-to-first-gold** | 9.4 | 8.8 | −0.6 | 0.61 |

Sutra is slower on **19 of 20 tickets**. Contention is ruled out: concurrency
measured at each solver's start instant was 6 (sutra) vs 7 (control), so if
anything the control arm ran in the busier slice.

The last row is the important one. **Halving the call count does not halve the
time.** Calls-to-gold drops from 2 to 1 (p=0.005) while time-to-first-gold is
statistically flat (p=0.61) — each index call costs more wall-clock and returns
more text to process than the grep it replaced. Anyone quoting "the index
halves navigation" is quoting a call count and implying a clock.

### 2. It misses call sites, because the index boundary becomes the answer boundary

This is the most consequential finding, and it is mechanistic rather than
statistical.

Sutra's caller **recall is worse** (0.873 vs 0.982). The cause, verified by
querying the served index directly:

```
sutra_get_callers("...src/requests/utils.py select_proxy().")
  -> HTTPAdapter.get_connection
     HTTPAdapter.get_connection_with_tls_context
     HTTPAdapter.request_url
```

Three callers. The verified gold has **four** — the fourth is
`tests/test_utils.py:565`. A follow-up search confirms the root cause: the
requests index contains **no test symbols at all**; every result comes from
`src/requests/`. Tests are outside the index boundary, so `get_callers`
structurally cannot return them.

Measured sutra recall on that ticket: **0.75 — exactly 3/4.**

`get_callers` was used in 15/15 blast-radius sutra trials. The agent asks
"what calls this?", receives a clean, confident, complete-*looking* list, and
reports it. Grep finds the test because grep does not know what an index is.

For the specific question "what breaks if I change this signature", omitting
every test is close to the worst possible omission — tests are precisely what
breaks. The tool's own documentation says resolved edges are "a lower bound",
but nothing at the point of use signals that the answer is truncated.

### 3. Cross-repo correctness went the wrong way

Class B was predicted as sutra's strongest suit and it lost: 0.400 vs 0.500.
Both arms are poor here and both fabricate at ~50%. Multi-repo comparison is
simply hard, and searching all repos in one call gets the agent to candidate
symbols quickly without helping it reason about three codebases at once.

### 4. More context, more tokens

Context growth (cache-write) is +3,464 tokens per trial, p=0.060 — marginal,
directionally against sutra. Over the run: 1.87M vs 1.73M cache-write tokens
and 41.9 vs 36.3 minutes.

### 5. Module-level constants

The only clear loss inside the sutra-hostile class: GF4 (module constant),
0.83 vs 1.00, with the index used in 3/3 trials.

## Where it made no difference

**Correctness: none.** sutra 0.767 vs control 0.800 across 120 graded answers;
paired across 20 tickets the delta is −0.033 with **p = 0.42**. Per-ticket,
sutra is better on 4, control on 6, tied on 10.

This is the third independent attempt to find a correctness effect and the
third failure to find one — this time with a call budget specifically designed
to convert navigation speed into accuracy. It did not convert.

**The sutra-hostile class mostly failed to hurt it — because the agent stopped
using the index.**

| ticket | kind | sutra \| control | index used |
|---|---|---|---|
| GF1 | verbatim identifier | 1.00 \| 1.00 | 3/3 |
| GF2 | string literal | 1.00 \| 1.00 | 3/3 |
| GF3 | non-code file | 1.00 \| 1.00 | **0/3** |
| GF4 | module constant | **0.83** \| 1.00 | 3/3 |
| GF5 | index staleness | 1.00 \| 1.00 | **0/3** |

Handed a question about `pyproject.toml`, or about code newer than the index
snapshot, the agent simply reached for grep. Adoption drops to 9/15 on this
class against 15/15 on blast-radius. The index does not have to be good at
everything, because the agent routes around it — as long as it can tell it
should, which is exactly what GF4 shows it sometimes cannot.

## What the pilot caught

The calibration pilot is worth reporting because it nearly produced a fake
result. Caps of 3/6/12 were tried; cap 3 showed a **+0.67 sutra advantage**,
by far the largest separation.

It was an artifact. A prompt-stated "hard budget" is unenforceable, and it
failed *asymmetrically*: violations ran sutra 3/9 vs control 1/9, and on BR1 at
cap 3 the sutra arm used **14 tool calls against a cap of 3** and scored
correct while control obeyed at 3 and scored wrong. That is a 14-vs-3
handicap, not a measurement.

Selecting the cap that maximised the delta would have shipped that number as
the finding. Cap 6 was chosen instead as the only cap both *binding* (every
trial reached it) and *obeyed*. After hardening the budget language to name MCP
calls explicitly, main-run violations came out balanced at 7/60 vs 10/60.

## Verdict

1. **Correctness: no effect, again.** Three designs, three nulls. On tasks a
   competent agent can solve at all, the index does not change whether it
   solves them.
2. **Navigation: real in calls, absent in seconds.** −1.1 calls-to-gold
   (p=0.005) alongside a flat time-to-first-gold (p=0.61) and a *worse* wall
   clock (+5.9s, p<0.001, losing 19/20). If you are optimising an interactive
   loop, the index currently costs you time.
3. **Fabrication: promising, unproven.** 13% vs 22% overall, 7% vs 33% on
   blast-radius, with a clear mechanism — but p=0.24. This is the effect worth
   powering a future run on, and it is not yet a claim.
4. **The real risk is silent incompleteness.** `get_callers` returning 3 of 4
   call sites with no indication anything is missing is worse than a slow
   answer, because it is a *confident* answer. Fixing this — indexing tests, or
   marking results as partial at the point of use — matters more than any
   latency work.

**Worth installing** if you want fewer, better-targeted tool calls and care
about agents not inventing symbol names. **Not yet** if you are optimising
wall-clock latency, or if you rely on call-graph answers being complete.

## Limitations

- One model tier (haiku), 3 trials/ticket, 20 tickets. Underpowered for the
  false-localization effect specifically.
- Judges grade final answers only. Deliberate — trace-level judging cannot be
  blinded — but it means grading cannot see *how* an answer was reached.
- calls-to-gold fires when a marker appears in a tool input or result. Markers
  were audited and tightened (three tickets had markers hitting 60+ places,
  which would have credited the arm with more verbose output); max is now 53.
- Budget compliance is imperfect even after hardening (7/60, 10/60).
- One repo (sutra's own) is 3 commits ahead of its index; this was used
  deliberately as the GF5 staleness ticket rather than corrected.

## Reproduction

- Tickets, golds and verifier report: `benchmarks/fresh_ab/tickets.json`,
  `verification.json`
- Workflows: `benchmarks/fresh_ab/pilot_wf.js`, `main_wf.js`
- Analyzer: `benchmarks/fresh_ab/analyze.py` (`AB_BUDGET=6 python3 analyze.py <transcript_dir> tickets.json`)
- Raw rows: `main_efficiency.json` (120), `main_grades.json` (120),
  `pilot_efficiency.json`, `pilot_grades.json`
- Protocol specs: `prompt_template.md`, `judging.md`
- Branch `checkpoint/embedder-and-recall-groundwork`
