# Corpus size, cross-repo, and where sutra actually beats grep (2026-08-19)

A controlled A/B built to test one hypothesis — *the index earns its keep on
large codebases* — over a corpus ladder spanning 497 to 12,830 definitions,
plus questions that span three large repos at once.

**The hypothesis failed. Corpus size does not predict the advantage. Task
shape does — and the effect is concentrated almost entirely in questions
that cross repository boundaries, where the grep-only arm scored zero.**

## Setup

- **Repos indexed for this run**: celery (3,882 symbols), django (12,132),
  sqlalchemy (13,175). Combined with flask (496) this gives a ~26x ladder.
- **20 tickets, 3 kinds**:
  - **blast radius** (8) — "what breaks if I change X?", graded on precision
    and recall against a hand-verified call-site list.
  - **disambiguation** (8) — a name with many sibling definitions; which one
    actually runs, and what selects it. django's `as_oracle` has **39**
    definitions across 13 files.
  - **cross-repo** (4) — one question answerable only by comparing django,
    celery and sqlalchemy together (~27,000 definitions).
- **120 solver runs** (20 x 2 arms x 3 trials), claude-haiku, byte-identical
  prompts differing only in the tool list, **budget fixed at 6 tool calls for
  every repo** — holding it constant is the instrument that would reveal a
  size effect.
- Judges see only final answers, never traces; one blind sonnet judge per
  ticket over 6 shuffled unlabelled answers.
- Golds built without the index. No author, verifier or judge had sutra tools.

### Run hygiene

| check | result |
|---|---|
| control-arm sutra calls | **0.00** |
| budget violations | sutra 5/60, control 7/60 |
| concurrency at solver start | sutra 6, control 7 |
| index adoption | blast 23/24, disambiguation 23/24, cross-repo **12/12** |
| ticket verification | 16/16 pass, zero repairs, every caller set re-derived |

## Result 1 — corpus size does not predict the advantage

| repo | definitions | sutra | control | delta |
|---|---|---|---|---|
| flask | 497 | 0.944 | 1.000 | −0.056 |
| celery | 3,601 | 0.792 | 0.583 | **+0.208** |
| django | 11,010 | 0.933 | 0.867 | +0.067 |
| sqlalchemy | 12,830 | 0.792 | 0.875 | −0.083 |

Non-monotonic, and the largest repo in the set is one where sutra *loses*.
Whatever drives the difference, raw corpus size is not it.

This matters because "use an index because your monorepo is big" is the
industry's standard pitch, and this run does not support it.

## Result 2 — task shape predicts it, and cross-repo is decisive

| task kind | n | sutra | control | delta | tickets favouring sutra |
|---|---|---|---|---|---|
| blast radius | 8 | 0.812 | **0.875** | −0.062 | 1 of 8 |
| disambiguation | 8 | **0.917** | 0.771 | +0.146 | 3 of 8, 0 against |
| **cross-repo** | 4 | **0.583** | 0.167 | **+0.417** | **4 of 4** |

### The cross-repo collapse

| arm | correct | partial | wrong | false localization |
|---|---|---|---|---|
| sutra | 4 | 6 | 2 | 6/12 |
| control | **0** | 4 | 8 | 8/12 |

**The grep-only arm did not answer a single cross-repo question correctly in
12 attempts.** It localized the gold mechanism in only 7 of 12 trials against
sutra's 10, and fabricated a confident wrong symbol in 8 of 12.

The mechanism is not subtle. A cross-repo question requires holding three
codebases side by side; grep must be pointed at one tree at a time and the
agent must then assemble and compare the results itself, under a 6-call
budget. `sutra_search` with no repo argument returns ranked hits across all
three in one call. Index adoption on these tickets was 12/12 with a mean of
**5.33 index calls per trial** — more than double any other task kind. The
agent reached for it because nothing else could do the job.

### Disambiguation

| arm | named the right implementation | named a wrong sibling |
|---|---|---|
| sutra | **24/24 (100%)** | 0 |
| control | 20/24 (83%) | 0 |

The gap concentrates on the hardest case: **DD1 (`as_oracle`, 39 sibling
definitions) — sutra 3/3, control 1/3.**

Note what the control arm did *not* do: it never named a wrong sibling. Its
failure mode was declining to identify which implementation at all — naming
the bare function and stopping. Grep tells you the name exists 39 times; it
offers nothing to rank them by, and the agent correctly declined to guess.

### Blast radius — control still wins

Consistent with the previous run. Control localized the gold in **24/24**
trials; grep is genuinely excellent at "find every place this name appears,"
which is most of what a call-site enumeration needs.

## Result 3 — sutra is still slower, and now measurably more expensive

Paired by ticket across all 20:

| metric | delta (sutra − control) | p |
|---|---|---|
| wall clock | **+7.2 s** | **<0.0001** |
| context growth (cache-write) | **+4,913 tok** | **0.040** |
| time-to-first-gold | +3.8 s | 0.11 |
| calls-to-gold | −0.3 | 0.67 |

The latency penalty replicates the previous run (+5.9s there, +7.2s here,
both p<0.001). Context growth, marginal last time (p=0.060), is now
significant. **Buying cross-repo capability costs roughly 7 seconds and 5k
tokens per task.**

## On statistical power — read this before quoting any p-value

Overall correctness is +0.117 with **p = 0.11**: not significant across 20
tickets. Per class, nothing reaches p<0.05 either.

The cross-repo result deserves a specific caveat in *both* directions. Sutra
wins **4 of 4** cross-repo tickets and the control arm scores zero correct out
of twelve, which is a large effect. But with 4 paired tickets, a sign-flip
permutation test cannot produce a p below 0.125 no matter how big the effect
is — the test is floor-limited by sample size. **The cross-repo finding is
therefore strong evidence and weak statistics simultaneously**, and the right
response is to power it properly (12-15 cross-repo tickets), not to quote
p=0.12 as though it were a null.

## What this means for the story

The pitch is not "your repo is big." It is:

> **Grep cannot see across repository boundaries, and an agent under a call
> budget cannot paper over that. On questions spanning three services, the
> grep-only agent got zero right out of twelve.**

The secondary claim is disambiguation: when a name has 39 implementations,
grep returns 39 indistinguishable hits and a careful agent refuses to guess.
The index returns qualified names and owning classes, and the agent answers.

And the honest counterweight, which belongs in the same article: on
single-repo call-site enumeration — probably the most common real task —
**grep-only wins**, and the index costs 7 seconds and 5k tokens per task to
run. Sutra is not a faster grep. It answers questions grep structurally
cannot, and it charges latency for the privilege.

## Limitations

- One model tier (haiku), 3 trials/ticket, 4 cross-repo tickets. The headline
  effect is the least-powered cell in the run.
- Budget 6 may itself favour the arm that needs fewer calls for cross-repo
  work; a budget sweep on cross-repo tickets alone would separate "sutra is
  better at this" from "sutra is better at this *under a budget*".
- Two sutra trials badly overran the budget (XL4 +14, CE1 +10). Violations
  are balanced overall (5/60 vs 7/60) but the tail is not.
- Corpus ladder is 4 repos; per-repo scores rest on 4-5 tickets each, which is
  why the non-monotonic size result should be read as "no evidence for a size
  effect" rather than "proof there is none".

## Reproduction

- Tickets/golds: `benchmarks/fresh_ab/scale_tickets_v2.json`, verification in
  `scale_verification_v2.json`
- Workflows: `scale_authoring_v2.js`, `scale_main_wf.js`
- Analyzer: `AB_BUDGET=6 python3 analyze.py <transcript_dir> scale_tickets_v2.json`
- Raw rows: `scale_efficiency.json` (120), `scale_grades.json` (120)
- Supporting no-agent measurements: `grep_noise.json`, `resolver_rates.json`
