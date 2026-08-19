# Why the call graph gets less complete as repos grow (2026-08-19)

Follow-up to `SCALE_REPORT.md`, which measured call-edge resolution falling
monotonically from **83% (requests, 321 defs) to 56% (sqlalchemy, 12,830 defs)**
with unresolved edges growing 152×. That report established *that* it happens.
This one establishes *why*, and sizes the fixes.

Everything below is measured from the served artifacts
(`~/.sutra/artifacts/*/graph.json`) and the resolver source. No agents were
used. Scripts and raw output are listed under Reproduction.

---

## TL;DR

1. **100% of unresolved call edges are ambiguity refusals.** Not one edge in any
   repo is unresolved for any other reason. This is by design — the resolver
   refuses to guess — so the decline is the design meeting a property of large
   codebases, not a bug in the rules.
2. **Name ambiguity is the driver.** Correlation between the share of
   definitions carrying a duplicated name and the resolution rate is **−0.865**.
3. **The rule that collapses is `local`, not `unique`.** Same-file resolution
   falls from 46.9% to 25.2% of edges as codebases modularise. `unique` holds
   flat and cannot absorb the overflow.
4. **~88% of misses are `obj.f()` / `self.f()` method calls** needing a receiver
   type the resolver has no way to infer.
5. **Zero `extends` edges are resolved in any repo** — 0 of 1,875 in django,
   0 of 1,712 in sqlalchemy. Nothing in the codebase resolves them. The class
   hierarchy is empty, which both blocks the cheapest call-resolution fix and
   means one advertised MCP capability has no data behind it.
6. **Three cheap fixes, benchmarked offline against the real graphs, recover
   +6 to +13 points** of resolution (django 63%→72%, sqlalchemy 56%→65%) with
   no type inference and no new dependency. They do **not** reverse the scale
   trend — a ~24-point gap between the smallest and largest repo survives all
   of them.

---

## How the resolver works

`sutra/core/resolver/heuristic.py`. Per unresolved CALLS edge with callee name
N, in precedence order:

| rule | fires when |
|---|---|
| `local` | exactly one symbol named N in the **caller's own file** |
| `import` | the caller's file imports N from module M, and exactly one candidate named N lives in a module matching M |
| `unique` | exactly one call-form-compatible candidate named N **in the whole repo** |

Every rule requires **exactly one** candidate. The docstring is explicit about
why:

> Anything else stays unresolved — ambiguity is left for P20-full (LSP) rather
> than guessed at. **Precision over recall**: a wrong edge poisons graph
> expansion; a missing edge just doesn't help it.

That trade is defensible. The finding of this investigation is not that the
trade is wrong, but that **its cost is not constant** — it scales against you,
and it is invisible at the point of use.

---

## Finding 1 — every miss is an ambiguity refusal

Categorising all unresolved matchable edges by reason:

| repo | defs | rate | ambiguous (>1 candidate) | any other reason |
|---|---|---|---|---|
| requests | 321 | 83% | **100%** | 0% |
| flask | 497 | 78% | **100%** | 0% |
| fastapi | 1,063 | 79% | **100%** | 0% |
| pydantic | 2,799 | 77% | **100%** | 0% |
| celery | 3,601 | 65% | **100%** | 0% |
| django | 11,010 | 63% | **100%** | 0% |
| sqlalchemy | 12,830 | 56% | **100%** | 0% |

There is no long tail of parse failures, missing symbols, or odd call forms.
The resolver sees a name, finds several definitions of it, and declines. Every
single time.

This means **resolution rate is a direct function of name ambiguity**, and any
fix must either disambiguate or accept a guess.

### How many candidates does a miss face?

| repo | median candidates | p90 candidates |
|---|---|---|
| requests | 4 | 4 |
| flask | 2 | 7 |
| fastapi | 6 | 12 |
| pydantic | 3 | 5 |
| celery | 4 | 23 |
| django | 8 | **29** |
| sqlalchemy | 8 | 23 |

In django the median unresolved edge is choosing between **8** same-named
definitions, and the top decile between 29.

---

## Finding 2 — ambiguity predicts the decline

From `grep_noise.json` (measured independently of sutra, by parsing source):

| repo | defs carrying a duplicated name | call-edge resolution |
|---|---|---|
| requests | 34.6% | 83% |
| fastapi | 34.8% | 79% |
| flask | 44.1% | 79% |
| celery | 46.9% | 65% |
| pydantic | 51.1% | 77% |
| django | 55.6% | 63% |
| sqlalchemy | 64.1% | 56% |

**corr = −0.865** (n=7). pydantic is the visible residual — 51% ambiguity but
77% resolution — so ambiguity explains most of the variance but not all.

---

## Finding 3 — `local` collapses; `unique` does not

Per-rule yield as a share of all matchable edges:

| repo | defs | `local` | `import` | `unique` | `local-scope` | total |
|---|---|---|---|---|---|---|
| requests | 321 | **46.9%** | 28.2% | 6.2% | 1.4% | 83% |
| flask | 497 | 40.8% | 6.8% | 29.3% | 2.0% | 79% |
| fastapi | 1,063 | 49.7% | 10.7% | 17.7% | 0.9% | 79% |
| pydantic | 2,799 | 37.6% | 12.5% | 26.3% | 1.1% | 77% |
| celery | 3,601 | 39.0% | 8.7% | 16.2% | 1.2% | 65% |
| django | 11,010 | 26.7% | 9.8% | 25.2% | 1.5% | 63% |
| sqlalchemy | 12,830 | **25.2%** | 3.5% | 26.4% | 1.3% | 56% |

This overturned my first hypothesis. I expected `unique` to decay, since it is
the rule that depends on repo-wide name uniqueness. It doesn't — it holds around
25%. What collapses is `local` (46.9% → 25.2%) and `import` (28.2% → 3.5%).

Corroborating from the graph directly, the same-file share of **resolved** edges
falls from 57.6% (requests) to 43.6% (django) / 46.1% (sqlalchemy).

**The mechanism is modularity, not naming.** A large codebase spreads its calls
across modules. The highest-yield rule only covers same-file calls, so its
territory shrinks as a codebase grows up. The overflow lands on `import` and
`unique`, which need global uniqueness — exactly the property large codebases
lose.

Two forces therefore push the same way:
- fewer calls are same-file → `local` covers less
- more names are duplicated → `unique`/`import` succeed less on what's left

### A compounding detail

The share of calls whose callee isn't indexed at all (stdlib, third-party) drops
from **63% (requests) to 23% (sqlalchemy)**. Large repos mostly call themselves.
So as repos grow there are proportionally *more* in-repo edges available to
resolve — and a lower share of them gets resolved. Both terms move against you.

---

## Finding 4 — the misses are method calls

| repo | `obj.f()` / `self.f()` | bare `f()` |
|---|---|---|
| requests | 80% | 20% |
| flask | 95% | 5% |
| fastapi | 86% | 14% |
| pydantic | 81% | 19% |
| celery | 90% | 10% |
| django | 88% | 12% |
| sqlalchemy | 89% | 11% |

Around **85–90% of every repo's misses are method calls**. Resolving them
properly requires knowing the receiver's type, which the heuristic resolver
explicitly does not attempt. This is the bulk of the problem and it is the
expensive part to fix.

---

## Finding 5 — the class hierarchy is entirely unresolved

| repo | `extends` edges | resolved |
|---|---|---|
| requests | 51 | **0** |
| flask | 42 | **0** |
| fastapi | 334 | **0** |
| pydantic | 429 | **0** |
| celery | 242 | **0** |
| django | 1,875 | **0** |
| sqlalchemy | 1,712 | **0** |

Cause, confirmed in source — `heuristic.py:127`:

```python
for rel in relationships:
    if rel.kind != RelationKind.CALLS:
        continue
```

The extractors emit `EXTENDS` relationships (`extractor/adapters/python.py:572`
and equivalents for TypeScript and Go), but **nothing in the codebase ever
resolves them**, so `target_id` is null on every one. This was found by
accident: an MRO-aware fix estimate recovered exactly zero edges beyond the
caller's own class, because there was no hierarchy to walk.

Two consequences:

1. **It blocks the cheapest call-resolution fix** (see M1/M2 below). Inheritance
   edges must be resolved before inheritance can help resolve calls.
2. **An advertised capability has no data behind it.** The MCP server describes
   `sutra_expand_neighbors` as walking "resolved callers, callees, **type
   hierarchy** and references." The type-hierarchy dimension currently
   contributes nothing on any repo, and the tool does not say so.

---

## Hypotheses

Labelled by evidential status, so nothing here reads as established that isn't.

| # | hypothesis | status |
|---|---|---|
| H1 | Every miss is an ambiguity refusal | **Confirmed** — 100%, all 7 repos |
| H2 | Ambiguity growth drives the decline | **Strongly supported** — r=−0.865, plus an independent source-level ambiguity measure |
| H3 | The `unique` rule decays with size | **Refuted** — flat at ~25% |
| H4 | The `local` rule decays with size, via modularisation | **Confirmed** — 46.9%→25.2%, corroborated by same-file share of resolved edges |
| H5 | Most misses need receiver-type inference | **Supported** — 85–90% are method-form calls |
| H6 | `extends` is unresolved because the resolver skips non-CALLS edges | **Confirmed in source** — `heuristic.py:127` |
| H7 | pydantic's residual (51% ambiguity, 77% resolution) is explained by file granularity — 24 defs/file vs django's 12, so more calls stay same-file | **Untested.** sqlalchemy has 42 defs/file and the *worst* rate, so this is at best partial. Needs a proper multivariate check |
| H8 | Re-exports inflate apparent ambiguity — several "candidates" are one symbol reached by several paths, so dedup-by-identity recovers edges at zero precision cost | **Confirmed** — recovers 6–11% of misses (sqlalchemy 998 of 9,320; django 499 of 8,580). See M6 |
| H9 | Dynamic/metaprogrammed dispatch (sqlalchemy's generated methods, django's `__getattr__`) sets a hard ceiling no static resolver can pass | **Untested** — would explain sqlalchemy being worst despite mid-size |

---

## Mitigation strategies

Ranked by measured or estimated recovery against implementation cost. Where a
number is measured it says so; where it is projected it says that too.

### M1 — self-first rule: prefer a candidate owned by the caller's own class

When `self.f()` is called from inside a class that defines `f`, that is the
match. No type inference required — the receiver is `self`.

**Measured recovery** (from `resolver_mro_estimate.json`):

| repo | unresolved | recoverable | share | rate before → after |
|---|---|---|---|---|
| requests | 61 | 10 | 16% | 83% → 85% |
| flask | 106 | 24 | 23% | 78% → 83% |
| fastapi | 273 | 51 | 19% | 79% → 83% |
| pydantic | 951 | 73 | 8% | 77% → 79% |
| celery | 2,086 | 341 | 16% | 65% → 70% |
| django | 8,580 | 1,326 | 15% | 63% → 68% |
| sqlalchemy | 9,320 | 1,080 | 12% | 56% → 61% |

Cost: small — the graph already carries `enclosing_class_id`. Precision risk is
low, since Python resolves `self.f()` to the class's own `f` unless a subclass
overrides it, and the caller's own definition is the correct answer for the
class being analysed.

**Caveat**: this recovers +2 to +5 points. It does not reverse the trend — the
gap between requests and sqlalchemy stays roughly 24 points.

### M2 — resolve `extends`, then extend M1 to the full MRO

Currently zero hierarchy edges resolve, so M1 cannot look past the caller's own
class. Resolving `extends` (the same three rules would apply — class names are
usually less ambiguous than method names) unlocks walking base classes for an
inherited `f`.

**Measured, by prototyping it offline.** Applying the resolver's own
local-then-unique logic to `extends` edges resolves the hierarchy easily —
class names are far less ambiguous than method names:

| repo | extends matchable | resolvable | rate |
|---|---|---|---|
| requests | 37 | 35 | 95% |
| flask | 22 | 22 | 100% |
| fastapi | 85 | 80 | 94% |
| pydantic | 252 | 175 | 69% |
| celery | 121 | 108 | 89% |
| django | 1,495 | 1,194 | **80%** |
| sqlalchemy | 1,201 | 940 | **78%** |

Walking that recovered hierarchy for inherited `self.f()` then recovers
additional call edges beyond M1: **django +469, sqlalchemy +214, celery +50,
flask +15**. Smaller than M1 but real, and unobtainable without this step.

Secondary benefit: it makes `sutra_expand_neighbors`'s advertised type-hierarchy
traversal real rather than nominal.

### M3 — surface ambiguity instead of dropping it

**This is the one that addresses the harm actually observed in the benchmark.**

`sutra_get_callers` currently returns a clean list with no indication that
anything was withheld. In the earlier run it returned 3 of 4 callers for
`select_proxy` and the agent reported 3 with confidence. A confidently
incomplete answer is worse than a slow one.

Proposal: return unresolved-but-matchable edges as low-confidence candidates,
or at minimum attach a count — `{"resolved": 3, "ambiguous_unresolved": 5}` — so
a caller knows the list is a lower bound at the point of use, not in the
documentation.

**Recovery: zero.** Recall does not improve. What improves is that the failure
becomes visible, which is the difference between a tool that is incomplete and
a tool that misleads. Cheapest change here, and arguably the most important.

### M4 — index test files

Independent of ambiguity. Test files are excluded by policy (`config/sutra.yaml`:
"Test files are always excluded"), so no call site in a test can ever be
returned. For the "what breaks if I change this signature" question — the exact
question `get_callers` exists to serve — tests are among the first things that
break.

Either index them, or state the exclusion in the tool response. **Additive with
M1–M3**; it addresses a different population of missing edges entirely.

### M5 — LSP / type inference for the method-call bulk

The `P20-full` path the resolver docstring already anticipates
(`lsp_resolver.py` exists). Addresses the 85–90% of misses that need a receiver
type — the only mitigation that could plausibly restore large-repo resolution to
small-repo levels.

Cost is by far the highest: a language server per language, indexing time, and a
hard dependency. **Measure M2's recovery before committing to this** — if
hierarchy-aware resolution recovers most of the inherited-method population, the
residual may not justify the dependency.

### M6 — dedup candidates by identity before declaring ambiguity (tests H8)

If several "candidates" are the same underlying symbol reached through
re-exports, the ambiguity is an artifact of module structure rather than real.
Deduplicating candidates by definition site (`file_path`, `line_start`) before
the uniqueness check recovers those at **zero precision cost** — the edge was
never genuinely ambiguous.

**Measured recovery:**

| repo | unresolved | recovered by dedup | share |
|---|---|---|---|
| requests | 61 | 14 | 23% |
| flask | 106 | 31 | 29% |
| fastapi | 273 | 42 | 15% |
| pydantic | 951 | 85 | 9% |
| celery | 2,086 | 215 | 10% |
| django | 8,580 | 499 | 6% |
| sqlalchemy | 9,320 | **998** | 11% |

Confirms H8. Roughly a tenth of every large repo's "ambiguity" is not ambiguity
at all.

### M7 — rank by package proximity as a tiebreak

When several genuine candidates remain, prefer the one nearest in package path
to the caller. This is a **guess**, and it breaks the resolver's stated
precision-over-recall contract — so it should only ship behind M3's confidence
marking, never as a silently-resolved edge.

---

## Stacked result — all three cheap fixes together

M6 + M1 + M2, de-duplicated so no edge is counted twice, applied to the real
graphs:

| repo | defs | unresolved | recovered | rate now | projected |
|---|---|---|---|---|---|
| requests | 321 | 61 | 23 | 83% | **89%** |
| flask | 497 | 106 | 60 | 78% | **91%** |
| fastapi | 1,063 | 273 | 63 | 79% | **84%** |
| pydantic | 2,799 | 951 | 143 | 77% | **81%** |
| celery | 3,601 | 2,086 | 477 | 65% | **73%** |
| django | 11,010 | 8,580 | 2,094 | 63% | **72%** |
| sqlalchemy | 12,830 | 9,320 | 1,999 | 56% | **65%** |

**+6 to +13 points, no type inference, no new dependency.**

Read the last column honestly, though: the spread between the smallest and
largest repo is 89% → 65%. That is still a **24-point scale gap** — almost
exactly the 27-point gap we started with. These fixes raise the whole curve;
they do not flatten it. Flattening it needs receiver-type inference (M5), which
is the only mitigation that attacks the 85–90% method-call bulk directly.

## Recommended order

1. **M3** (surface ambiguity) — fixes the observed harm, costs almost nothing,
   independent of everything else.
2. **M6** (dedup re-exports) — cheap, tests H8, zero precision risk.
3. **M1** (self-first) — measured +2 to +5 points, low risk.
4. **M2** (resolve `extends`) — unlocks the MRO walk and makes an advertised
   capability real. **Measure its recovery; it is the biggest unknown.**
5. **M4** (tests) — a policy decision as much as an engineering one.
6. **M5** (LSP) — only after M2's number is known.

Note what this order implies: **the first two mitigations do not improve recall
at all.** They make the incompleteness visible and remove ambiguity that was
never real. That is the right starting point when the measured harm was a
confident wrong answer rather than a missing one.

---

## Reproduction

- Per-rule yields: parsed from the indexer's `[resolver] CALLS:` log lines
- `benchmarks/fresh_ab/resolver_rates.json` — resolution rate per repo
- `benchmarks/fresh_ab/resolver_forensics.json` — unresolved-edge categorisation,
  candidate counts, same-file and external shares
- `benchmarks/fresh_ab/resolver_fixsize.json` — call-form split, caller-class hits
- `benchmarks/fresh_ab/resolver_mro_estimate.json` — M1 recovery per repo
- `benchmarks/fresh_ab/extends_rates.json` — hierarchy edge resolution (as served)
- `benchmarks/fresh_ab/resolver_mitigations.json` — M6/M1/M2 recovery and the
  stacked projection, measured by applying each rule offline to the real graphs
- `benchmarks/fresh_ab/grep_noise.json` — source-level ambiguity, independent of sutra
- Source read: `sutra/core/resolver/heuristic.py` (rules at :60–92, CALLS-only
  filter at :127)

**Accuracy note.** Unresolved counts reconstructed from `graph.json` differ from
the indexer's own log by at most 4 edges out of 23,245 (django: 8,580 vs 8,576),
because the candidate index is rebuilt from the artifact rather than shared with
the resolver. Under 0.05%; it does not affect any conclusion here.
