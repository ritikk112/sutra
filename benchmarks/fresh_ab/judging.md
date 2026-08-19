# Grading protocol (fresh A/B)

## Judges grade final answers, not traces

The obvious design — hand the judge each full trace — cannot be blinded. A trace from the
index arm contains `mcp__sutra__sutra_search` tool calls in plain sight, so any judge
reading traces knows exactly which arm it is grading and the blind is worthless.

So judges see **only the solver's final answer text**, never the trace. This is both
properly blind and sufficient: correctness is a property of the answer, and every process
metric (calls-to-gold, wall clock, tool mix) is computed mechanically by `analyze.py` from
the transcripts, where no judgement is involved.

## Batched per ticket

One judge per ticket, seeing all 6 answers for that ticket at once (2 arms x 3 trials),
shuffled and unlabelled as A-F.

Batching is deliberate: a judge that grades all 6 answers against one gold applies one
consistent standard, whereas 6 independent judges drift — and drift between arms is
exactly the bias that would corrupt the result. It also cuts judge agents 6x.

Shuffle is by deterministic rotation (answers rotated by ticket index mod 6) rather than
random, because workflow scripts cannot call `Math.random()`. Rotation is enough: it
prevents any fixed position from always holding the same arm, which is the only property
blinding needs here.

## What the judge returns, per answer

- `verdict`: `correct` | `partial` | `wrong`
  - correct: identifies the actual mechanism, with the right files/symbols, and the
    explanation of how it produces the behaviour is accurate
  - partial: finds part of the mechanism, or the right area with a materially wrong or
    missing explanation
  - wrong: wrong mechanism, or confidently asserts something the code does not do
- `false_localization`: true when the answer names a specific wrong symbol or file
  **confidently**. This is scored separately from `wrong` because it is the failure mode
  that actually costs an engineer time: a confident wrong pointer sends someone to read
  the wrong file, whereas "I could not determine this" merely wastes the question.
- `caller_precision` / `caller_recall` (BR tickets only): scored against the ticket's
  verified `caller_set`, counting the answer's claimed call sites. Decoys claimed as real
  call sites reduce precision; genuine sites omitted reduce recall.
- `evidence`: the quote from the answer that justifies the verdict.

## Anti-bias instructions given to every judge

1. You do not know which tool produced which answer, and you must not speculate. If you
   find yourself reasoning about how an answer was produced, stop — grade the content.
2. Grade against the gold, not against the other answers. Do not curve. All six may be
   correct; all six may be wrong. A ticket where every answer is correct is a real and
   publishable result, not a sign you graded too leniently.
3. Longer is not better. An answer that states the right mechanism in two sentences beats
   one that surrounds it with three paragraphs of adjacent detail. Verbosity is not
   evidence of understanding, and one arm's tooling may produce more verbose answers.
4. Do not reward an answer for naming many plausible files. Precision counts.
5. If the gold itself appears wrong against the code, say so in `gold_dispute` rather than
   forcing verdicts — a bad gold must surface as a bad gold, not as six wrong answers.

That last rule matters: it is the only mechanism by which an authoring error that survived
verification can still be caught before it reaches the report.
