# Solver prompt template (fresh A/B)

Both arms receive EXACTLY this text. The only permitted difference between arms is the
`{{TOOLS}}` block. Nothing else — not a word, not a line break — may differ, or the run
measures prompt framing instead of tool capability.

```
{{TICKET_TEXT}}

Repo checkout(s):
{{REPO_PATHS}}

[trial marker: {{TICKET_ID}}-{{ARM}}-t{{TRIAL}}]

Tools available to you:
{{TOOLS}}

You have a hard budget of {{BUDGET}} tool calls for this task. Once you have used
{{BUDGET}} calls you must stop investigating and answer with whatever you have
established so far. Budget your calls deliberately.

Answer the ticket. Name the specific mechanism in the source that explains it: the files,
the functions or classes involved, and how they interact to produce the described
behaviour. Be concrete and cite paths. Your final message is your answer — write it for an
engineer who has not read this code.
{{EXTRA}}
```

## The two `{{TOOLS}}` blocks

CONTROL:
```
- Bash, Grep, Glob, Read
Use only these tools.
```

SUTRA:
```
- The sutra code-index MCP tools, which serve a pre-built index of this repo (indexed as
  "{{INDEXED_REPO}}"): mcp__sutra__sutra_search, mcp__sutra__sutra_get_symbol,
  mcp__sutra__sutra_get_callers, mcp__sutra__sutra_get_callees,
  mcp__sutra__sutra_expand_neighbors, mcp__sutra__sutra_list_repos
- Bash, Grep, Glob, Read
```

Neither block editorialises. The control block does not apologise for missing tools; the
sutra block does not recommend its tools or claim they are faster. An arm is never told it
is expected to win.

## `{{EXTRA}}`

Empty for almost every ticket. Two cases append a line, and when they do it is appended
identically to BOTH arms:

- Tickets in the `sutra` repo:
  `\nThis repository's markdown files contain notes that give away answers; base your answer on the source code only.`
- Cross-repo (XR) tickets: `{{REPO_PATHS}}` lists every repo involved, for both arms.

## Budget

`{{BUDGET}}` is a single integer fixed by the calibration pilot and then held constant for
the entire main run, identical in both arms. It is never tuned per class or per ticket —
tuning it after seeing results would let the cap be chosen to produce a preferred outcome.

## Trial marker

`[trial marker: ...]` exists solely so the analyzer can match a transcript back to its
ticket, arm, and trial. It names the arm, which the solver can see — unavoidable, since it
must survive in the transcript — but it carries no valence: "sutra" and "control" are bare
labels, and the arm is already obvious from the tool list anyway.

## ToolSearch does not count against the budget

The index arm must load its MCP tool schemas via `ToolSearch` before it can call them.
That is a harness artifact, not investigation, and only one arm pays it — left counted, it
would silently hand the control arm a free extra call under every budget.

So the budget line in the shared template ends with a sentence appended identically to
both arms:

    Loading tool definitions does not count against this budget.

and `analyze.py` excludes `ToolSearch` from `total_tool_calls` and from the calls-to-gold
index. For the control arm the sentence is a no-op — there is nothing to load — which is
precisely why it can be added to both without breaking byte-identity.
