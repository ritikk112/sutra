# Sutra launch benchmark on LocBench — design spec

Date: 2026-09-24 · Status: approved design, pre-implementation
Purpose: produce **launch numbers** for Sutra that are comparable, reproducible and honest.

## 1. Questions, headline numbers, guard metric

| # | Question | Layer | Headline metric | Claim strength |
|---|----------|-------|-----------------|----------------|
| 1 | How good is Sutra's index at finding the code a GitHub issue is about? | Layer 1 (index-only) | function-level **Acc@5**, file-level **Acc@1** on LocBench, next to published LocAgent / FastCode numbers and our own BM25 baseline | large-n, deterministic → README headline |
| 2 | Does Claude Code localize better/cheaper with Sutra attached? | Layer 2 (paired agent A/B) | paired **Δ accuracy, Δ cost, Δ tokens, Δ turns** grep-only vs grep+Sutra, with bootstrap CI | n=30 → reported as *directional* with the CI in the same sentence |
| G | Does the agent even use Sutra when it is merely available? | Layer 2 guard | **adoption**: % runs with ≥1 `sutra_*` call, mean calls/run, per-tool counts | always reported; gates whether Δ means anything |

Out of scope (follow-ups, not this spec): SWE-bench task resolution, TS/Go claims, reranker / graph-expansion variants, LSP resolver, prompt variants that mention Sutra.

Fixed assumptions: agent model `claude-sonnet-5`; index embedder = the shipped default in `config/sutra.yaml` (`provider: local`, `BAAI/bge-base-en-v1.5`, 768d — decided 2026-09-24; the runtime check in §4.1 used MiniLM, so its 104 s index / 42 s cold-start figures are lower bounds); `--resolver heuristic`; `rerank=False`.

## 2. Data

Source: HF dataset `czlll/Loc-Bench_V1`, split `test`, 560 rows, 165 repos, 526 distinct `(repo, base_commit)`. Fields used: `instance_id, repo, base_commit, problem_statement, category, edit_functions`.

Fetch via the HF datasets-server rows API (`https://datasets-server.huggingface.co/rows?dataset=czlll/Loc-Bench_V1&config=default&split=test&offset=N&length=100`, stdlib `urllib`, no `datasets` dependency). Store as `benchmarkings/locbench/data/locbench_v1.json` with its SHA-256 recorded in `subset.json`; a mismatch on re-fetch is an error, not a warning.

### 2.1 Subset rule (deterministic, committed as `subset.json`)

- **Layer 1 set**: all issues from repos with ≥ 8 issues in the dataset (20 repos, ≈ 200 issues). Exact list produced by `prepare.py --select` and committed.
- **Layer 2 set**: 30 issues sampled from the Layer 1 set with `random.Random(20260924)`, stratified by `category` proportionally (Bug Report / Feature Request / Performance Issue / Security Vulnerability), max 3 per repo. Committed in the same file.
- **Shrink rule** (decided by the timing pilot, §6 stage 1): if the projected indexing wall-clock for the Layer 1 set exceeds 24 h on this machine, restrict to the 10 repos with the most issues. The Layer 2 sample is drawn *after* that decision.

### 2.2 Per-issue index

For each selected issue:

```
git clone https://github.com/<repo> bench/repos/<instance_id>      (full clone, cached per repo then worktree/checkout per commit is fine)
git -C bench/repos/<instance_id> checkout <base_commit>
sutra index bench/repos/<instance_id> --repo-url <repo> --resolver heuristic \
            --output-dir benchmarkings/locbench/artifacts/<instance_id>
```

- One artifacts root **per issue** (same repo at different commits must not share a slug directory).
- `manifest.jsonl` records per issue: `instance_id, repo, base_commit, sutra_git_sha, sutra_version, embedder_model, symbol_count, file_count, index_seconds, status`. Indexes are cached; `prepare.py` skips anything already `status=ok`.
- The checkout is **kept on disk** — Layer 2 uses the same directory as the agent's cwd.
- All `bench/` and `artifacts/` paths are gitignored; only `subset.json`, `manifest.jsonl`, results summaries and reports are committed.

### 2.3 Gold ↔ moniker normalizer (shared by both layers)

`edit_functions` entries have the shape `path/to/file.py:func` or `path/to/file.py:Class.method`. Sutra monikers have the shape `sutra python <repo> <path> Class#method().` / `<path> func().`.

`score.normalize(x) -> (path, qualified_name)` maps both forms to a canonical key: forward-slash path relative to repo root, `Class.method` / `func` with no parens or trailing `.`. Nested classes: join with `.`. It is one function, unit-tested against real gold strings and real monikers from a small indexed repo.

Gold functions absent from the index (parser gap, decorator-generated code, …) are **counted as misses and logged** in `layer1/missing_gold.jsonl` — never dropped from the denominator. The count of such cases is a reported number.

## 3. Layer 1 — index-only localization

**Queries**, both scored for every issue:
- `full`: the entire `problem_statement`
- `title`: the first non-empty line of `problem_statement`

**Retrievers**, both scored for every query:
- `hybrid`: `RetrievalPipeline.search(query, top_k=50, rerank=False)` loaded exactly as `sutra serve` loads a bundle (same registry/loader code path).
- `bm25`: the BM25 channel alone, `top_k=50`, same kind filter (none).

**Ranking → candidates**: function-level list = the ranked symbols (functions, methods; class symbols are kept but can only match gold that names a class). File-level list = unique file paths in order of first appearance.

**Metrics** (LocBench definitions): `file_acc@{1,3,5}`, `func_acc@{5,10}` where Acc@k = 1 iff **every** gold location is in the top-k. Diagnostic: `func_recall@{5,10,50}`, `file_recall@{1,5}`, `func_mrr`. All broken down by `category`, `repo`, `query × retriever`.

**Outputs**: `layer1/results.jsonl` (one row per issue × query × retriever: ranked top-50 keys, gold keys, hit flags, per-metric values) and `layer1/summary.json` (aggregates + CIs, §5). Zero API cost.

## 4. Layer 2 — paired agent A/B (Claude Code headless)

**Common invocation** (both arms; cwd = the issue's checkout; verified live on 2026-09-24 with Claude Code 2.1.281, see §4.1):

```
MCP_TIMEOUT=120000 \
claude -p "<prompt>" \
       --setting-sources "" --disable-slash-commands \
       --model claude-sonnet-5 --output-format stream-json --verbose \
       --permission-mode dontAsk --max-turns 30 --max-budget-usd 0.75 \
       --json-schema <answer.schema.json> \
       --strict-mcp-config --mcp-config '{"mcpServers":{}}' \
       --allowedTools "Read,Grep,Glob" \
       --disallowedTools "Edit,Write,Bash,WebFetch,WebSearch,Agent,NotebookEdit" \
       < /dev/null
```

### 4.1 Runtime facts the runner must respect (all verified live)

- **No `--bare`.** It reads only `ANTHROPIC_API_KEY`, never OAuth → "Not logged in". **No `--safe-mode`** either: it also drops dynamic `--mcp-config` servers (`mcp_servers: []`). Isolation comes from `--setting-sources ""` (no user/project/local settings → no hooks, no user MCP) + `--strict-mcp-config` + `--disable-slash-commands`. There is no global or project `CLAUDE.md` on the bench machine; `prepare.py` asserts none exists in a checkout before running.
- **Prompt goes first.** `--allowedTools` / `--disallowedTools` are variadic and swallow a trailing positional prompt (the run then fails with "Input must be provided…"). Always `claude -p "<prompt>" …` with `< /dev/null`.
- **`MCP_TIMEOUT=120000`.** `sutra serve` takes ≈ 42 s to initialize on this CPU (it loads the sentence-transformers model at startup); Claude Code's default 30 s MCP timeout marks it `failed`. The grep arm passes `--mcp-config '{"mcpServers":{}}'` so both arms share the identical flag set.
- **Connection check** = `system/init` message has `mcp_servers == [{"name":"sutra","status":"connected",…}]` and the tool list contains all six `mcp__sutra__*` names. Anything else → invalid run (§4).
- **Cost fields** come from the `result` message: `total_cost_usd`, `num_turns`, `duration_ms`, `duration_api_ms`, `usage.{input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens}`. On an OAuth subscription `total_cost_usd` is computed, not billed — still the right per-run cost metric.
- **Windows artifact publish bug**: `atomic_writer.py` fsyncs the artifact *directory*, which raises `PermissionError` on Windows, so `.ready` is never written and `sutra serve` ignores the bundle. Must be fixed (stage 0) before `prepare.py` can index anything on this machine.

**Arms**:
- `grep`: exactly the above.
- `grep_sutra`: `--mcp-config <issue mcp.json>` replaces the empty one, and `mcp__sutra__*` is appended to `--allowedTools`. `mcp.json` launches `<venv>/Scripts/sutra.exe serve --artifacts-dir benchmarkings/locbench/artifacts/<instance_id>` over stdio (absolute paths), so the server holds only that one index. The ≈ 42 s model load is paid per run (≈ 1 h over 90 runs) — accepted, because it is what a real user pays too. The run is **invalid** (recorded, retried once, then excluded and counted) if the `system/init` message does not show the `sutra` server as connected.

**Prompt**: byte-identical across arms, stored at `layer2/prompt.md`. Contents: the `problem_statement`, the instruction to identify the functions that must be edited to fix it, return up to 5 ranked functions and their files. **The prompt does not mention Sutra or MCP** — the as-shipped condition. Tool descriptions are the only advertising the index gets.

**Answer schema** (`answer.schema.json`): `{ "functions": [{"path": str, "name": str}], "files": [str] }`, max 5 functions. Scored with the §2.3 normalizer → `func_acc@5`, `file_acc@1`, plus recall. No LLM judge.

**Per-run capture** (from the stream-json transcript, saved verbatim per run): `total_cost_usd, num_turns, duration_ms, duration_api_ms, input/output/cache_read/cache_creation tokens, is_error`, ordered tool-call list (`name`, `input`), parsed answer, and post-hoc constraint check (`grep` arm: 0 `mcp__` calls; `grep_sutra`: server connected). Violations are excluded and counted.

**Design**: 30 issues × 2 arms × 3 trials = 180 runs. Arm order randomized per issue with the §2.1 seed. Concurrency ≤ 2. Expected spend $60–85, hard ceiling $135 (per-run cap).

**Pre-registered decision rule (pilot)**: run 5 issues × 2 arms × 1 trial first. If fewer than 50 % of `grep_sutra` pilot runs make ≥ 1 `sutra_*` call, **stop**: the finding is "tool descriptions don't earn adoption", fix that in the product, and re-pilot before spending the remaining budget. Pilot runs are never mixed into the final 180.

## 5. Statistics and reporting

- **CIs**: 95 % via cluster bootstrap over **repos** (resample repos with replacement, 10 000 draws) for every Layer 1 aggregate and every Layer 2 paired Δ. Issues within a repo are correlated; per-issue bootstrap would understate error.
- **Layer 2 paired Δ**: per issue, mean of its 3 trials per arm; Δ = grep_sutra − grep; bootstrap over issues clustered by repo. Also `pass^3` per arm (all 3 trials `func_acc@5 = 1`) and the adoption metrics from §1.
- **No LLM judge, no p-hacking**: metrics, n, seed, subset rule and the adoption decision rule are fixed in `PREREG.md` **before** any Layer 2 run; the report script refuses to run if `PREREG.md` is missing or its hash differs from the one recorded at pilot time.
- **`report.py`** renders `REPORT.md` from `layer1/summary.json` + `layer2/summary.json` with a fixed template: headline table → guard metric → CIs → method → caveats (Python-only, 20 repos, n=30 directional) → README-ready snippet. Published LocAgent/FastCode numbers appear in the headline table labelled "published; different setup".

## 6. Implementation plan (handoff)

Branch `bench/locbench`. Code in `benchmarkings/locbench/`:

| File | Responsibility |
|------|----------------|
| `prepare.py` | fetch + checksum dataset; `--select` writes `subset.json`; clone/checkout/index per issue; `manifest.jsonl` |
| `score.py` | normalizer (§2.3), Acc@k / Recall@k / MRR |
| `layer1.py` | run §3, write `layer1/results.jsonl`, `layer1/summary.json` |
| `layer2.py` | build per-issue `mcp.json`, invoke `claude -p`, parse stream-json, write `layer2/runs/<id>.jsonl` + `layer2/results.jsonl` |
| `stats.py` | cluster bootstrap, paired Δ, pass^k |
| `report.py` | `REPORT.md` + README snippet; PREREG hash check |
| `PREREG.md` | pre-registration |

Tests in `tests/benchmark/` — real data, no mocks: normalizer on real gold strings and real monikers from a tiny indexed fixture repo; Acc@k on hand-checked examples; stream-json parser on a real recorded Claude Code transcript; bootstrap on a known distribution.

Stages (each ends with a report back to the designer for review before the next starts):
0. **Windows publish fix** in `sutra/core/artifact/atomic_writer.py`: skip the directory `fsync` when `os.name == "nt"` (per-file fsync + `os.replace` remain), with a real test that publishes a bundle and asserts `.ready` exists on the current OS. Product bug, ships independently of the benchmark.
1. `prepare.py` + **timing pilot** on 3 issues (small / medium / django-sized) → shrink decision (§2.1). Reference point: the `sutra` repo (≈ 700 symbols) indexes in ≈ 104 s with the local embedder on this machine.
2. `score.py` + `layer1.py` + tests → full Layer 1 run → `layer1/summary.json`.
3. `layer2.py` + **manual MCP-runtime check** (Sutra MCP connected to a headless Claude Code run on one index, verified from `system/init`) + 5-issue adoption pilot → decision rule.
4. Full 180-run Layer 2 → `stats.py`, `report.py`, `PREREG.md` hash check → `REPORT.md`.

## 7. Prior evidence this design accounts for

Earlier Sutra A/B/C runs (2026-07) on two repos and on `sutra` itself found equal answer quality across arms, +67–75 % cost for a Sutra-only arm, and **0/36 index usage** in the free-choice arm. This spec therefore (a) drops the Sutra-only arm as non-representative, (b) makes adoption a first-class guard metric with a stop rule, and (c) replaces hand-authored tickets with LocBench's 560 real issues to remove authoring bias.
