# Sutra MCP Server — Usage

The Sutra MCP server is **the product**: an AI agent (Claude Code, Claude
Desktop, Cursor, or anything MCP-aware) connects to it and asks questions
about your team's indexed repositories — "which function saves the meeting
in the db", "who calls `upload_voice_note`", "show me the `Meeting` model".

The entire consumer stack runs **in-memory from a directory of artifacts**.
No Postgres, no services, no infrastructure: `pip install` + a folder is the
whole deployment. (Postgres exists only on the *indexer* side, for
incremental-update bookkeeping.)

```
indexer machine                      any machine (zero infra)
┌─────────────────────┐   share    ┌──────────────────────────────┐
│ pipelines.full_index │  ───────▶ │  ~/sutra-artifacts/          │
│  (tree-sitter +      │  (rsync,  │    booth/      ← one repo    │
│   embedder +         │   drive,  │    gin/        ← another     │
│   Postgres)          │   USB…)   │  python -m sutra.mcp         │
└─────────────────────┘            └──────────────────────────────┘
```

---

## 1. Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional — ONLY needed for the local embedder and the reranker.
# Artifacts embedded with all-MiniLM-L6-v2 need this at query time too:
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build, ~1GB
pip install -r requirements-ml.txt
```

## 2. Index a repository → artifact

```bash
export SUTRA_PG_URL=postgresql://postgres:postgers@localhost:5434/postgres   # indexer side only

python -m pipelines.full_index \
    --root /path/to/your-repo \
    --repo-url https://github.com/org/your-repo \
    --output-dir ~/sutra-artifacts/your-repo
```

Each artifact directory contains `graph.json`, `embeddings.npy`,
`embeddings_index.json`. Put one such directory per repo under a common
root (`~/sutra-artifacts/`). That root is what the server loads.

Intra-repo call resolution (`--resolver heuristic`) is on by default — it
powers the caller/callee tools and graph expansion. For maximum resolution
on Python repos, `--resolver lsp` chains pyright type inference on top
(`pip install pyright`; resolves typed-receiver calls the heuristic leaves
ambiguous — 92% → 99% on the booth reference repo).

## 3. Run the server

### Local (stdio) — what agents on this machine spawn

```bash
python -m sutra.mcp --artifacts-dir ~/sutra-artifacts
```

### Shared team server (HTTP + bearer token)

```bash
export SUTRA_MCP_TOKEN=some-long-random-string
python -m sutra.mcp --artifacts-dir ~/sutra-artifacts --http --host 0.0.0.0 --port 8765
# endpoint: http://host:8765/mcp   (Authorization: Bearer <token>)
```

Flags / env:

| Flag | Env | Meaning |
|---|---|---|
| `--artifacts-dir` | `SUTRA_ARTIFACTS_DIR` | root containing one artifact dir per repo |
| `--http` / `--host` / `--port` | — | streamable-HTTP transport instead of stdio |
| — | `SUTRA_MCP_TOKEN` | bearer token required on every HTTP request (unset = no auth) |
| `--no-watch` | — | disable hot reload |
| `--audit-db` | — | SQLite audit log path (default `~/.sutra/mcp_audit.db`) |
| — | `OPENAI_API_KEY` | only if artifacts were embedded with an OpenAI model |

At boot the server validates every artifact (schema version, torn-artifact
cross-checks, embedding model identity) and **refuses bad ones individually**
— one broken repo never takes the server down.

## 4. Connect an agent

### Claude Code

```bash
claude mcp add sutra -- python -m sutra.mcp --artifacts-dir ~/sutra-artifacts
# or for a shared server:
claude mcp add --transport http sutra http://host:8765/mcp \
    --header "Authorization: Bearer $SUTRA_MCP_TOKEN"
```

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "sutra": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "sutra.mcp", "--artifacts-dir", "/home/you/sutra-artifacts"]
    }
  }
}
```

### Cursor (`.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "sutra": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "sutra.mcp", "--artifacts-dir", "/home/you/sutra-artifacts"]
    }
  }
}
```

## 5. Tools

| Tool | Arguments | Returns |
|---|---|---|
| `sutra_list_repos` | — | indexed repos + symbol counts + commit SHAs |
| `sutra_search` | `query`, `repo?`, `top_k=10`, `rerank=False` | ranked symbols with file/line, signature, docstring, per-channel provenance |
| `sutra_get_symbol` | `moniker` | full metadata for one symbol + its callers/callees |
| `sutra_get_callers` | `moniker` | symbols with resolved CALLS edges into it |
| `sutra_get_callees` | `moniker` | symbols it calls |
| `sutra_expand_neighbors` | `moniker`, `depth=1`, `kinds?` | BFS over the relationship graph (calls/extends/implements/references/contains/imports) |

`sutra_search` runs the full Phase 2 pipeline per query: vector ∥ BM25 ∥
moniker channels → kind filter → RRF fusion → one-hop graph expansion.
The agent workflow is: `sutra_search` to find an entry point, then
`sutra_get_symbol` / `sutra_get_callers` / `sutra_expand_neighbors` to walk
outward from it.

`rerank=True` adds a cross-encoder pass (`BAAI/bge-reranker-v2-m3`).
**On CPU this costs ~60s+ per query** — leave it off unless you're on GPU
or precision on one hard query is worth the wait.

## 6. Updating artifacts (hot reload)

The server watches each artifact directory's **`.ready` sentinel** (never
the data files) and atomically swaps in the new snapshot when it changes —
in-flight queries finish on the old snapshot.

Re-running `pipelines.full_index --output-dir <same dir>` rewrites the
artifact in place; for *remote* sync, copy data files first and touch
`.ready` **last**:

```bash
rsync -a --exclude .ready  builder:~/out/booth/  ~/sutra-artifacts/booth/
rsync -a builder:~/out/booth/.ready ~/sutra-artifacts/booth/.ready   # commit point
```

A half-copied (torn) artifact is detected by integrity cross-checks and
rejected — the previous in-memory snapshot keeps serving.

## 7. Audit log

Every tool call is recorded in SQLite (`~/.sutra/mcp_audit.db` by default):

```bash
sqlite3 ~/.sutra/mcp_audit.db \
  'SELECT ts, tool, repo, duration_ms, result_count, error FROM tool_calls ORDER BY id DESC LIMIT 20;'
```

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Embedding model mismatch` | artifact built with a different embedder than the query side constructs — the server builds the right one from artifact metadata automatically; for `openai/...` artifacts set `OPENAI_API_KEY` |
| `Torn artifact: …` | partial copy — re-sync; remember `.ready` last |
| `Unsupported schema_version` | artifact from a newer/older Sutra — re-index or upgrade |
| `skipping <repo>: …` at boot | that one artifact is broken; the rest serve normally |
| first query slow | sentence-transformers model loads lazily on first use (~5–20s), then cached |
| `No loadable artifacts` | `--artifacts-dir` must contain *subdirectories* each holding `graph.json` (+ npy + index) |
