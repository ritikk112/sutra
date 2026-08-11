# Sutra — Environment Specifications

This file is the authoritative reference for the development environment. Update it whenever the stack changes.

## Python

- **Version:** 3.11.14
- **Virtualenv:** `/home/ritik/Desktop/sutra/.venv`
- **Activate:** `source /home/ritik/Desktop/sutra/.venv/bin/activate`

## Installed Python Packages (relevant)

| Package | Version | Notes |
|---------|---------|-------|
| `tree-sitter` | 0.25.2 | New API (node-based captures) — use `Language`, `Parser`, `Node` from this version |
| `tree-sitter-python` | 0.25.0 | Grammar for Python parsing |
| `tree-sitter-go` | 0.25.0 | Grammar for Go parsing |
| `tree-sitter-typescript` | 0.23.2 | Grammar for TypeScript/TSX parsing |
| `openai` | 2.30.0 | OpenAI SDK v2 — async client via `AsyncOpenAI` |
| `psycopg2` | 2.9.11 | Postgres driver (sync) |
| `PyYAML` | 6.0.3 | Config file parsing |
| `numpy` | 2.4.4 | Required for .npy embeddings file — confirmed installed |

| `gitpython` | 3.1.46 | Git clone/diff operations — confirmed installed |
| `rustworkx` | 0.17.1 | In-memory graph for MCP-side traversal (Phase 2) |
| `rank-bm25` | 0.2.2 | In-memory BM25 channel (Phase 2 P15) |
| `mcp` | 1.27.2 | Official MCP Python SDK (FastMCP) — Phase 2 P19 |
| `torch` | 2.12.0+cpu | CPU-only build (installed via `--index-url https://download.pytorch.org/whl/cpu` — do NOT reinstall from PyPI, the CUDA build is ~5GB) |
| `sentence-transformers` | 5.5.1 | Local embedder (all-MiniLM-L6-v2) + cross-encoder reranker |
| `fastapi` | 0.136.3 | **Upgraded from 0.116.1** — mcp requires starlette ≥1.x; old fastapi pinned starlette <0.48 |
| `starlette` | 1.2.1 | Pulled by mcp 1.27.2 |
| `pyright` | 1.1.410 | P20-full LSP resolver (`--resolver lsp`); bundles its own node runtime |

## PostgreSQL (OPTIONAL)

> **Postgres is not required.** Sutra's MVP runs **JSON-only** — index to a
> folder of artifacts and serve them from memory, no database at any stage
> (matches the README). Postgres is only needed for the deferred **incremental
> re-indexing** bookkeeping path (`--pg-url` / `SUTRA_PG_URL`); omit it and
> everything else works. The rest of this section is the optional Postgres +
> pgvector setup for that path only.

- **Version:** 16
- **Host:** `localhost`
- **Port:** `5434` (non-default — always specify in connection strings; 5432/5433/5437 in use by other Postgres containers)
- **Running in:** Docker container `sutra-postgres` (image `sutra-pgvector:latest`)
- **Volume:** named volume `sutra-pg-data` (persists across container restarts)
- **Extensions installed:** pgvector (AGE removed in Phase 2 P0)

### Connection string template
```
postgresql://USER:PASSWORD@localhost:5434/DBNAME
```

### Verify extensions are active
```sql
SELECT name, default_version, installed_version
FROM pg_available_extensions
WHERE name IN ('age', 'vector');
```

## pgvector

- Embedding vectors stored as `vector(dimensions)` column type.
- `CREATE EXTENSION vector;` must be run per database.
- cosine similarity: `<=>`, L2 distance: `<->`, inner product: `<#>`

## Docker

- Postgres container runs on port **5434** (mapped from container's 5432).
- `Dockerfile` and `docker-entrypoint-initdb.d/` are present in repo root — likely initializes AGE + pgvector extensions on first run.

## tree-sitter API Notes (v0.25.x)

The 0.25.x API differs from older versions. Key usage:

```python
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)
tree = parser.parse(source_bytes)
```

Queries use the `Language.query()` method:
```python
query = PY_LANGUAGE.query("(function_definition name: (identifier) @func.name) @func.def")
captures = query.captures(tree.root_node)
```

`captures` returns a dict of `{capture_name: list[Node]}` in v0.25.x (not a list of tuples as in older versions — confirm behavior when implementing).

## Steps to run the pipeline (JSON-only, the default):
- cd /home/ritik/Desktop/sutra
- source .venv/bin/activate
- git clone https://github.com/gin-gonic/gin /tmp/gin-repo

- python -m pipelines.full_index \
  --root /tmp/gin-repo \
  --repo-url https://github.com/gin-gonic/gin \
  --output-dir /tmp/gin-out

### Optional: with Postgres for incremental bookkeeping
Only if you're exercising the deferred incremental path — add `--pg-url`:
- export SUTRA_PG_URL=postgresql://postgres:postgers@localhost:5434/postgres
- python -m pipelines.full_index \
  --root /tmp/gin-repo \
  --repo-url https://github.com/gin-gonic/gin \
  --output-dir /tmp/gin-out \
  --pg-url $SUTRA_PG_URL

## Postgres container management

Start (if stopped):
- `docker start sutra-postgres`

Stop (without losing data — volume persists):
- `docker stop sutra-postgres`

Rebuild image (after Dockerfile changes):
- `docker stop sutra-postgres && docker rm sutra-postgres`
- `docker build -t sutra-pgvector:latest .`
- `docker run -d --name sutra-postgres -p 5434:5432 -e POSTGRES_PASSWORD=postgers -v sutra-pg-data:/var/lib/postgresql/data sutra-pgvector:latest`

Reset all data (DESTRUCTIVE):
- `docker stop sutra-postgres && docker rm sutra-postgres && docker volume rm sutra-pg-data`