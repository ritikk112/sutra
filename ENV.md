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

**Missing — must install before use:**
- `sentence-transformers` — only if using local embedder (Phase 1 can skip)

## PostgreSQL

- **Version:** 16
- **Host:** `localhost`
- **Port:** `5433` (non-default — always specify in connection strings)
- **Running in:** Docker container
- **Extensions installed:** Apache AGE, pgvector

### Connection string template
```
postgresql://USER:PASSWORD@localhost:5433/DBNAME
```

### Verify extensions are active
```sql
SELECT name, default_version, installed_version
FROM pg_available_extensions
WHERE name IN ('age', 'vector');
```

## Apache AGE

- AGE supports PostgreSQL 11–16 — PG 16 is within range.
- AGE uses openCypher via a SQL wrapper. Queries look like:
  ```sql
  SELECT * FROM ag_catalog.cypher('graph_name', $$ MATCH (n) RETURN n $$) AS (n ag_catalog.agtype);
  ```
- `CREATE EXTENSION age;` must be run per database. The `ag_catalog` schema must be in `search_path`.

## pgvector

- Embedding vectors stored as `vector(dimensions)` column type.
- `CREATE EXTENSION vector;` must be run per database.
- cosine similarity: `<=>`, L2 distance: `<->`, inner product: `<#>`

## Docker

- Postgres container runs on port **5433** (mapped from container's 5432).
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

## Steps to run the pipeline:
- cd /home/ritik/Desktop/sutra
- source .venv/bin/activate
- export SUTRA_PG_URL=postgresql://postgres:postgers@localhost:5433/postgres
- git clone https://github.com/gin-gonic/gin /tmp/gin-repo

- python -m pipelines.full_index \
  --root /tmp/gin-repo \
  --repo-url https://github.com/gin-gonic/gin \
  --output-dir /tmp/gin-out \
  --pg-url $SUTRA_PG_URL