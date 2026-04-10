# Sutra Frontend (Localhost)

Single-port local app:
- FastAPI API + worker queue + SSE logs
- React UI served by FastAPI static files

## Quick Start

```bash
cd /home/ritik/Desktop/sutra
cp .env.example .env
set -a && source .env && set +a
make ui-install
make ui-build
make ui-run
```

## Prerequisites

- Python env for this repo is active
- Node.js + npm installed
- PostgreSQL reachable via `SUTRA_PG_URL`
- `OPENAI_API_KEY` set if `config/sutra.yaml` uses `embedder.provider: openai`

## Install

```bash
cd /home/ritik/Desktop/sutra
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
cd /home/ritik/Desktop/sutra/frontend/web
npm install
npm run build
```

Or use make:

```bash
cd /home/ritik/Desktop/sutra
make ui-install
make ui-build
```

## Run

```bash
cd /home/ritik/Desktop/sutra
source .venv/bin/activate
export SUTRA_PG_URL=postgresql://postgres:postgers@localhost:5433/postgres
uvicorn frontend.api.main:app --host 127.0.0.1 --port 8000
```

Open: `http://127.0.0.1:8000`

Or use make:

```bash
cd /home/ritik/Desktop/sutra
cp .env.example .env
set -a && source .env && set +a
make ui-run
```

## Notes

- Jobs DB: `~/.sutra/jobs.db`
- Outputs: `~/.sutra/jobs/<job-id>/out/`
- Clones are temporary and are deleted after each run from `/tmp/sutra-jobs/...`
- Artifact endpoint allowlist is fixed to:
  - `graph.json`
  - `embeddings.npy`
  - `embeddings_index.json`

## Cost / Tokens

- Per-job embedding tokens and estimated USD cost are stored in SQLite (`~/.sutra/jobs.db`), not Postgres.
- Quick check:

```bash
sqlite3 ~/.sutra/jobs.db "SELECT repo_url, status, embedding_total_tokens, embedding_estimated_cost_usd, created_at FROM jobs ORDER BY created_at DESC LIMIT 20;"
```

- Aggregate endpoint:

```bash
curl http://127.0.0.1:8000/api/jobs/cost-summary
```
