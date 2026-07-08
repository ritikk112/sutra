# Sutra `sutra init` — Guided Installation Experience

**Status:** Approved design · **Date:** 2026-07-08 · **Branch:** `feature/cli-setup-wizard`

## Problem

Sutra is mature but has high setup friction. It is not pip-installable (no
`pyproject.toml`), everything runs as `python -m ...` from the repo root inside a
venv, embedder selection means hand-editing YAML, and there are ~6 env vars plus
optional pieces (ML extras with the CPU-torch trap, `pyright`, Postgres). New
users have no guided path. This design adds a real `sutra` console command and an
interactive `sutra init` wizard that guides users through the whole setup,
including choosing an embedder (local, OpenAI, or any OpenAI-compatible endpoint).

## Locked Decisions

1. **Distribution:** real package with a `sutra` console-script entry point
   (`pip install` / `pipx` / `uvx`), not an in-repo-only wizard.
2. **Provisioning model:** wizard *offers each step* and runs it *on consent*,
   showing the exact command first (like `gh auth login`). Never fully automatic.
3. **"Custom" embedder = any OpenAI-compatible endpoint** (Ollama, LM Studio,
   vLLM, Together, Azure, …) via a `base_url` on the existing OpenAI provider. No
   Python-plugin API.
4. **Unified CLI:** `sutra init | index | serve | ui | doctor`. New commands
   delegate to the existing entrypoints; the old `python -m` invocations keep
   working.
5. **Stack:** Typer (command tree) + Rich (styled output/progress) + questionary
   (arrow-key select menus).

## Non-Goals

- No Python-plugin embedder API (dotted-path custom classes) — deferred.
- No new database engines; Postgres stays optional and JSON-only remains default.
- No rewrite of the indexer/MCP/frontend internals — the CLI wraps them.

---

## Architecture

### 1. Packaging

Add `pyproject.toml` at repo root:

- `[project]`: name `sutra`, `requires-python >=3.11`, core dependencies migrated
  from `requirements.txt` **plus** the new CLI deps `typer`, `rich`,
  `questionary`. `requirements.txt` stays as the pinned dev-install lockfile and
  must stay in sync (same versions).
- Optional extras:
  - `sutra[local]` → `sentence-transformers` (documented, but the **wizard** does
    the correct two-step CPU-torch install rather than relying on this extra, to
    avoid the multi-GB CUDA torch trap).
- `[project.scripts]`: `sutra = "sutra.cli.main:app"`.
- Package discovery configured so `pip install -e .` from a clone and `pip
  install .` both work, permanently removing the "must run from repo root" /
  `python -m` friction.

### 2. CLI package layout (all new files)

```
sutra/cli/
  __init__.py
  main.py        # Typer app; registers init, index, serve, ui, doctor
  init.py        # the wizard (orchestrates steps)
  doctor.py      # non-interactive validation; reuses detect/validate functions
  detect.py      # read-only checks (pure functions, no prompting)
  provision.py   # consent-gated actions (pip installs, docker run, model dl)
  config_io.py   # read/merge/write config/sutra.yaml + .env (atomic, diff-shown)
  embed_setup.py # the "choose an embedder" step + live validation via probe
```

- `sutra index <path-or-url>` → imports and calls `pipelines.full_index`'s
  `main()`/callable; accepts a local path or a git URL.
- `sutra serve` → calls `sutra.mcp.__main__`'s entry (`--artifacts-dir` etc.).
- `sutra ui` → launches uvicorn on `frontend.api.main:app`.
- `sutra doctor` → runs every `detect.py` check + embedder validation
  non-interactively with ✓/✗ and fix hints.

No logic is duplicated: the command wrappers translate flags and call existing
`main()` functions. The `detect`/`validate`/`provision` functions are written
once and shared by both `init` and `doctor`.

### 3. Wizard flow (`sutra init`)

Each step: **detect → explain → ask → act-on-consent (show command first) →
verify.** Every step is skippable. The wizard is **idempotent/re-runnable**: it
loads existing `config/sutra.yaml` + `.env` and offers current values as
defaults. Config is written only at the end (step 7), atomically; Ctrl-C before
that writes nothing.

1. **Welcome + environment check** — Python version, platform, in-a-venv? (warn
   if not), docker present?, GPU present?
2. **Choose an embedder** (centerpiece; arrow-key menu with honest guidance):
   - **Local (free, offline)** — sentence-transformers `all-MiniLM-L6-v2` (384).
     If missing, offer the CPU-torch two-step install; then offer to pre-download
     the model (show size).
   - **OpenAI** — default `text-embedding-3-small` (1536); model picker for other
     embedding models. Prompt for `OPENAI_API_KEY` (masked, or "use existing env
     var"); **validate with one tiny live embed call** before proceeding.
   - **OpenAI-compatible endpoint** — prompt for `base_url`, model name, optional
     API-key env var (Ollama needs none). Make a **probe embed** to auto-discover
     `dimensions` and confirm the endpoint works.
   - Advanced (defaulted/skippable): `batch_size`.
   - Escape hatch: "skip validation" for offline setup.
3. **Artifacts directory** — default `~/.sutra/artifacts` → `SUTRA_ARTIFACTS_DIR`.
4. **Postgres (optional, clearly framed)** — "Sutra works fully without a
   database; Postgres only enables incremental re-indexing." If wanted: offer to
   build/run the existing pgvector Dockerfile with a generated password, or accept
   an existing URL; then test-connect. Writes `SUTRA_PG_URL`.
5. **Resolver / pyright** — explain heuristic vs LSP; if LSP (or the UI, which
   requires it), check for `pyright-langserver`, offer `pip install pyright`.
6. **MCP registration (optional)** — detect Claude Code; offer `claude mcp add
   sutra -- sutra serve --artifacts-dir …`; else print JSON snippet for other MCP
   clients.
7. **Write config** — `config/sutra.yaml` (embedder block) + `.env` (env vars).
   Never overwrite without showing a diff and confirming. Secrets go only in
   `.env` (gitignored), never YAML.
8. **Finale: prove it works** — offer to index a small repo now; print a summary
   card (what was configured) and the next commands (`sutra serve`, `sutra ui`).

### 4. Embedder core changes (the only change to existing core code)

- `OpenAIEmbedder` (`sutra/core/embedder/openai.py`): add optional `base_url`
  (passed straight to `openai.OpenAI(base_url=...)`). When `base_url` is set,
  `dimensions` must be provided explicitly (the wizard discovers it via probe and
  stores it in YAML).
- Config schema under `embedder:` gains optional `base_url`. `provider: openai`
  covers both real OpenAI and compatible endpoints — **no new factory branch**,
  only validation tweaks: don't require `OPENAI_API_KEY` when `base_url` is set
  and `api_key_env` is blank/empty.
- **New shared interface for the wizard (cross-agent contract):** add
  `from_dict(config: dict) -> Embedder` to `sutra/core/embedder/factory.py`
  alongside the existing `from_config(path)`. The wizard builds a candidate
  embedder from an in-memory dict and calls `.embed(["probe"])` to validate and
  read `.dimensions` — without writing a config file first. `from_config` should
  be refactored to load YAML then delegate to `from_dict`.
- `frontend/api/main.py` startup check: the hard-fail on missing
  `OPENAI_API_KEY` with `provider: openai` must respect `base_url`/`api_key_env`
  (otherwise Ollama users can't start the UI).

### 5. Error handling

- Every provisioning action prints the command, streams output, and on failure
  prints what failed + a manual-fix hint, then offers retry/skip/abort. The
  wizard never dies mid-flow on a step failure.
- API-key / endpoint validation failures loop back to re-entry; "skip validation"
  is available.
- Ctrl-C exits cleanly with no partial config (config written atomically at
  step 7 only).

### 6. Testing (no mocks — real instances, real data)

- `detect`/`validate`/`provision` are testable pure-ish functions, separated from
  prompting. Tested against the real environment: real config files in tmp dirs,
  real `FixtureEmbedder`, a **real local dummy OpenAI-compatible HTTP server**
  spun up in-test for the probe path, real pip-availability checks.
- `from_dict` gets unit coverage for each provider branch (local/openai/
  openai-compatible/fixture) and the "base_url skips key requirement" rule.
- Interactive flow: scripted-answers test via questionary's input piping.
- All tests live in `sutra/tests/`.

### 7. Docs

- README: `pip install` + `sutra init` becomes *the* quick-start.
- ENV.md: correct the stale "Postgres required" section (Postgres is optional).
- Makefile: fix the port 8001/8000 mismatch and align with `sutra ui`.

---

## Work decomposition (parallel agents, disjoint file ownership)

| Agent | Owns (files) | Contract it depends on |
|-------|--------------|------------------------|
| **A — Packaging** | `pyproject.toml` (new), `requirements.txt` (add typer/rich/questionary) | `[project.scripts] sutra = "sutra.cli.main:app"` |
| **B — Embedder** | `sutra/core/embedder/openai.py`, `factory.py`, `config/sutra.yaml`, `frontend/api/main.py`, embedder tests | Provides `factory.from_dict(dict) -> Embedder` |
| **C — CLI/wizard** | `sutra/cli/**` (all new), CLI tests | Consumes `factory.from_dict`; entry `sutra.cli.main:app` |
| **D — Docs** | `README.md`, `ENV.md`, `Makefile` | — |

Cross-agent coupling is logical only (shared via this spec); file ownership is
disjoint, so no edit conflicts. Integration step: install the package, run
`sutra doctor`, run the full test suite, reconcile the `from_dict` interface if B
and C diverge.
