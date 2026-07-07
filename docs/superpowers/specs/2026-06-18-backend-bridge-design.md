# Backend Bridge — Design Spec

- **Date:** 2026-06-18
- **Status:** Approved design, pending spec review → implementation plan
- **Scope owner:** Sutra backend
- **Reviewers consulted:** elder-agent (architecture), planner-backend-tech-lead (SOLID/DRY, failure modes)

## 1. Problem

Indexing a repo through the frontend produces artifacts the MCP server **never sees**, so "index a repo → query it over MCP" does not work end-to-end. Two root causes:

1. **Wrong layout.** The frontend writes each index to a per-*job* timestamped folder (`~/.sutra/jobs/<job-id>/out/`), but the MCP server only loads `<ARTIFACTS_DIR>/<repo>/` (one subdir per repo).
2. **No commit signal.** `JsonGraphExporter.export()` writes its three files directly and **never writes the `.ready` sentinel**. The `AtomicArtifactWriter` (staged commit + `.ready`-last + one `.prev` generation) exists in the tree but was **never wired into the export path**. The MCP watcher hot-reloads only on `.ready` changes, so even with the right layout, live reload would never fire.

A secondary correctness gap rides along: repo identity is the URL's **last path segment** (`repo_name_from_url`), so two repos named `api` from different owners collide — same artifact directory (clobber) *and* same `repo_name`, so one repo's symbols overwrite the other's in the MCP registry (and would collide on moniker primary keys if Postgres were enabled). It is also computed **twice** (`repo_name_from_url` for the moniker/registry key; the frontend's `_repo_slug` for the folder), which can drift.

## 2. Goals

- Indexing through the frontend lands each repo in a stable per-repo artifact directory the co-located MCP server hot-reloads automatically.
- Artifact publication is **atomic** and crash-consistent on a local filesystem.
- Each repo keeps a **distinct, collision-proof identity** derived from a **single** canonical function.
- The published artifact bundle is the single source of truth (the MVP runs JSON-only — no Postgres).

## 3. Non-goals (explicitly out of scope)

- Cross-service / HTTP-API linking and cross-repo call resolution (deliberately deferred; the fleet is network-coupled microservices).
- The advisory lock (dropped for MVP; see §8).
- **Postgres / persistent indexer state.** The MVP runs JSON-only (no `--pg-url`); the artifact bundle is the single source of truth. Postgres returns with incremental indexing.
- **Incremental indexing** (the frontend always full-indexes with `--replace`). Deferred to its own spec — it requires Postgres back, a re-clone + `git diff` path, the unresolved-new-relationships fix, and LSP-whole-project re-analysis.
- The frontend's private-repo / PAT UI and public/private toggle (the *next* spec; this spec only relocates output underneath the existing UI).

## 4. Decisions locked

| # | Decision | Rationale |
|---|---|---|
| D1 | **Content identity**: `repo_name_from_url()` returns the full `owner/repo` path; it is the moniker `repo_name`. | Greenfield, simple, greppable; re-index (the user's normal operation) absorbs any future rename churn. Reversible into a `repo_id` layer later. |
| D2 | **Single source of truth**: a `repo_dir_slug()` helper in `moniker.py` takes the **output of `repo_name_from_url`** and maps `/`→`__`. The frontend uses it; `_repo_slug`'s identity role is deleted. | Folder and in-graph identity cannot drift. |
| D3 | **ArtifactSink protocol** injected into `Indexer`; the exporter stays a pure serializer. | Keeps "serialize the bundle" and "commit it atomically" as separate responsibilities, matching the existing DI style (`Indexer` already injects `embedder`/`resolver`/`graph_writer`). |
| D4 | **JSON-only MVP**: no Postgres. The atomic artifact commit is the **single** publish step (the `.ready` stamp is the commit point). | Nothing reads Postgres in the MVP, so it's pure overhead and a divergence source; dropping it removes a deployed service and the PG-vs-artifact failure mode. Reversible (re-add `--pg-url`) when incremental lands. |
| D5 | **Drop per-job artifact download**; downloads serve the repo's *current* artifact, labelled as such. | Overwrite-in-place makes per-job artifact retention a silent metadata/bytes mismatch. |
| D6 | **No advisory lock for MVP**; document the single-writer assumption. | The frontend's single serialized worker makes it moot; the only real race is the artifact dir, guardable by `flock` later if a CLI ever races the queue. |
| D7 | **LSP enabled + hardened in this spec** (`--resolver lsp`): wire the per-request timeout, fail-fast on missing pyright, fail-the-index on crash, drain pyright before clone teardown. | LSP is non-negotiable for the product. Folding it in (vs a flag flip) closes the unwired-timeout / swallowed-crash gaps the reviewers found. |

## 5. Architecture

Single machine, two long-running processes, one shared local artifacts directory (`$SUTRA_ARTIFACTS_DIR`, default `~/.sutra/artifacts`). Local filesystem only — the atomic writer's `fsync` guarantees hold.

```
   ┌─────────────────────────── single machine ───────────────────────────┐
   │  Frontend API (uvicorn :8000, 0.0.0.0)        MCP server (:8765,      │
   │   ├─ git clone (PAT in-memory — next spec)     0.0.0.0, bearer token) │
   │   ├─ run pipelines.full_index                  ├─ loads all repos     │
   │   │     --output-dir $ARTIFACTS/<owner__repo>  ├─ watcher on .ready ──┐│
   │   ├─ delete clone (structure-only)             └─ serves sutra_* tools││
   │   └─ writes ─────────────────────┐                       ▲           ││
   │                                  ▼                        │ hot-reload││
   │            $SUTRA_ARTIFACTS_DIR/owner__repo/ ─────────────┘           ││
   │              graph.json  embeddings.npy  embeddings_index.json  .ready ││
   │  (no Postgres — JSON-only; the artifact bundle is the source of truth) ││
   └───────────────────────────────────────────────────────────────────────┘
        teammates' MCP clients ──HTTP+token──▶ :8765      (browsers ▶ :8000)
```

## 6. Detailed design

### 6.1 Repo identity (D1, D2)

**Canonicalization contract** (one function, tested):

`repo_name_from_url(url) -> str` returns the canonical identity:
1. Strip scheme (`http://`, `https://`, `ssh://`, `git@host:` shorthand).
2. Strip the host (everything up to the first `/`, or after `:` for SSH shorthand).
3. Strip a trailing `/` and a trailing `.git`.
4. Keep the **full remaining path** (`owner/repo`, or `group/subgroup/repo` for nested GitLab groups).
5. **Lowercase** the result.

Consequences of the contract:
- SSH and HTTPS spellings of the same repo canonicalize identically (host + scheme stripped): `git@github.com:Org/Repo.git` and `https://github.com/org/repo` → `org/repo`.
- Case variants dedupe (lowercased).
- **Known limitation:** the host is *not* part of the identity, so the same `owner/repo` on two different hosts would collide. Acceptable for a single-team fleet; documented. (Adding host is a future change if it ever bites.)

`repo_dir_slug(repo_name) -> str` (new, in `moniker.py`): maps the canonical identity to a filesystem-safe directory name by replacing `/` → `__`. `org/repo` → `org__repo`; `group/subgroup/repo` → `group__subgroup__repo`.

Identity flows from **one** place: the frontend computes the artifact dir as `repo_dir_slug(repo_name_from_url(url))`; `Indexer`/`incremental_updater` compute the moniker `repo_name` as `repo_name_from_url(url)`. Both call the same function, so the folder and the in-graph identity always agree.

**Moniker safety** (verified): the moniker stays `sutra <language> <repo_name> <file_path> <descriptor>` — five space-delimited fields. `owner/repo` contains `/` but no space, so `parse_moniker`'s `split(" ", 4)` and the MCP server's `_unit_for_moniker` (`split(" ", 3)`, `parts[2]`) resolve it unchanged.

### 6.2 ArtifactSink + atomic delivery (D3)

Define a structural protocol:

```python
class ArtifactSink(Protocol):
    def commit(
        self, artifact_dir: Path,
        write_files: Callable[[Path], None],
        generation: str = "",
    ) -> None: ...
```

`AtomicArtifactWriter` already matches this exactly (`commit(artifact_dir, write_files, generation)`), so it satisfies the protocol structurally with no change to its body.

- `Indexer.__init__` gains an injected `artifact_sink: ArtifactSink` (default `AtomicArtifactWriter()`).
- `JsonGraphExporter.export(result, vectors, output_dir, …)` stays a **pure serializer**: it writes the three files into whatever directory it is given.
- `Indexer.index()` publishes via:
  `artifact_sink.commit(output_dir, write_files=lambda staging: exporter.export(result, vectors, staging, …), generation=result.commit_hash)`.

The sink stages the three files, fsyncs, atomically `os.replace`s them into place (retaining one `<name>.prev`), fsyncs the directory, then writes `.ready` **last**. `.ready` is rewritten every commit, so its mtime always changes → the watcher always fires on re-index.

### 6.3 Publish step (D4)

The MVP runs JSON-only, so `Indexer.index()` is: extract → resolve → build chunks → embed → **publish**. Publishing is the single `artifact_sink.commit(...)` call — the atomic `.ready` stamp is the only commit point. Any earlier failure (extraction, resolver/pyright, embedding) aborts before `commit()` is reached, so nothing is published and the previous `.ready` bundle keeps serving.

`Indexer` retains its optional `graph_writer` / `pgvector_store` collaborators (already `None`-guarded), so a future deployment that passes `--pg-url` would write Postgres **before** the publish — but the MVP frontend passes no `--pg-url`, and no Postgres runs.

### 6.4 Frontend wiring (D5)

- `SUTRA_ARTIFACTS_DIR` env (default `~/.sutra/artifacts`), created at startup if absent.
- `jobs.py::enqueue` / `_run_job`: the job's `output_path` becomes `$SUTRA_ARTIFACTS_DIR/<repo_dir_slug(repo_name_from_url(repo_url))>/`. The `full_index` command runs with `--replace --resolver lsp` and **no `--pg-url`** (JSON-only). Clone deletion (structure-only) stays.
- The artifact-download endpoint serves `graph.json` / `embeddings.npy` / `embeddings_index.json` from the per-repo dir as the **current** artifact for that repo (no per-job artifact retention; SQLite still keeps per-job metadata + embedding cost).
- `_repo_slug` is retained only as the cosmetic job-id label (timestamp-disambiguated), with its identity role removed.

### 6.5 Process model / runbook

The MCP server runs co-located, pointed at the same directory, watch on:

```
python -m sutra.mcp --artifacts-dir "$SUTRA_ARTIFACTS_DIR" --http --host 0.0.0.0 --port 8765
# SUTRA_MCP_TOKEN=<long-random> set for bearer auth
```

### 6.6 LSP-grade resolution, hardened (D7)

The frontend runs `pipelines.full_index --resolver lsp`, which chains the heuristic resolver with pyright — `ChainResolver(HeuristicResolver(), LspResolver(root))`: cheap local/import/unique rules first, pyright type-inference only on the ambiguous residue. Pyright runs against the clone **before** it is deleted, so LSP-at-index-time and structure-only-at-query-time do not conflict. Python files gain pyright resolution (~99% intra-repo); non-Python files (TS/Go) fall through to the heuristic.

The resolver runs after symbol aggregation but **before** chunk-building, embedding, and the artifact publish. So a resolver failure aborts the index before the `.ready` stamp — nothing partial is ever published.

**Hardening (the gaps that make LSP unsafe today):**
- **Missing binary → fail fast at startup.** The frontend probes for pyright at startup and refuses to start when LSP is the configured resolver but pyright is absent. No silent degradation to heuristic-only — LSP is a product guarantee, so its absence is a hard error, not a fallback.
- **Per-request hang → bounded.** Wire the declared-but-unused `_DEFINITION_TIMEOUT_S` (`lsp_resolver.py:39`) into every `textDocument/definition` request. A timed-out lookup abandons that one edge (which stays heuristic-resolved) and the index continues — one slow file never hangs the job.
- **Process crash → fail the index loudly.** If pyright dies mid-run, the index fails with a clear error and publishes nothing (the atomic sink never commits). We do not silently ship a heuristic-only graph when LSP was promised.
- **Drain before teardown.** `LspResolver.resolve()` must fully drain pyright's in-flight analysis and shut the subprocess down **before** it returns, so the frontend's subsequent `rmtree` of the clone cannot race pyright reading files being deleted. No background work outlives the `resolve()` call.

Cost: indexing is slower (pyright warmup + per-file analysis). Acceptable; the frontend already streams progress.

## 7. Affected files

| File | Change |
|---|---|
| `sutra/core/extractor/moniker.py` | `repo_name_from_url` → canonical `owner/repo` (lowercased, full path); add `repo_dir_slug()`. |
| `sutra/core/output/json_graph_exporter.py` | `export()` writes into the dir it's given (already does); no commit logic added. |
| `sutra/core/artifact/__init__.py` (or new `sink.py`) | Declare `ArtifactSink` protocol. |
| `sutra/core/indexer.py` | Inject `artifact_sink`; publish via `sink.commit(..., exporter.export, ...)` as the single commit point (JSON-only — the existing PG writers stay `None`-guarded for a future deployment). |
| `sutra/core/resolver/lsp_resolver.py` | Wire `_DEFINITION_TIMEOUT_S` into definition requests; propagate a pyright crash (fail, don't swallow); guarantee subprocess drain before `resolve()` returns. |
| `pipelines/_common.py` | Construct and inject `AtomicArtifactWriter` as the default `artifact_sink`. |
| `frontend/api/jobs.py` | `output_path` → per-repo dir via the canonical helper; drop `_repo_slug` identity role; pass `--resolver lsp` to `full_index`. |
| `frontend/api/main.py` | Drop the hard `SUTRA_PG_URL` startup requirement (JSON-only); ensure `SUTRA_ARTIFACTS_DIR` exists at startup; download endpoint labelled "current"; probe pyright at startup and refuse to start if pyright is missing. |
| `tests/test_moniker.py` | Update expectations (incl. `:53` GitLab subgroup → `group/subgroup/my-app`); add canonicalization-contract + `repo_dir_slug` tests. |

`incremental_updater.py:175` (`sutra/core/incremental_updater.py`) inherits the new identity automatically and is otherwise untouched (incremental is deferred to its own spec).

## 8. Concurrency

No advisory lock (D6). The frontend's single worker serializes all indexing. **Documented assumption:** exactly one writer indexes a given repo at a time. If a hand-run CLI is ever used alongside the queue, the guard is an OS `flock` on the artifact directory held across index-and-commit (the resource that actually races) — added then, not now.

## 9. Failure modes & error handling

- **Torn artifact** (crash mid-commit): no `.ready` written → watcher never reloads it; a manual/boot load of a mixed generation is rejected by the loader's cross-checks → previous in-memory snapshot keeps serving.
- **Failed index → nothing published:** any failure before `commit()` (clone, resolver/pyright, embedding, serialization) leaves no new `.ready`, so the previous bundle keeps serving. The artifact bundle is the only persisted state (JSON-only), so there is no second store to diverge from.
- **Boot scan vs mid-commit:** `scan_artifacts_root` iterates `<root>/*` (repo dirs); `.staging` is a grandchild and `.prev` files are siblings (`graph.json.prev`), so neither is loaded as a repo. The loader reads the three named files only (no `*.json` glob) — confirm in tests.
- **Same `owner/repo` on two hosts:** documented collision (§6.1); out of scope to fix now.

## 10. Testing (real instances, no mocks)

Following the project rule — real artifacts, real code paths, no mocks, tests that catch hidden errors:

- **T1 — distinct identity:** index `team-a/api` and `team-b/api` → two artifact dirs, two `repo_name`s, both loaded as distinct repos in the MCP registry, both queryable, zero clobber.
- **T2 — re-index in place + hot-reload:** index, then re-index the same repo with `--replace` → same dir overwritten, `.ready` mtime bumps, watcher swaps in the new snapshot; in-flight queries finish on the old one.
- **T3 — atomic publish:** a `write_files` that omits a file leaves **no** `.ready` and the previous bundle still serving; a successful commit writes `.ready` last with all three files present.
- **T4 — identity contract:** `repo_name_from_url` over HTTPS/SSH/case/nested-group/trailing-slash/`.git` forms; `repo_dir_slug` round-trips; moniker parse + MCP `_unit_for_moniker` resolve an `owner/repo` repo_name.
- **T5 — LSP timeout:** a `textDocument/definition` exceeding `_DEFINITION_TIMEOUT_S` is abandoned for that one edge (left heuristic-resolved) and the index still completes.
- **T6 — LSP crash → no publish:** a pyright process failure aborts the index with a clear error and stamps no `.ready` (the previous artifact still serves).
- **T7 — drain before teardown:** `LspResolver.resolve()` returns only after the pyright subprocess has exited, so a subsequent clone deletion cannot race it.
- **T8 — missing pyright → startup refusal:** with LSP configured and pyright absent, the frontend refuses to start with a clear message.
- **Publish-is-commit-point:** an index that fails during embedding/serialization stamps no new `.ready`; the previously published bundle still serves.

## 11. Follow-up specs (not this one)

1. **Frontend private-repo UI:** public/private toggle, username + PAT (in-memory clone URL only, never logged/persisted, scrubbed from streamed logs), then index.
2. **Incremental indexing:** re-enable Postgres, add a re-clone + `git diff` update path, fix the unresolved-new-relationships gap (run resolution repo-wide on update), handle pyright whole-project re-analysis, and whole-bundle atomic re-emit.
3. Optional later: `flock` artifact-dir guard; cross-host identity; cross-service API linking.

## 12. Open questions

- **Resolved — Postgres dropped from the MVP** (JSON-only). Nothing reads it in this scope; it returns with incremental indexing (§11). This removes Postgres as a deployed service and the PG-vs-artifact divergence entirely.
- The host-in-identity and per-job-artifact-retention questions are resolved as documented limitations (§6.1, D5).
