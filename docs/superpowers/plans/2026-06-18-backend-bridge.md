# Backend Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "index a repo through the frontend → it becomes queryable in the running MCP server" work end-to-end on a single machine, with collision-proof `owner/repo` identity, atomic artifact delivery, and hardened pyright LSP resolution.

**Architecture:** The indexer writes each repo's artifact bundle (`graph.json` + `embeddings.npy` + `embeddings_index.json`) into a stable per-repo directory under `$SUTRA_ARTIFACTS_DIR`, publishing atomically via an injected `ArtifactSink` that stamps a `.ready` sentinel last; the co-located MCP server hot-reloads on that sentinel. Identity is `owner/repo` derived from one canonical function. MVP is JSON-only — no Postgres.

**Tech Stack:** Python 3.11, tree-sitter, pyright (LSP), FastAPI frontend, numpy, pytest (real instances — no mocks).

**Spec:** `docs/superpowers/specs/2026-06-18-backend-bridge-design.md`

## Global Constraints

- **No mocks in tests.** Real instances, real files, real subprocesses, real artifacts. (Project testing rule.)
- **JSON-only.** The frontend indexes without `--pg-url`; no Postgres runs. Do not add Postgres reads/writes.
- **Identity is `owner/repo`, lowercased, host-stripped** — one canonical `repo_name_from_url`; the artifact folder slug derives from its output, never from the URL directly.
- **LSP (`--resolver lsp`) is non-negotiable** and must be hardened (bounded timeout, fail-the-index on crash, drain before teardown).
- **Structure-only:** clones are deleted after indexing; do not retain source.
- **Branch:** `feature/backend-bridge` (already created). Commit after every task.
- **Run tests with:** `python -m pytest` from repo root with the project venv active.

---

### Task 1: Add `repo_dir_slug` helper (additive)

**Files:**
- Modify: `sutra/core/extractor/moniker.py`
- Test: `tests/test_moniker.py`

**Interfaces:**
- Produces: `repo_dir_slug(repo_name: str) -> str` — maps a canonical `owner/repo` identity to a filesystem-safe directory name by replacing `/` with `__`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_moniker.py`:

```python
from sutra.core.extractor.moniker import repo_dir_slug


class TestRepoDirSlug:
    def test_single_owner_repo(self):
        assert repo_dir_slug("team-a/api") == "team-a__api"

    def test_nested_group(self):
        assert repo_dir_slug("group/subgroup/svc") == "group__subgroup__svc"

    def test_no_slash_passthrough(self):
        assert repo_dir_slug("standalone") == "standalone"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_moniker.py::TestRepoDirSlug -v`
Expected: FAIL with `ImportError: cannot import name 'repo_dir_slug'`

- [ ] **Step 3: Implement** — add to `sutra/core/extractor/moniker.py` directly after `repo_name_from_url`:

```python
def repo_dir_slug(repo_name: str) -> str:
    """
    Filesystem-safe directory name for a canonical `owner/repo` identity.

    The artifact *directory* cannot contain '/', but the moniker `repo_name`
    can.  Both derive from `repo_name_from_url`, so folder and in-graph
    identity never drift.

      team-a/api          -> team-a__api
      group/subgroup/svc  -> group__subgroup__svc
    """
    return repo_name.replace("/", "__")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_moniker.py::TestRepoDirSlug -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sutra/core/extractor/moniker.py tests/test_moniker.py
git commit -m "feat(moniker): add repo_dir_slug helper for artifact dir names"
```

---

### Task 2: Change `repo_name_from_url` to `owner/repo` + migrate all affected tests

**Files:**
- Modify: `sutra/core/extractor/moniker.py` (`repo_name_from_url`)
- Modify: `tests/test_moniker.py`
- Migrate: `tests/test_e2e_fixture_repo.py`, `tests/test_mcp_loader.py`, `tests/test_mcp_server.py`, `tests/test_eval_harness.py`, `tests/test_retrieval_channels.py`, `tests/test_reranker.py`, `tests/test_fusion_pipeline.py`, `tests/test_query_analyzer.py`
- Regenerate: `tests/fixtures/sample_python_repo_expected.json`

**Interfaces:**
- Produces: `repo_name_from_url(url: str) -> str` now returns the full `owner/repo` path (or `group/subgroup/repo`), lowercased, host-stripped. Every fixture indexed with `repo_url="https://github.com/test/sample_python_repo"` now has `repo_name == "test/sample_python_repo"`.

**Why one task:** changing the function breaks every test that indexes the fixture and asserts the old `sample_python_repo` identity. To keep the suite green at the commit boundary, the function change and the test migration ship together.

- [ ] **Step 1: Update the `repo_name_from_url` unit tests** — in `tests/test_moniker.py`, replace the existing assertions (the block that currently asserts `== "my-app"` / `== "sutra"` etc.) with the new contract:

```python
class TestRepoNameFromUrl:
    def test_https_owner_repo(self):
        assert repo_name_from_url("https://github.com/org/my-app.git") == "org/my-app"

    def test_https_no_git_suffix(self):
        assert repo_name_from_url("https://github.com/org/my-app") == "org/my-app"

    def test_ssh_shorthand(self):
        assert repo_name_from_url("git@github.com:org/my-app.git") == "org/my-app"

    def test_ssh_scheme(self):
        assert repo_name_from_url("ssh://git@github.com/org/my-app") == "org/my-app"

    def test_trailing_slash(self):
        assert repo_name_from_url("https://github.com/org/my-app/") == "org/my-app"

    def test_lowercased(self):
        assert repo_name_from_url("https://github.com/Org/My-App") == "org/my-app"

    def test_nested_gitlab_group(self):
        assert repo_name_from_url(
            "https://gitlab.com/group/subgroup/my-app.git"
        ) == "group/subgroup/my-app"

    def test_internal_host_keeps_owner(self):
        assert repo_name_from_url("https://git.internal.io/team/sutra.git") == "team/sutra"
```

(Keep `TestRepoDirSlug` from Task 1. Delete any old per-URL assertions that expected a bare last segment.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_moniker.py::TestRepoNameFromUrl -v`
Expected: FAIL (current impl returns the last segment, e.g. `my-app`)

- [ ] **Step 3: Implement the new `repo_name_from_url`** — replace the function body in `sutra/core/extractor/moniker.py`:

```python
def repo_name_from_url(url: str) -> str:
    """
    Canonical repository identity: the full owner/repo path, lowercased.

      https://github.com/org/my-app.git         -> org/my-app
      git@github.com:org/my-app.git              -> org/my-app
      ssh://git@github.com/org/my-app            -> org/my-app
      https://gitlab.com/group/subgroup/app.git  -> group/subgroup/app

    Scheme, host, trailing slash and `.git` are stripped; the result is
    lowercased so SSH/HTTPS spellings and case variants of the same repo
    dedupe.  The host is NOT part of the identity (documented limitation:
    the same owner/repo on two hosts would collide).
    """
    url = url.strip().rstrip("/")

    host_stripped = False
    # SSH shorthand: git@github.com:org/repo.git -> "org/repo.git"
    if ":" in url and not url.startswith(("http://", "https://", "ssh://")):
        url = url.split(":", 1)[1]
        host_stripped = True
    else:
        for prefix in ("ssh://", "https://", "http://"):
            if url.startswith(prefix):
                url = url[len(prefix):]
                break

    # Strip the leading host segment only when a host is still present.
    if not host_stripped and "/" in url:
        url = url.split("/", 1)[1]

    if url.endswith(".git"):
        url = url[:-4]

    return url.strip("/").lower()
```

- [ ] **Step 4: Verify the moniker unit tests pass**

Run: `python -m pytest tests/test_moniker.py -v`
Expected: PASS

- [ ] **Step 5: Migrate the fixture-indexing tests** — apply these targeted replacements (each pattern is unambiguous; directory paths like `tmp_path / "sample_python_repo"` are deliberately untouched):

```bash
# Moniker repo_name segment (note the trailing space — only matches monikers)
sed -i 's#sutra python sample_python_repo #sutra python test/sample_python_repo #g' \
  tests/test_e2e_fixture_repo.py

# Identity assertions (registry keys + repo_name), per exact call shapes:
sed -i 's#registry.get("sample_python_repo")#registry.get("test/sample_python_repo")#g' \
  tests/test_mcp_loader.py tests/test_mcp_server.py
sed -i 's#repo_name == "sample_python_repo"#repo_name == "test/sample_python_repo"#g' \
  tests/test_mcp_loader.py tests/test_mcp_server.py tests/test_eval_harness.py
sed -i 's#repos() == \["sample_python_repo"\]#repos() == ["test/sample_python_repo"]#g' \
  tests/test_mcp_loader.py
sed -i 's#loaded == \["sample_python_repo"\]#loaded == ["test/sample_python_repo"]#g' \
  tests/test_mcp_loader.py
sed -i 's#repo\["name"\] == "sample_python_repo"#repo["name"] == "test/sample_python_repo"#g' \
  tests/test_e2e_fixture_repo.py
```

- [ ] **Step 6: Regenerate the e2e snapshot**

Run: `UPDATE_SNAPSHOTS=1 python -m pytest tests/test_e2e_fixture_repo.py::TestSnapshot -q`
Expected: PASS (snapshot rewritten with `test/sample_python_repo` monikers)

- [ ] **Step 7: Run the full suite and fix any straggler identity assertions**

Run: `python -m pytest -q -k "not pgvector and not sql"`
Expected: PASS. (Postgres tests are skipped — JSON-only MVP.) If a test still asserts a bare `sample_python_repo` identity, apply the same `-> test/sample_python_repo` fix to that exact line and re-run.

- [ ] **Step 8: Commit**

```bash
git add sutra/core/extractor/moniker.py tests/
git commit -m "feat(moniker)!: owner/repo identity (lowercased, host-stripped) + test migration"
```

---

### Task 3: Declare the `ArtifactSink` protocol (additive)

**Files:**
- Create: `sutra/core/artifact/sink.py`
- Modify: `sutra/core/artifact/__init__.py` (re-export)
- Test: `tests/test_artifact_sink.py`

**Interfaces:**
- Produces: `ArtifactSink` — a `runtime_checkable` `Protocol` with `commit(self, artifact_dir: Path, write_files: Callable[[Path], None], generation: str = "") -> None`. `AtomicArtifactWriter` satisfies it structurally (no change to its body).

- [ ] **Step 1: Write the failing test** — create `tests/test_artifact_sink.py`:

```python
from sutra.core.artifact import ArtifactSink
from sutra.core.artifact.atomic_writer import AtomicArtifactWriter


def test_atomic_writer_satisfies_sink_protocol():
    assert isinstance(AtomicArtifactWriter(), ArtifactSink)


def test_arbitrary_object_does_not_satisfy_protocol():
    assert not isinstance(object(), ArtifactSink)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_artifact_sink.py -v`
Expected: FAIL with `ImportError: cannot import name 'ArtifactSink'`

- [ ] **Step 3: Create the protocol** — `sutra/core/artifact/sink.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class ArtifactSink(Protocol):
    """
    Commits an artifact bundle atomically.

    `write_files(staging_dir)` must create the three artifact files inside
    the provided staging directory; the sink promotes them as a unit and
    stamps the `.ready` sentinel last.  AtomicArtifactWriter is the
    production implementation.
    """

    def commit(
        self,
        artifact_dir: Path,
        write_files: Callable[[Path], None],
        generation: str = "",
    ) -> None: ...
```

- [ ] **Step 4: Re-export from the package** — add to `sutra/core/artifact/__init__.py`:

```python
from sutra.core.artifact.sink import ArtifactSink  # noqa: F401
```

(Append to the existing imports; keep the existing `__all__`/exports intact and add `"ArtifactSink"` if an `__all__` list is present.)

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_artifact_sink.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sutra/core/artifact/sink.py sutra/core/artifact/__init__.py tests/test_artifact_sink.py
git commit -m "feat(artifact): ArtifactSink protocol (satisfied by AtomicArtifactWriter)"
```

---

### Task 4: Wire `artifact_sink` into the Indexer (atomic publish, PG-before-publish)

**Files:**
- Modify: `sutra/core/indexer.py` (`__init__`, `index`)
- Modify: `tests/test_e2e_fixture_repo.py` (`test_no_extra_files`)
- Test: `tests/test_indexer_sink.py`

**Interfaces:**
- Consumes: `ArtifactSink` (Task 3); `AtomicArtifactWriter.commit` (existing); `JsonGraphExporter.export(result, output_dir, vectors, moniker_order, embedding_usage=, embedding_model_id=)` (existing, unchanged — stays a pure serializer).
- Produces: `Indexer.__init__(..., artifact_sink: Optional[ArtifactSink] = None)`. After `index()`, `output_dir` contains the three files **plus** a `.ready` sentinel; any Postgres writers run **before** the publish.

- [ ] **Step 1: Write the failing test** — create `tests/test_indexer_sink.py`:

```python
from pathlib import Path

from sutra.core.artifact.atomic_writer import READY_SENTINEL
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter

_FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_python_repo"


def test_index_publishes_ready_sentinel_last(tmp_path):
    out = tmp_path / "repo"
    Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
    ).index(root=_FIXTURE_REPO, repo_url="https://github.com/test/svc", output_dir=out)

    assert (out / "graph.json").exists()
    assert (out / "embeddings.npy").exists()
    assert (out / "embeddings_index.json").exists()
    assert (out / READY_SENTINEL).exists()
    assert not (out / ".staging").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_indexer_sink.py -v`
Expected: FAIL (no `.ready` — the exporter still writes directly)

- [ ] **Step 3: Inject the sink** — in `sutra/core/indexer.py`, add the import near the top:

```python
from sutra.core.artifact.atomic_writer import AtomicArtifactWriter
from sutra.core.artifact.sink import ArtifactSink
```

Add the parameter to `__init__` (after `resolver`):

```python
        resolver: Optional["Resolver"] = None,
        artifact_sink: Optional[ArtifactSink] = None,
    ) -> None:
        self.adapters = adapters
        self.exporter = exporter
        self.embedder = embedder
        self.graph_writer = graph_writer
        self.pgvector_store = pgvector_store
        self.gitignore_filter = gitignore_filter
        self.resolver = resolver
        self.artifact_sink = artifact_sink or AtomicArtifactWriter()
```

- [ ] **Step 4: Reorder writes and publish via the sink** — in `index()`, replace the block that currently runs `self.exporter.export(...)` then the pgvector/graph writes (the lines from `self.exporter.export(` through `self.graph_writer.write_repository(result, replace=replace)`) with:

```python
        # Postgres first (if configured) — the transactional store commits
        # before we publish, so the .ready stamp never advertises a snapshot
        # Postgres did not record.  None-guarded: JSON-only MVP skips these.
        if self.pgvector_store is not None:
            self.pgvector_store.write(monikers, vectors)
        if self.graph_writer is not None:
            self.graph_writer.write_repository(result, replace=replace)

        # Artifact publish is the single commit point: stage the three files,
        # promote atomically, stamp .ready last.
        def _write_bundle(staging: Path) -> None:
            self.exporter.export(
                result,
                staging,
                vectors,
                monikers,
                embedding_usage=embedding_usage,
                embedding_model_id=self.embedder.model_id,
            )

        self.artifact_sink.commit(
            output_dir, _write_bundle, generation=result.commit_hash
        )
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_indexer_sink.py -v`
Expected: PASS

- [ ] **Step 6: Update the e2e "no extra files" assertion** — in `tests/test_e2e_fixture_repo.py::TestOutputFiles::test_no_extra_files`, change the expected set to include the sentinel:

```python
        assert created == {
            "graph.json", "embeddings.npy", "embeddings_index.json", ".ready",
        }
```

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -q -k "not pgvector and not sql"`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add sutra/core/indexer.py tests/test_indexer_sink.py tests/test_e2e_fixture_repo.py
git commit -m "feat(indexer): publish artifacts atomically via injected ArtifactSink"
```

---

### Task 5: Harden the LSP resolver (bounded timeout, fail-loud, drain)

**Files:**
- Modify: `sutra/core/resolver/lsp_resolver.py`
- Create: `tests/helpers/__init__.py` (empty), `tests/helpers/fake_lsp_server.py`
- Test: `tests/test_lsp_hardening.py`

**Interfaces:**
- Consumes: the existing `LspResolver(root, command=None)` and `_LspClient`.
- Produces: `LspResolver(root, command=None, definition_timeout: float = _DEFINITION_TIMEOUT_S)`; `_LspClient.request(method, params, timeout: float | None = None)` raising `TimeoutError` on deadline; `LspResolverError(RuntimeError)` raised by `resolve()` when pyright dies mid-run; `_LspClient.close()` always reaps the subprocess (bounded shutdown → kill+wait).

- [ ] **Step 1: Create the real fake LSP server** — `tests/helpers/fake_lsp_server.py` (a real subprocess, not a mock — speaks LSP framing over stdio):

```python
"""A real minimal LSP server for resolver-hardening tests (no mocks).

argv[1] mode:
  stall          - initialize OK, then never answer textDocument/definition
  crash          - initialize OK, then exit(1) on first textDocument/definition
  stall_shutdown - initialize OK, never answer shutdown (forces kill path)
"""
import json
import sys


def _read():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.decode("ascii").strip()
        if not line:
            break
        k, _, v = line.partition(":")
        headers[k.lower()] = v.strip()
    length = int(headers.get("content-length", 0))
    return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))


def _send(msg):
    data = json.dumps(msg).encode("utf-8")
    sys.stdout.buffer.write(b"Content-Length: %d\r\n\r\n" % len(data) + data)
    sys.stdout.buffer.flush()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "stall"
    while True:
        msg = _read()
        if msg is None:
            return
        method = msg.get("method")
        if method == "initialize":
            _send({"jsonrpc": "2.0", "id": msg["id"], "result": {"capabilities": {}}})
        elif method == "textDocument/definition":
            if mode == "crash":
                sys.exit(1)
            # stall: never reply -> client's timeout fires
        elif method == "shutdown":
            if mode == "stall_shutdown":
                continue   # never reply -> client kill path
            _send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
        elif method == "exit":
            return
        # initialized / didOpen / didClose / others: ignore


if __name__ == "__main__":
    main()
```

Also create empty `tests/helpers/__init__.py`.

- [ ] **Step 2: Write the failing tests** — `tests/test_lsp_hardening.py`:

```python
import sys
from pathlib import Path

import pytest

from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.core.resolver.lsp_resolver import LspResolver, LspResolverError, _LspClient

_FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_python_repo"
_FAKE = [sys.executable, str(Path(__file__).parent / "helpers" / "fake_lsp_server.py")]


def _fixture_graph(tmp_path):
    """Index the fixture WITHOUT a resolver to get unresolved Python CALLS."""
    out = tmp_path / "repo"
    result = Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
    ).index(root=_FIXTURE_REPO, repo_url="https://github.com/test/svc", output_dir=out)
    return result.symbols, result.relationships


def test_definition_timeout_is_skipped_not_hung(tmp_path):
    symbols, rels = _fixture_graph(tmp_path)
    resolver = LspResolver(
        root=_FIXTURE_REPO, command=_FAKE + ["stall"], definition_timeout=0.3
    )
    stats = resolver.resolve(symbols, rels)   # must return, not hang
    assert stats.resolved == 0                # nothing resolved by the stub


def test_pyright_crash_fails_the_index(tmp_path):
    symbols, rels = _fixture_graph(tmp_path)
    resolver = LspResolver(root=_FIXTURE_REPO, command=_FAKE + ["crash"])
    with pytest.raises(LspResolverError):
        resolver.resolve(symbols, rels)


def test_close_reaps_process_even_when_shutdown_hangs():
    client = _LspClient(_FAKE + ["stall_shutdown"], _FIXTURE_REPO)
    client.close()
    assert client._proc.poll() is not None   # process is reaped, no leak
```

- [ ] **Step 3: Run to verify they fail**

Run: `python -m pytest tests/test_lsp_hardening.py -v`
Expected: FAIL — `ImportError: cannot import name 'LspResolverError'` (and, without the timeout, `test_definition_timeout...` would hang — that is the bug being fixed).

- [ ] **Step 4: Implement the hardening** — in `sutra/core/resolver/lsp_resolver.py`:

(a) Add imports at the top (with the existing `import subprocess`, `import sys`):

```python
import select
import time
```

(b) Add the exception class just below `_DEFINITION_TIMEOUT_S = 30.0`:

```python
class LspResolverError(RuntimeError):
    """Fatal pyright failure mid-resolve — aborts the index (nothing published)."""
```

(c) Replace `_LspClient.request` with a timeout-bounded version:

```python
    def request(self, method: str, params: dict, timeout: Optional[float] = None) -> Any:
        msg_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method,
                    "params": params})
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"LSP {method} timed out after {timeout}s")
                ready, _, _ = select.select([self._proc.stdout], [], [], remaining)
                if not ready:
                    raise TimeoutError(f"LSP {method} timed out after {timeout}s")
            msg = self._recv()
            if msg is None:
                raise RuntimeError(
                    f"pyright-langserver died while waiting for {method!r}"
                )
            if msg.get("id") == msg_id and ("result" in msg or "error" in msg):
                if "error" in msg:
                    raise RuntimeError(f"LSP {method} error: {msg['error']}")
                return msg["result"]
            if "id" in msg and "method" in msg:
                self._send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
```

(d) Replace `_LspClient.close` so shutdown is bounded and the process is always reaped:

```python
    def close(self) -> None:
        try:
            self.request("shutdown", {}, timeout=5)
            self.notify("exit", {})
            self._proc.wait(timeout=5)
        except Exception:   # noqa: BLE001 — best-effort teardown
            self._proc.kill()
            self._proc.wait()
```

(e) Add `definition_timeout` to `LspResolver.__init__`:

```python
    def __init__(
        self,
        root: Path | str,
        command: Optional[list[str]] = None,
        definition_timeout: float = _DEFINITION_TIMEOUT_S,
    ) -> None:
        self._root = Path(root).resolve()
        self._definition_timeout = definition_timeout
        self._command = command or [
            str(Path(sys.executable).parent / "pyright-langserver"), "--stdio",
        ]
```

(f) In `LspResolver.resolve`, replace the `try/except RuntimeError: break` around the definition request so a timeout skips one edge and a crash is fatal:

```python
                    try:
                        result = client.request(
                            "textDocument/definition",
                            {
                                "textDocument": {"uri": uri},
                                "position": {"line": pos[0], "character": pos[1]},
                            },
                            timeout=self._definition_timeout,
                        )
                    except TimeoutError:
                        continue   # abandon this one edge; it stays heuristic-resolved
                    except RuntimeError as exc:
                        raise LspResolverError(
                            f"pyright failed mid-resolve: {exc}"
                        ) from exc
```

- [ ] **Step 5: Run to verify they pass**

Run: `python -m pytest tests/test_lsp_hardening.py -v`
Expected: PASS (all three; the timeout test returns in well under a second)

- [ ] **Step 6: Commit**

```bash
git add sutra/core/resolver/lsp_resolver.py tests/helpers/ tests/test_lsp_hardening.py
git commit -m "feat(resolver): harden LspResolver — bounded timeout, fail-loud crash, drain"
```

---

### Task 6: Frontend — per-repo output dir, `--resolver lsp`, JSON-only command

**Files:**
- Modify: `frontend/api/jobs.py` (`JobManager.__init__`, `enqueue`, extract a `_full_index_cmd` helper used by `_run_job`)
- Test: `tests/test_frontend_jobs.py`

**Interfaces:**
- Consumes: `repo_name_from_url`, `repo_dir_slug` (Tasks 1–2).
- Produces: `JobManager(repo_root, sutra_home, artifacts_root)` (the `pg_url` parameter is removed); the job's `output_path` is `artifacts_root / repo_dir_slug(repo_name_from_url(repo_url))`; the index command is `[python, -m, pipelines.full_index, --root, <clone>, --repo-url, <url>, --output-dir, <out>, --config, <cfg>, --resolver, lsp]` plus `--replace` when requested, with **no** `--pg-url`.

- [ ] **Step 1: Write the failing test** — `tests/test_frontend_jobs.py`:

```python
from pathlib import Path

from frontend.api.jobs import _full_index_cmd


def test_index_cmd_has_lsp_and_no_pg_url():
    cmd = _full_index_cmd(
        python="/usr/bin/python", root=Path("/clone"),
        repo_url="https://github.com/team-a/api",
        output_dir=Path("/art/team-a__api"), config=Path("config/sutra.yaml"),
        replace=True,
    )
    assert "--resolver" in cmd and "lsp" in cmd
    assert "--pg-url" not in cmd
    assert "--replace" in cmd
    assert cmd[cmd.index("--output-dir") + 1] == "/art/team-a__api"


def test_index_cmd_without_replace():
    cmd = _full_index_cmd(
        python="/usr/bin/python", root=Path("/clone"),
        repo_url="https://github.com/team-a/api",
        output_dir=Path("/art/team-a__api"), config=Path("config/sutra.yaml"),
        replace=False,
    )
    assert "--replace" not in cmd
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_frontend_jobs.py -v`
Expected: FAIL with `ImportError: cannot import name '_full_index_cmd'`

- [ ] **Step 3: Add the command helper** — in `frontend/api/jobs.py`, add a module-level function:

```python
def _full_index_cmd(
    python: str, root: Path, repo_url: str, output_dir: Path,
    config: Path, replace: bool,
) -> list[str]:
    """Build the JSON-only full_index command (no --pg-url; LSP on)."""
    cmd = [
        python, "-m", "pipelines.full_index",
        "--root", str(root),
        "--repo-url", repo_url,
        "--output-dir", str(output_dir),
        "--config", str(config),
        "--resolver", "lsp",
    ]
    if replace:
        cmd.append("--replace")
    return cmd
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_frontend_jobs.py -v`
Expected: PASS

- [ ] **Step 5: Wire the helper + per-repo dir into the manager** — in `frontend/api/jobs.py`:

(a) Change `JobManager.__init__` signature from `(self, repo_root, pg_url, sutra_home)` to `(self, repo_root, sutra_home, artifacts_root)`; replace `self.pg_url = pg_url` with `self.artifacts_root = artifacts_root` and ensure the dir exists:

```python
    def __init__(self, repo_root: Path, sutra_home: Path, artifacts_root: Path) -> None:
        self.repo_root = repo_root
        self.sutra_home = sutra_home
        self.artifacts_root = artifacts_root
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root = sutra_home / "jobs"
        self.db_path = sutra_home / "jobs.db"
        # ... (leave the remaining queue/runtime fields unchanged)
```

(b) In `enqueue`, compute the per-repo output dir from the canonical identity (add the import `from sutra.core.extractor.moniker import repo_dir_slug, repo_name_from_url` at the top of the file). Replace the `output_path = self.jobs_root / job_id / "out"` line with:

```python
        output_path = self.artifacts_root / repo_dir_slug(repo_name_from_url(repo_url))
```

(c) In `_run_job`, replace the inline command construction (the `cmd = [sys.executable, "-m", "pipelines.full_index", ... ]` block, including the `--pg-url` args and the `if replace: cmd.append("--replace")`) with a call to the helper:

```python
            cmd = _full_index_cmd(
                python=sys.executable,
                root=clone_path,
                repo_url=repo_url,
                output_dir=output_path,
                config=self.repo_root / "config" / "sutra.yaml",
                replace=replace,
            )
            exit_code = await self._run_cmd(job_id, runtime, cmd)
```

- [ ] **Step 6: Run the frontend job tests + full suite**

Run: `python -m pytest tests/test_frontend_jobs.py -q && python -m pytest -q -k "not pgvector and not sql"`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/api/jobs.py tests/test_frontend_jobs.py
git commit -m "feat(frontend): per-repo artifact dir + --resolver lsp + JSON-only index cmd"
```

---

### Task 7: Frontend startup — JSON-only, ensure artifacts dir, pyright probe

**Files:**
- Modify: `frontend/api/main.py` (`startup`, `health`, JobManager construction; add `_pyright_available`, `_artifacts_dir`)
- Test: `tests/test_frontend_startup.py`

**Interfaces:**
- Consumes: `JobManager(repo_root, sutra_home, artifacts_root)` (Task 6).
- Produces: `_pyright_available() -> bool`; `_artifacts_dir() -> Path` (reads `SUTRA_ARTIFACTS_DIR`, default `~/.sutra/artifacts`). Startup no longer requires `SUTRA_PG_URL`; it raises `RuntimeError` if pyright is unavailable.

- [ ] **Step 1: Write the failing test** — `tests/test_frontend_startup.py`:

```python
import os
from pathlib import Path

from frontend.api.main import _artifacts_dir, _pyright_available


def test_artifacts_dir_defaults_under_home(monkeypatch):
    monkeypatch.delenv("SUTRA_ARTIFACTS_DIR", raising=False)
    assert _artifacts_dir() == Path.home() / ".sutra" / "artifacts"


def test_artifacts_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SUTRA_ARTIFACTS_DIR", str(tmp_path / "art"))
    assert _artifacts_dir() == tmp_path / "art"


def test_pyright_probe_returns_bool():
    assert isinstance(_pyright_available(), bool)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_frontend_startup.py -v`
Expected: FAIL with `ImportError: cannot import name '_artifacts_dir'`

- [ ] **Step 3: Add the helpers** — in `frontend/api/main.py`, add `import shutil` and `import sys` if absent, then:

```python
def _artifacts_dir() -> Path:
    raw = os.environ.get("SUTRA_ARTIFACTS_DIR", "").strip()
    return Path(raw) if raw else Path.home() / ".sutra" / "artifacts"


def _pyright_available() -> bool:
    candidate = Path(sys.executable).parent / "pyright-langserver"
    return candidate.exists() or shutil.which("pyright-langserver") is not None
```

- [ ] **Step 4: Run to verify the helpers pass**

Run: `python -m pytest tests/test_frontend_startup.py -v`
Expected: PASS

- [ ] **Step 5: Rewire `startup()`** — in `frontend/api/main.py`, replace the body that requires `SUTRA_PG_URL` and calls `_check_postgres` with JSON-only startup:

```python
@app.on_event("startup")
async def startup() -> None:
    SUTRA_HOME.mkdir(parents=True, exist_ok=True)
    artifacts_root = _artifacts_dir()
    artifacts_root.mkdir(parents=True, exist_ok=True)

    _check_pipeline_importable()

    if not _pyright_available():
        raise RuntimeError(
            "pyright-langserver not found, but the indexer runs --resolver lsp. "
            "Install it: pip install pyright"
        )

    needs_openai = _config_requires_openai(REPO_ROOT / "config" / "sutra.yaml")
    openai_key_set = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if needs_openai and not openai_key_set:
        raise RuntimeError("OPENAI_API_KEY is required by config/sutra.yaml (embedder.provider=openai).")

    print(
        "[sutra-ui] startup: "
        f"repo_root={REPO_ROOT} artifacts_root={artifacts_root} "
        f"pyright_ok=True openai_required={needs_openai} openai_key_set={openai_key_set}"
    )

    global manager
    manager = JobManager(
        repo_root=REPO_ROOT, sutra_home=SUTRA_HOME, artifacts_root=artifacts_root
    )
    await manager.start()
```

Delete the now-unused `_check_postgres` import/usage and the `psycopg2` import if nothing else uses them. (Grep `psycopg2` and `_check_postgres` in `frontend/api/main.py` first; remove only if unreferenced.)

- [ ] **Step 6: Update `health()`** — replace the `pg_ok` probe with artifacts + pyright status:

```python
@app.get("/api/health")
async def health() -> dict[str, Any]:
    artifacts_root = _artifacts_dir()
    disk = shutil.disk_usage(artifacts_root if artifacts_root.exists() else SUTRA_HOME)
    return {
        "pyright_ok": _pyright_available(),
        "artifacts_dir": str(artifacts_root),
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
    }
```

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -q -k "not pgvector and not sql"`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add frontend/api/main.py tests/test_frontend_startup.py
git commit -m "feat(frontend): JSON-only startup, ensure artifacts dir, pyright probe"
```

---

### Task 8: Integration — two-repo identity + re-index hot-reload

**Files:**
- Test: `tests/test_backend_bridge_integration.py`

**Interfaces:**
- Consumes: `Indexer` (with default `AtomicArtifactWriter`), `scan_artifacts_root`/`SnapshotRegistry`/`EmbedderCache` (`sutra.mcp.registry`), `ArtifactWatcher` (`sutra.mcp.watcher`), `READY_SENTINEL`.

- [ ] **Step 1: Write the integration tests** — `tests/test_backend_bridge_integration.py`:

```python
from pathlib import Path

from sutra.core.artifact.atomic_writer import READY_SENTINEL
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.extractor.moniker import repo_dir_slug, repo_name_from_url
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.mcp.registry import EmbedderCache, SnapshotRegistry, scan_artifacts_root
from sutra.mcp.watcher import ArtifactWatcher
from sutra.core.resolver.lsp_resolver import LspResolver, LspResolverError

_FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_python_repo"


def _index(repo_url: str, artifacts_root: Path) -> Path:
    out = artifacts_root / repo_dir_slug(repo_name_from_url(repo_url))
    Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
    ).index(root=_FIXTURE_REPO, repo_url=repo_url, output_dir=out)
    return out


def test_two_same_named_repos_keep_distinct_identity(tmp_path):
    root = tmp_path / "artifacts"
    _index("https://github.com/team-a/svc", root)
    _index("https://github.com/team-b/svc", root)

    # Two distinct directories, no clobber.
    assert (root / "team-a__svc" / "graph.json").exists()
    assert (root / "team-b__svc" / "graph.json").exists()

    registry = SnapshotRegistry()
    loaded = scan_artifacts_root(root, registry, EmbedderCache(), strict=True)
    assert sorted(loaded) == ["team-a/svc", "team-b/svc"]
    assert registry.get("team-a/svc") is not None
    assert registry.get("team-b/svc") is not None


def test_crashing_lsp_aborts_index_and_publishes_nothing(tmp_path):
    """Spec T6, end-to-end: a pyright crash fails index() before the publish."""
    import sys
    import pytest
    fake = [sys.executable,
            str(Path(__file__).parent / "helpers" / "fake_lsp_server.py"), "crash"]
    out = tmp_path / "artifacts" / "team-a__svc"
    with pytest.raises(LspResolverError):
        Indexer(
            adapters={"python": PythonAdapter()},
            exporter=JsonGraphExporter(),
            embedder=FixtureEmbedder(),
            resolver=LspResolver(root=_FIXTURE_REPO, command=fake),
        ).index(root=_FIXTURE_REPO, repo_url="https://github.com/team-a/svc",
                output_dir=out)
    assert not (out / READY_SENTINEL).exists()   # crash before publish -> no .ready


def test_reindex_bumps_ready_and_watcher_fires(tmp_path):
    root = tmp_path / "artifacts"
    out = _index("https://github.com/team-a/svc", root)

    # Watcher primed on the existing sentinel — no startup re-fire.
    fired: list[Path] = []
    watcher = ArtifactWatcher(root, fired.append)
    assert watcher.check_once() == []

    # Re-index the same repo: same dir, fresh .ready (new mtime).
    import os
    before = (out / READY_SENTINEL).stat().st_mtime
    _index("https://github.com/team-a/svc", root)
    os.utime(out / READY_SENTINEL,
             (before + 5, before + 5))   # guarantee a visible mtime bump
    assert watcher.check_once() == [out]
```

- [ ] **Step 2: Run to verify they pass** (all production code already exists from Tasks 1–4)

Run: `python -m pytest tests/test_backend_bridge_integration.py -v`
Expected: PASS

- [ ] **Step 3: Final full-suite gate**

Run: `python -m pytest -q -k "not pgvector and not sql"`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_backend_bridge_integration.py
git commit -m "test(bridge): two-repo identity + re-index hot-reload integration"
```

---

## Notes for the implementer

- **Postgres tests are out of scope** (`-k "not pgvector and not sql"`). The JSON-only MVP does not run Postgres; those tests need a live DB and are unaffected by this work. Do not "fix" them by changing identity — they construct monikers with explicit repo_name strings.
- **Pre-built eval artifacts** (`tests/eval/artifacts/booth/`) are static and keep their `booth` monikers; tests that read them are unaffected. Rebuilding them (`scripts/build_eval_artifacts.py`) would re-derive identity under the new rule — out of scope here.
- **Artifact download (spec D5):** the existing `download_artifact` endpoint reads `job["outputPath"]`, which Task 6 repoints at the per-repo dir — so it already serves the repo's *current* artifact (only the three allow-listed files; `.ready`/`.prev` are not downloadable). No separate task; the relabel is a cosmetic frontend-UI concern handled in the later private-repo UI spec.
- **Manual end-to-end smoke** (after Task 8, optional): with pyright installed, run the frontend (`SUTRA_ARTIFACTS_DIR=~/.sutra/artifacts uvicorn frontend.api.main:app --port 8000`), index a small public Python repo, then start the MCP server against the same dir and confirm `sutra_list_repos` shows it.
