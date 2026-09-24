# LocBench Launch Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce launch numbers for Sutra: index-only localization accuracy on a LocBench subset (Layer 1) and a paired grep vs grep+Sutra Claude Code A/B with adoption as a guard metric (Layer 2), with pre-registered statistics and a generated report.

**Architecture:** A small tracked package `benchmarks/locbench/` with one module per responsibility (dataset → indexing → scoring → layer1 → claude runner/transcript → layer2 → stats → report) and a `prepare.py` CLI. Layer 1 calls Sutra's in-process retrieval stack (`build_serving_unit` → `RetrievalPipeline.search`, plus `Bm25Channel` alone as the lexical baseline). Layer 2 shells out to headless `claude -p` with the runtime recipe verified in the spec (§4.1) and parses the `stream-json` transcript. Everything is resumable from files on disk; nothing is mocked in tests.

**Tech Stack:** Python 3.11, numpy, Sutra's own `sutra.core.*` / `sutra.mcp.registry`, `sutra index` CLI, git, Claude Code CLI 2.1.281 (`claude -p`), pytest. No new third-party dependencies.

**Spec:** `docs/superpowers/specs/2026-09-24-locbench-benchmark-design.md`

## Global Constraints

- Branch: `bench/locbench`. Commit after every task with the attribution lines from the session (see existing commits `0bde296`, `3219468`, `8484963` for the exact trailer format).
- Code lives in `benchmarks/locbench/` (tracked). `benchmarkings/` is gitignored — never put code there.
- Never `git add -A`; add the exact files each step names.
- Tests: real instances, real data, real code paths — **no mocks, no monkeypatching**. Tests that need a live Claude Code call are gated on `SUTRA_BENCH_LIVE=1` with `pytest.mark.skipif`; tests that need network (HF fetch) are gated on `SUTRA_BENCH_NET=1`.
- Embedder for real indexes: the shipped default `config/sutra.yaml` (`provider: local`, `BAAI/bge-base-en-v1.5`, 768d). Tests index with `provider: fixture` (a config file the plan creates) so they need no ML weights.
- `--resolver heuristic` everywhere. `rerank=False`. No graph expansion (pipeline default).
- Agent model: `claude-sonnet-5`. Runtime recipe (spec §4.1): prompt FIRST, `--setting-sources "" --disable-slash-commands --strict-mcp-config`, never `--bare`, never `--safe-mode`, env `MCP_TIMEOUT=120000`, stdin from devnull.
- Windows host (PowerShell / Git Bash). Paths in JSON are forward-slash. Use `shutil.which("claude")` — never hardcode `claude.cmd`.
- LocBench metric definition: Acc@k = 1 only if **every** gold location is in the top-k.
- Budget: Layer 2 = 30 issues × 2 arms × 3 trials, `--max-budget-usd 0.75` per run, ≤ 2 concurrent (default 1). Pilot = 5 issues × 2 arms × 1 trial, never mixed into the final 180.
- `PREREG.md` is hashed at pilot time; `report.py` refuses to run on a mismatch.

## Review Focus

1. **Gold functions that the index never produced** (decorator-generated, parse failure, file excluded by `exclude_globs` or test-file exclusion) must be counted as misses and logged, never silently dropped — Task 1 tests `missing_gold` accounting; Task 5 asserts `missing_gold.jsonl` is written.
2. **Same repo at two commits** must not share an artifact directory or a checkout — Task 3 tests that two issues from one repo produce two distinct `artifacts/<instance_id>` dirs and the manifest records both commits.
3. **A `grep_sutra` run whose server failed to connect** must be marked invalid and excluded, not scored as "Sutra unused" — Task 7 tests a transcript with `status: failed`; Task 9 tests that invalid runs are counted and excluded from Δ.
4. **Resumability**: re-running `prepare.py index` or `layer2.py` must skip completed work and never re-spend budget — Task 3 tests that `ensure_checkout` is idempotent; Task 9 step 7 re-runs the pilot and asserts every row is `resumed: true` at zero new cost (the real-transcript resume path, verified live rather than mocked).
5. **`problem_statement` with no newline / empty first line** must still yield a `title` query — Task 5 tests the title extraction on a single-line statement and a statement that starts with blank lines.

---

### Task 1: Package scaffold + `score.py` (normalizer and metrics)

**Files:**
- Create: `benchmarks/__init__.py` (empty), `benchmarks/locbench/__init__.py` (empty), `benchmarks/locbench/.gitignore`, `benchmarks/locbench/score.py`, `tests/benchmark/__init__.py` (empty), `tests/benchmark/conftest.py`, `tests/benchmark/fixture_config.yaml`
- Test: `tests/benchmark/test_score.py`

**Interfaces:**
- Produces:
  - `Key = tuple[str, str]` — `(posix_file_path, qualified_name_without_module)`
  - `normalize_gold(entry: str) -> Key` — `"pkg/mod.py:Class.method"` → `("pkg/mod.py", "Class.method")`
  - `symbol_key(sym: dict, symbols: dict[str, dict]) -> Key` — from a graph.json symbol dict (`name`, `file_path`, `enclosing_moniker`) walking the enclosing chain
  - `ranked_keys(monikers: list[str], symbols: dict[str, dict]) -> list[Key]` — dedup-preserving-order
  - `file_ranking(keys: list[Key]) -> list[str]` — unique paths in first-appearance order
  - `acc_at_k(gold: set, ranked: list, k: int) -> float` (all-of), `recall_at_k(gold, ranked, k) -> float` (fraction), `mrr(gold, ranked) -> float`
  - `score_ranking(gold_keys: set[Key], ranked: list[Key]) -> dict[str, float]` — the full metric row: `file_acc@1/3/5, func_acc@5/10, func_recall@5/10/50, file_recall@1/5, func_mrr`
  - `missing_gold(gold_keys: set[Key], symbols: dict[str, dict]) -> set[Key]` — gold keys with no symbol in the index
- `tests/benchmark/conftest.py` produces fixture `fixture_artifact_dir` (session-scoped Path) — a real index of `tests/fixtures/sample_python_repo` built with `Indexer` + `FixtureEmbedder`.

- [ ] **Step 1: Create the scaffold files**

`benchmarks/locbench/.gitignore`:
```
data/locbench_v1.json
artifacts/
repos/
```

`tests/benchmark/fixture_config.yaml`:
```yaml
# Fixture embedder — deterministic fake vectors, no ML weights (tests only).
embedder:
  provider: fixture
  dimensions: 384
```

`tests/benchmark/conftest.py`:
```python
from pathlib import Path

import pytest

from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter

FIXTURE_REPO = Path(__file__).resolve().parents[1] / "fixtures" / "sample_python_repo"
FIXTURE_CONFIG = Path(__file__).resolve().parent / "fixture_config.yaml"


@pytest.fixture(scope="session")
def fixture_artifact_dir(tmp_path_factory) -> Path:
    """A real Sutra index of tests/fixtures/sample_python_repo (fixture embedder)."""
    out = tmp_path_factory.mktemp("artifacts") / "sample"
    Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
    ).index(root=FIXTURE_REPO, repo_url="https://github.com/test/sample_python_repo", output_dir=out)
    return out
```

- [ ] **Step 2: Write the failing tests**

`tests/benchmark/test_score.py`:
```python
import json

from benchmarks.locbench.score import (
    acc_at_k,
    file_ranking,
    missing_gold,
    mrr,
    normalize_gold,
    ranked_keys,
    recall_at_k,
    score_ranking,
    symbol_key,
)


def _symbols(fixture_artifact_dir):
    graph = json.loads((fixture_artifact_dir / "graph.json").read_text(encoding="utf-8"))
    return {s["id"]: s for s in graph["symbols"]}


def test_normalize_gold_function_and_method():
    assert normalize_gold("uxarray/grid/coordinates.py:_construct_face_centroids") == (
        "uxarray/grid/coordinates.py", "_construct_face_centroids")
    assert normalize_gold("pandas/core/frame.py:DataFrame.to_dict") == ("pandas/core/frame.py", "DataFrame.to_dict")
    # Backslashes and a leading ./ are normalized away.
    assert normalize_gold(r".\pkg\mod.py:f") == ("pkg/mod.py", "f")


def test_symbol_key_walks_enclosing_chain(fixture_artifact_dir):
    symbols = _symbols(fixture_artifact_dir)
    method = symbols["sutra python test/sample_python_repo src/services/user.py UserService#create_user()."]
    assert symbol_key(method, symbols) == ("src/services/user.py", "UserService.create_user")
    func = symbols["sutra python test/sample_python_repo src/services/user.py _generate_id()."]
    assert symbol_key(func, symbols) == ("src/services/user.py", "_generate_id")
    cls = symbols["sutra python test/sample_python_repo src/services/user.py UserService#"]
    assert symbol_key(cls, symbols) == ("src/services/user.py", "UserService")


def test_ranked_keys_dedups_and_file_ranking_orders_by_first_seen(fixture_artifact_dir):
    symbols = _symbols(fixture_artifact_dir)
    m = "sutra python test/sample_python_repo src/services/user.py UserService#create_user()."
    keys = ranked_keys([m, m], symbols)
    assert keys == [("src/services/user.py", "UserService.create_user")]
    assert file_ranking([("b.py", "x"), ("a.py", "y"), ("b.py", "z")]) == ["b.py", "a.py"]


def test_acc_is_all_of_recall_is_fraction_mrr_is_first_hit():
    gold = {("a.py", "f"), ("b.py", "g")}
    ranked = [("a.py", "f"), ("c.py", "h"), ("b.py", "g")]
    assert acc_at_k(gold, ranked, 2) == 0.0
    assert acc_at_k(gold, ranked, 3) == 1.0
    assert recall_at_k(gold, ranked, 2) == 0.5
    assert mrr(gold, ranked) == 1.0
    assert mrr({("z.py", "q")}, ranked) == 0.0


def test_score_ranking_row_has_every_metric():
    gold = {("a.py", "f")}
    row = score_ranking(gold, [("x.py", "u"), ("a.py", "f")])
    assert row["file_acc@1"] == 0.0 and row["file_acc@3"] == 1.0
    assert row["func_acc@5"] == 1.0 and row["func_mrr"] == 0.5
    assert set(row) == {
        "file_acc@1", "file_acc@3", "file_acc@5", "func_acc@5", "func_acc@10",
        "func_recall@5", "func_recall@10", "func_recall@50", "file_recall@1", "file_recall@5", "func_mrr",
    }


def test_missing_gold_reports_keys_absent_from_index(fixture_artifact_dir):
    symbols = _symbols(fixture_artifact_dir)
    present = ("src/services/user.py", "UserService.create_user")
    absent = ("src/services/user.py", "does_not_exist")
    assert missing_gold({present, absent}, symbols) == {absent}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/benchmark/test_score.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks.locbench.score'`

- [ ] **Step 4: Implement `benchmarks/locbench/score.py`**

```python
"""Gold ↔ Sutra-symbol normalizer and LocBench metrics (shared by both layers)."""
from __future__ import annotations

from typing import Iterable, Sequence

Key = tuple[str, str]  # (posix file path relative to repo root, qualified name without module)

_ENCLOSING_KINDS = {"class", "function", "method"}


def _posix(path: str) -> str:
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def normalize_gold(entry: str) -> Key:
    """LocBench `edit_functions` entry `path:Qual.name` → Key."""
    path, _, qual = entry.partition(":")
    return _posix(path.strip()), qual.strip()


def symbol_key(sym: dict, symbols: dict[str, dict]) -> Key:
    """Key for a graph.json symbol dict, walking `enclosing_moniker` to build Class.method."""
    parts = [sym["name"]]
    enc = sym.get("enclosing_moniker")
    while enc:
        parent = symbols.get(enc)
        if parent is None or parent.get("kind") not in _ENCLOSING_KINDS:
            break
        parts.insert(0, parent["name"])
        enc = parent.get("enclosing_moniker")
    return _posix(sym["file_path"]), ".".join(parts)


def ranked_keys(monikers: Iterable[str], symbols: dict[str, dict]) -> list[Key]:
    out: list[Key] = []
    seen: set[Key] = set()
    for m in monikers:
        sym = symbols.get(m)
        if sym is None:
            continue
        k = symbol_key(sym, symbols)
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def file_ranking(keys: Sequence[Key]) -> list[str]:
    out: list[str] = []
    for path, _ in keys:
        if path not in out:
            out.append(path)
    return out


def acc_at_k(gold: set, ranked: Sequence, k: int) -> float:
    top = set(ranked[:k])
    return 1.0 if gold and all(g in top for g in gold) else 0.0


def recall_at_k(gold: set, ranked: Sequence, k: int) -> float:
    if not gold:
        return 0.0
    top = set(ranked[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def mrr(gold: set, ranked: Sequence) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in gold:
            return 1.0 / i
    return 0.0


def score_ranking(gold_keys: set[Key], ranked: Sequence[Key]) -> dict[str, float]:
    gold_files = {p for p, _ in gold_keys}
    files = file_ranking(ranked)
    return {
        "file_acc@1": acc_at_k(gold_files, files, 1),
        "file_acc@3": acc_at_k(gold_files, files, 3),
        "file_acc@5": acc_at_k(gold_files, files, 5),
        "func_acc@5": acc_at_k(gold_keys, ranked, 5),
        "func_acc@10": acc_at_k(gold_keys, ranked, 10),
        "func_recall@5": recall_at_k(gold_keys, ranked, 5),
        "func_recall@10": recall_at_k(gold_keys, ranked, 10),
        "func_recall@50": recall_at_k(gold_keys, ranked, 50),
        "file_recall@1": recall_at_k(gold_files, files, 1),
        "file_recall@5": recall_at_k(gold_files, files, 5),
        "func_mrr": mrr(gold_keys, ranked),
    }


def missing_gold(gold_keys: set[Key], symbols: dict[str, dict]) -> set[Key]:
    present = {symbol_key(s, symbols) for s in symbols.values()}
    return {g for g in gold_keys if g not in present}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/benchmark/test_score.py -q`
Expected: 6 passed. If `test_symbol_key_walks_enclosing_chain` fails on a KeyError for a moniker, print `sorted(symbols)` and fix the moniker string in the test to the real one (the expected monikers are listed in `tests/fixtures/sample_python_repo_expected.json`).

- [ ] **Step 6: Commit**

```bash
git add benchmarks/__init__.py benchmarks/locbench/__init__.py benchmarks/locbench/.gitignore benchmarks/locbench/score.py tests/benchmark/__init__.py tests/benchmark/conftest.py tests/benchmark/fixture_config.yaml tests/benchmark/test_score.py
git commit -m "bench(locbench): scaffold + gold/moniker normalizer and Acc@k metrics"
```

---

### Task 2: `dataset.py` — fetch LocBench with checksum, deterministic subset

**Files:**
- Create: `benchmarks/locbench/dataset.py`
- Test: `tests/benchmark/test_dataset.py`

**Interfaces:**
- Produces:
  - `HF_ROWS_URL = "https://datasets-server.huggingface.co/rows?dataset=czlll/Loc-Bench_V1&config=default&split=test&offset={offset}&length=100"`
  - `fetch_locbench(dest: Path) -> tuple[list[dict], str]` — downloads all 560 rows (stdlib urllib), writes `dest` as JSON, returns `(rows, sha256_hex)`
  - `load_locbench(path: Path, expected_sha: str | None) -> list[dict]` — raises `ValueError` on checksum mismatch
  - `select_subset(rows, *, min_issues=8, seed=20260924, n_layer2=30, max_per_repo=3) -> dict` with keys `layer1: list[instance_id]`, `layer2: list[instance_id]`, `repos: list[str]`, `params: dict`, `dataset_sha256: str | None`
  - `gold_keys_for(row: dict) -> set[Key]` — `edit_functions` (a list, or a JSON-ish string) → normalized keys
  - `title_of(row: dict) -> str` — first non-empty line of `problem_statement`, else the whole statement stripped

- [ ] **Step 1: Write the failing tests**

`tests/benchmark/test_dataset.py`:
```python
import json
import os
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.locbench.dataset import (
    fetch_locbench,
    gold_keys_for,
    load_locbench,
    select_subset,
    title_of,
)

DATA = Path(__file__).resolve().parents[2] / "benchmarks" / "locbench" / "data" / "locbench_v1.json"


def _rows():
    if DATA.exists():
        return load_locbench(DATA, None)
    if not os.environ.get("SUTRA_BENCH_NET"):
        pytest.skip("needs benchmarks/locbench/data/locbench_v1.json or SUTRA_BENCH_NET=1")
    rows, _ = fetch_locbench(DATA)
    return rows


def test_gold_keys_handles_list_and_string_forms():
    assert gold_keys_for({"edit_functions": ["a/b.py:C.m", "a/c.py:f"]}) == {("a/b.py", "C.m"), ("a/c.py", "f")}
    assert gold_keys_for({"edit_functions": "['a/b.py:C.m']"}) == {("a/b.py", "C.m")}


def test_title_of_first_nonempty_line_and_single_line():
    assert title_of({"problem_statement": "\n\nBug: crash on save\n\ndetails"}) == "Bug: crash on save"
    assert title_of({"problem_statement": "only one line"}) == "only one line"
    assert title_of({"problem_statement": "   "}) == ""


def test_real_dataset_shape_and_subset_rules():
    rows = _rows()
    assert len(rows) == 560
    sub = select_subset(rows)
    by_repo = Counter(r["repo"] for r in rows)
    assert all(by_repo[r] >= 8 for r in sub["repos"])
    assert len(sub["repos"]) == 20
    assert set(sub["layer2"]) <= set(sub["layer1"])
    assert len(sub["layer2"]) == 30
    repo_of = {r["instance_id"]: r["repo"] for r in rows}
    assert max(Counter(repo_of[i] for i in sub["layer2"]).values()) <= 3
    # Deterministic.
    assert select_subset(rows) == sub


def test_checksum_mismatch_is_an_error(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps([{"instance_id": "a"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_locbench(p, "0" * 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/benchmark/test_dataset.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `benchmarks/locbench/dataset.py`**

```python
"""LocBench V1 fetch (HF datasets-server rows API), checksum, deterministic subset."""
from __future__ import annotations

import ast
import hashlib
import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks.locbench.score import Key, normalize_gold

HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=czlll/Loc-Bench_V1"
    "&config=default&split=test&offset={offset}&length=100"
)
TOTAL_ROWS = 560
CATEGORIES = ("Bug Report", "Feature Request", "Performance Issue", "Security Vulnerability")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_locbench(dest: Path) -> tuple[list[dict], str]:
    rows: list[dict] = []
    for offset in range(0, TOTAL_ROWS, 100):
        with urllib.request.urlopen(HF_ROWS_URL.format(offset=offset), timeout=60) as resp:
            payload = json.load(resp)
        rows.extend(r["row"] for r in payload["rows"])
    if len(rows) != TOTAL_ROWS:
        raise RuntimeError(f"expected {TOTAL_ROWS} LocBench rows, got {len(rows)}")
    data = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return rows, _sha256(data)


def load_locbench(path: Path, expected_sha: str | None) -> list[dict]:
    data = path.read_bytes()
    if expected_sha is not None and _sha256(data) != expected_sha:
        raise ValueError(f"{path} sha256 mismatch: expected {expected_sha}, got {_sha256(data)}")
    return json.loads(data.decode("utf-8"))


def gold_keys_for(row: dict) -> set[Key]:
    ef = row["edit_functions"]
    if isinstance(ef, str):
        ef = ast.literal_eval(ef)
    return {normalize_gold(e) for e in ef}


def title_of(row: dict) -> str:
    for line in row["problem_statement"].splitlines():
        if line.strip():
            return line.strip()
    return row["problem_statement"].strip()


def select_subset(
    rows: list[dict],
    *,
    min_issues: int = 8,
    seed: int = 20260924,
    n_layer2: int = 30,
    max_per_repo: int = 3,
    dataset_sha256: str | None = None,
) -> dict:
    by_repo = Counter(r["repo"] for r in rows)
    repos = sorted(r for r, n in by_repo.items() if n >= min_issues)
    layer1 = sorted(r["instance_id"] for r in rows if r["repo"] in repos)

    # Stratified by category (proportional), capped per repo, fixed seed.
    rng = random.Random(seed)
    pool = [r for r in rows if r["repo"] in repos]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_cat[r["category"]].append(r)
    quota = {c: round(n_layer2 * len(by_cat[c]) / len(pool)) for c in by_cat}
    # Fix rounding drift so quotas sum to n_layer2 (adjust the largest category).
    drift = n_layer2 - sum(quota.values())
    quota[max(quota, key=quota.get)] += drift

    chosen: list[str] = []
    per_repo: Counter = Counter()
    for cat in sorted(by_cat):
        cands = sorted(by_cat[cat], key=lambda r: r["instance_id"])
        rng.shuffle(cands)
        taken = 0
        for r in cands:
            if taken >= quota[cat]:
                break
            if per_repo[r["repo"]] >= max_per_repo:
                continue
            chosen.append(r["instance_id"])
            per_repo[r["repo"]] += 1
            taken += 1
    return {
        "repos": repos,
        "layer1": layer1,
        "layer2": sorted(chosen),
        "params": {"min_issues": min_issues, "seed": seed, "n_layer2": n_layer2, "max_per_repo": max_per_repo},
        "dataset_sha256": dataset_sha256,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `SUTRA_BENCH_NET=1 python -m pytest tests/benchmark/test_dataset.py -q` (Git Bash) — the first run downloads 3 MB to `benchmarks/locbench/data/locbench_v1.json` (gitignored).
Expected: 4 passed. If `len(sub["layer2"]) == 30` fails because a category ran out of eligible candidates under the per-repo cap, lower `max_per_repo` is NOT the fix — instead top up from the largest category in a second pass and keep the test.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/locbench/dataset.py tests/benchmark/test_dataset.py
git commit -m "bench(locbench): LocBench fetch with checksum and deterministic stratified subset"
```

---

### Task 3: `indexing.py` — checkout per issue, `sutra index`, manifest

**Files:**
- Create: `benchmarks/locbench/indexing.py`
- Test: `tests/benchmark/test_indexing.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure git + `sutra index` subprocess).
- Produces:
  - `ensure_checkout(repo: str, commit: str, repos_root: Path, *, clone_url: str | None = None) -> Path` — one full clone per `(repo, commit)` at `repos_root/<owner>__<name>__<commit[:12]>` via `git clone --no-checkout` + `git checkout <commit>` (falls back to `git fetch origin <commit>` if the checkout fails). `clone_url` defaults to `https://github.com/{repo}`; tests pass a local path. Returns the checkout path. Idempotent: an existing checkout already at `commit` is returned untouched.
  - `index_issue(instance_id: str, checkout: Path, repo: str, artifacts_root: Path, config: Path, sutra_exe: str) -> dict` — runs `sutra index <checkout> --repo-url <repo> --resolver heuristic --config <config> --output-dir artifacts_root/<instance_id>`; returns a manifest row `{instance_id, repo, base_commit, artifact_dir, sutra_git_sha, embedder_model, symbol_count, file_count, index_seconds, status, error}`; `status` is `"ok"` only if `.ready` exists afterwards.
  - `read_manifest(path) -> dict[str, dict]` / `append_manifest(path, row)` — `manifest.jsonl` keyed by `instance_id`.
  - `artifact_dir_for(artifacts_root, instance_id) -> Path`

- [ ] **Step 1: Write the failing tests**

`tests/benchmark/test_indexing.py`:
```python
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from benchmarks.locbench.indexing import (
    append_manifest,
    artifact_dir_for,
    ensure_checkout,
    index_issue,
    read_manifest,
)

FIXTURE_REPO = Path(__file__).resolve().parents[1] / "fixtures" / "sample_python_repo"
FIXTURE_CONFIG = Path(__file__).resolve().parent / "fixture_config.yaml"


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def local_repo(tmp_path) -> tuple[Path, str, str]:
    """A real git repo with two commits; returns (path, commit1, commit2)."""
    src = tmp_path / "origin"
    shutil.copytree(FIXTURE_REPO, src)
    _git(src, "init", "-q")
    _git(src, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(src, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "one")
    c1 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True).stdout.strip()
    (src / "src" / "services" / "extra.py").write_text("def added():\n    return 1\n", encoding="utf-8")
    _git(src, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(src, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "two")
    c2 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True).stdout.strip()
    return src, c1, c2


def test_checkout_per_commit_is_distinct_and_idempotent(tmp_path, local_repo):
    src, c1, c2 = local_repo
    root = tmp_path / "repos"
    a = ensure_checkout("test/sample", c1, root, clone_url=str(src))
    b = ensure_checkout("test/sample", c2, root, clone_url=str(src))
    assert a != b
    assert not (a / "src" / "services" / "extra.py").exists()
    assert (b / "src" / "services" / "extra.py").exists()
    assert ensure_checkout("test/sample", c1, root, clone_url=str(src)) == a


def test_index_issue_writes_ready_bundle_and_manifest(tmp_path, local_repo):
    src, c1, c2 = local_repo
    root = tmp_path / "repos"
    arts = tmp_path / "artifacts"
    manifest = tmp_path / "manifest.jsonl"
    for iid, commit in (("test__sample-1", c1), ("test__sample-2", c2)):
        checkout = ensure_checkout("test/sample", commit, root, clone_url=str(src))
        row = index_issue(iid, checkout, "test/sample", arts, FIXTURE_CONFIG, sutra_exe="sutra")
        append_manifest(manifest, row)
        assert row["status"] == "ok", row
        assert (artifact_dir_for(arts, iid) / ".ready").exists()
        assert row["symbol_count"] > 0 and row["index_seconds"] > 0
    m = read_manifest(manifest)
    assert set(m) == {"test__sample-1", "test__sample-2"}
    assert m["test__sample-1"]["base_commit"] != m["test__sample-2"]["base_commit"]
    graph2 = json.loads((artifact_dir_for(arts, "test__sample-2") / "graph.json").read_text(encoding="utf-8"))
    assert any(s["name"] == "added" for s in graph2["symbols"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/benchmark/test_indexing.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `benchmarks/locbench/indexing.py`**

```python
"""Per-issue checkout + `sutra index` + manifest bookkeeping."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

READY = ".ready"


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _checkout_dir(repos_root: Path, repo: str, commit: str) -> Path:
    owner, _, name = repo.partition("/")
    return repos_root / f"{owner}__{name}__{commit[:12]}"


def ensure_checkout(repo: str, commit: str, repos_root: Path, *, clone_url: str | None = None) -> Path:
    dest = _checkout_dir(repos_root, repo, commit)
    if (dest / ".git").exists():
        head = _run(["git", "rev-parse", "HEAD"], cwd=dest).stdout.strip()
        if head == commit:
            return dest
    url = clone_url or f"https://github.com/{repo}"
    repos_root.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        r = _run(["git", "clone", "--quiet", "--no-checkout", url, str(dest)])
        if r.returncode != 0:
            raise RuntimeError(f"git clone failed for {url}: {r.stderr.strip()}")
    r = _run(["git", "checkout", "--quiet", commit], cwd=dest)
    if r.returncode != 0:
        # Full clone should contain it; fetch explicitly as a fallback (e.g. unreachable commit).
        _run(["git", "fetch", "--quiet", "origin", commit], cwd=dest)
        r = _run(["git", "checkout", "--quiet", commit], cwd=dest)
        if r.returncode != 0:
            raise RuntimeError(f"git checkout {commit} failed in {dest}: {r.stderr.strip()}")
    return dest


def artifact_dir_for(artifacts_root: Path, instance_id: str) -> Path:
    return artifacts_root / instance_id


def index_issue(
    instance_id: str,
    checkout: Path,
    repo: str,
    artifacts_root: Path,
    config: Path,
    sutra_exe: str = "sutra",
) -> dict:
    out = artifact_dir_for(artifacts_root, instance_id)
    out.mkdir(parents=True, exist_ok=True)
    commit = _run(["git", "rev-parse", "HEAD"], cwd=checkout).stdout.strip()
    sutra_sha = _run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    t0 = time.time()
    # `sutra index` writes to <output-dir>/<repo slug>/ — pass the issue dir as
    # the root and then flatten so the bundle sits directly in artifacts/<id>/.
    r = _run([
        sutra_exe, "index", str(checkout), "--repo-url", repo, "--resolver", "heuristic",
        "--config", str(config), "--output-dir", str(out),
    ])
    seconds = round(time.time() - t0, 1)
    row = {
        "instance_id": instance_id, "repo": repo, "base_commit": commit, "artifact_dir": str(out),
        "sutra_git_sha": sutra_sha, "embedder_model": None, "symbol_count": 0, "file_count": 0,
        "index_seconds": seconds, "status": "failed", "error": None,
    }
    if r.returncode != 0:
        row["error"] = (r.stderr or r.stdout)[-2000:]
        return row
    bundle = _flatten_bundle(out)
    if bundle is None or not (bundle / READY).exists():
        row["error"] = "no .ready sentinel after index"
        return row
    graph = json.loads((bundle / "graph.json").read_text(encoding="utf-8"))
    row.update({
        "embedder_model": graph["embeddings"]["model_id"],
        "symbol_count": len(graph["symbols"]),
        "file_count": len(graph["files"]),
        "status": "ok",
    })
    return row


def _flatten_bundle(out: Path) -> Path | None:
    """`sutra index` nests the bundle in <out>/<slug>/; move it up to <out>/ so the
    artifact dir IS the instance dir (one bundle per issue, no slug collisions)."""
    if (out / "graph.json").exists():
        return out
    subs = [p for p in out.iterdir() if p.is_dir() and (p / "graph.json").exists()]
    if len(subs) != 1:
        return None
    for child in list(subs[0].iterdir()):
        child.replace(out / child.name)
    subs[0].rmdir()
    return out


def read_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["instance_id"]] = row  # last write wins → re-index overrides
    return rows


def append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/benchmark/test_indexing.py -q`
Expected: 2 passed. If `sutra index` is not found, the venv's Scripts dir is not on PATH — pass `sutra_exe=str(Path(sys.executable).parent / "sutra.exe")` in the test instead of `"sutra"`. If `.ready` is missing on Windows, verify commit `8484963` (the stage-0 fix) is on the branch.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/locbench/indexing.py tests/benchmark/test_indexing.py
git commit -m "bench(locbench): per-issue checkout, sutra index wrapper, manifest"
```

---

### Task 4: `prepare.py` CLI + timing pilot (stage 1 gate)

**Files:**
- Create: `benchmarks/locbench/prepare.py`
- Test: `tests/benchmark/test_prepare_cli.py`

**Interfaces:**
- Consumes: `fetch_locbench, load_locbench, select_subset` (Task 2); `ensure_checkout, index_issue, read_manifest, append_manifest` (Task 3).
- Produces CLI: `python -m benchmarks.locbench.prepare fetch | select | index [--ids ID ...] [--limit N] [--layer2-only]`, and `paths()` returning the module's canonical paths: `ROOT=benchmarks/locbench`, `DATA=ROOT/data/locbench_v1.json`, `SUBSET=ROOT/subset.json`, `MANIFEST=ROOT/manifest.jsonl`, `ARTIFACTS=ROOT/artifacts`, `REPOS=ROOT/repos`, `CONFIG=config/sutra.yaml`.

- [ ] **Step 1: Write the failing test**

`tests/benchmark/test_prepare_cli.py`:
```python
import json
from pathlib import Path

from benchmarks.locbench import prepare


def test_select_writes_subset_with_checksum(tmp_path):
    # Real rows from the fixture-shaped slice: enough to satisfy min_issues on one repo.
    rows = [
        {"instance_id": f"r__a-{i}", "repo": "r/a", "category": "Bug Report", "problem_statement": "x",
         "edit_functions": ["p.py:f"], "base_commit": "c" * 40}
        for i in range(9)
    ] + [
        {"instance_id": "r__b-1", "repo": "r/b", "category": "Bug Report", "problem_statement": "x",
         "edit_functions": ["p.py:f"], "base_commit": "d" * 40}
    ]
    data = tmp_path / "locbench_v1.json"
    data.write_text(json.dumps(rows), encoding="utf-8")
    subset = tmp_path / "subset.json"
    prepare.cmd_select(data=data, subset=subset, n_layer2=3, max_per_repo=3)
    out = json.loads(subset.read_text(encoding="utf-8"))
    assert out["repos"] == ["r/a"]
    assert len(out["layer1"]) == 9 and len(out["layer2"]) == 3
    assert len(out["dataset_sha256"]) == 64


def test_paths_are_under_benchmarks_locbench():
    p = prepare.paths()
    assert p["ROOT"].name == "locbench" and p["ROOT"].parent.name == "benchmarks"
    assert p["CONFIG"].name == "sutra.yaml"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/benchmark/test_prepare_cli.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `benchmarks/locbench/prepare.py`**

```python
"""CLI: fetch dataset, select subset, index issues.  Resumable; never re-indexes status=ok."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from benchmarks.locbench.dataset import fetch_locbench, load_locbench, select_subset
from benchmarks.locbench.indexing import append_manifest, ensure_checkout, index_issue, read_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def paths() -> dict[str, Path]:
    root = REPO_ROOT / "benchmarks" / "locbench"
    return {
        "ROOT": root,
        "DATA": root / "data" / "locbench_v1.json",
        "SUBSET": root / "subset.json",
        "MANIFEST": root / "manifest.jsonl",
        "ARTIFACTS": root / "artifacts",
        "REPOS": root / "repos",
        "CONFIG": REPO_ROOT / "config" / "sutra.yaml",
    }


def sutra_exe() -> str:
    exe = Path(sys.executable).parent / ("sutra.exe" if sys.platform == "win32" else "sutra")
    return str(exe) if exe.exists() else "sutra"


def cmd_fetch(data: Path) -> None:
    rows, sha = fetch_locbench(data)
    print(f"fetched {len(rows)} rows → {data} sha256={sha}")


def cmd_select(data: Path, subset: Path, **kw) -> None:
    sha = hashlib.sha256(data.read_bytes()).hexdigest()
    rows = load_locbench(data, None)
    sel = select_subset(rows, dataset_sha256=sha, **kw)
    subset.write_text(json.dumps(sel, indent=2), encoding="utf-8")
    print(f"layer1={len(sel['layer1'])} layer2={len(sel['layer2'])} repos={len(sel['repos'])} → {subset}")


def cmd_index(data: Path, subset: Path, manifest: Path, artifacts: Path, repos: Path, config: Path,
              ids: list[str] | None, limit: int | None, layer2_only: bool) -> int:
    sel = json.loads(subset.read_text(encoding="utf-8"))
    rows = {r["instance_id"]: r for r in load_locbench(data, sel.get("dataset_sha256"))}
    todo = ids or (sel["layer2"] if layer2_only else sel["layer1"])
    done = read_manifest(manifest)
    todo = [i for i in todo if done.get(i, {}).get("status") != "ok"]
    if limit:
        todo = todo[:limit]
    failures = 0
    for n, iid in enumerate(todo, 1):
        row = rows[iid]
        t0 = time.time()
        try:
            checkout = ensure_checkout(row["repo"], row["base_commit"], repos)
            m = index_issue(iid, checkout, row["repo"], artifacts, config, sutra_exe())
        except Exception as exc:  # clone/checkout failure — record and continue
            m = {"instance_id": iid, "repo": row["repo"], "base_commit": row["base_commit"],
                 "status": "failed", "error": str(exc)[-2000:], "index_seconds": round(time.time() - t0, 1)}
        append_manifest(manifest, m)
        failures += m["status"] != "ok"
        print(f"[{n}/{len(todo)}] {iid} {m['status']} {m.get('symbol_count', 0)} symbols "
              f"{m.get('index_seconds')}s{' — ' + str(m.get('error'))[:120] if m.get('error') else ''}")
    return failures


def main(argv: list[str] | None = None) -> int:
    p = paths()
    ap = argparse.ArgumentParser(prog="prepare")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    s = sub.add_parser("select")
    s.add_argument("--min-issues", type=int, default=8)
    i = sub.add_parser("index")
    i.add_argument("--ids", nargs="*")
    i.add_argument("--limit", type=int)
    i.add_argument("--layer2-only", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        cmd_fetch(p["DATA"])
    elif a.cmd == "select":
        cmd_select(p["DATA"], p["SUBSET"], min_issues=a.min_issues)
    else:
        return 1 if cmd_index(p["DATA"], p["SUBSET"], p["MANIFEST"], p["ARTIFACTS"], p["REPOS"], p["CONFIG"],
                              a.ids, a.limit, a.layer2_only) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/benchmark/test_prepare_cli.py -q`
Expected: 2 passed

- [ ] **Step 5: Fetch + select for real, commit `subset.json`**

Run (Git Bash, repo root):
```bash
python -m benchmarks.locbench.prepare fetch
python -m benchmarks.locbench.prepare select
git add benchmarks/locbench/prepare.py benchmarks/locbench/subset.json tests/benchmark/test_prepare_cli.py
git commit -m "bench(locbench): prepare CLI + committed subset.json (20 repos, 30 layer-2 issues)"
```
Expected: `layer1≈200 layer2=30 repos=20`.

- [ ] **Step 6: TIMING PILOT (stage-1 gate) — index 3 issues and report**

Pick from `subset.json`: the first `layer1` id whose repo is `tobymao/sqlglot` (medium), the first whose repo is `django/django` (large), and the first whose repo has the fewest issues in the subset (small). Run:
```bash
python -m benchmarks.locbench.prepare index --ids <small> <medium> <large>
```
Record for each: `index_seconds`, `symbol_count`, and the clone size. The first run also downloads `BAAI/bge-base-en-v1.5` (~440 MB) once.

**Report back to the designer before continuing** with: the three timings, projected hours for all of `layer1` (`sum(sec) / 3 × len(layer1) / 3600`), and any `status: failed`. **Shrink rule (spec §2.1)**: if projected > 24 h, re-run `select` with `--min-issues` raised until `repos` = 10, commit the new `subset.json`, and note it in the report. Also confirm `embedder_model` in the manifest reads `sentence-transformers/BAAI/bge-base-en-v1.5` (or whatever the LocalEmbedder records) — Task 5 needs the artifact to load through `EmbedderCache`.

- [ ] **Step 7: Commit the manifest rows**

```bash
git add benchmarks/locbench/manifest.jsonl
git commit -m "bench(locbench): timing pilot manifest (3 issues)"
```

---

### Task 5: `layer1.py` — index-only localization run + summary

**Files:**
- Create: `benchmarks/locbench/layer1.py`
- Test: `tests/benchmark/test_layer1.py`

**Interfaces:**
- Consumes: `score.*` (Task 1), `gold_keys_for, title_of, load_locbench` (Task 2), `read_manifest, artifact_dir_for` (Task 3), `paths()` (Task 4); Sutra: `sutra.mcp.registry.build_serving_unit, EmbedderCache`, `sutra.core.retrieval.channels.bm25_channel.Bm25Channel`, `sutra.core.retrieval.query_analyzer.QueryAnalyzer`.
- Produces:
  - `retrieve(unit, query: str, retriever: str, top_k=50) -> list[str]` — monikers; `retriever ∈ {"hybrid", "bm25"}`
  - `evaluate_issue(row: dict, artifact_dir: Path, embedders: EmbedderCache) -> tuple[list[dict], set[Key]]` — 4 result rows (query ∈ {full,title} × retriever ∈ {hybrid,bm25}) and the missing-gold set
  - `run_layer1(subset_path, data_path, manifest_path, artifacts_root, out_dir) -> dict` — writes `out_dir/results.jsonl`, `out_dir/missing_gold.jsonl`, `out_dir/summary.json` (summary = per (query, retriever) mean of every metric + n, plus per-category and per-repo means; CIs are added by Task 8's `stats.py`, which `report.py` calls — `summary.json` here is raw means)

- [ ] **Step 1: Write the failing tests**

`tests/benchmark/test_layer1.py`:
```python
import json

from benchmarks.locbench.layer1 import evaluate_issue, retrieve, run_layer1
from sutra.mcp.registry import EmbedderCache, build_serving_unit


def _row(iid="test__sample-1", gold=("src/services/user.py:UserService.create_user",)):
    return {
        "instance_id": iid, "repo": "test/sample_python_repo", "category": "Bug Report",
        "base_commit": "0" * 40,
        "problem_statement": "create_user returns the wrong id\n\nUserService.create_user should call _generate_id",
        "edit_functions": list(gold),
    }


def test_retrieve_hybrid_and_bm25_return_monikers(fixture_artifact_dir):
    unit = build_serving_unit(fixture_artifact_dir, EmbedderCache())
    for retriever in ("hybrid", "bm25"):
        monikers = retrieve(unit, "create user", retriever)
        assert monikers and all(m in unit.snapshot.symbols for m in monikers)
    assert any("create_user" in m for m in retrieve(unit, "create_user", "bm25")[:3])


def test_evaluate_issue_scores_four_rows_and_reports_missing_gold(fixture_artifact_dir):
    rows, missing = evaluate_issue(_row(), fixture_artifact_dir, EmbedderCache())
    assert {(r["query"], r["retriever"]) for r in rows} == {
        ("full", "hybrid"), ("full", "bm25"), ("title", "hybrid"), ("title", "bm25")}
    assert missing == set()
    for r in rows:
        assert r["instance_id"] == "test__sample-1" and "func_acc@5" in r and len(r["ranked"]) <= 50
    # BM25 on the title (which names create_user) must reach the gold within top-50.
    bm = next(r for r in rows if r["query"] == "title" and r["retriever"] == "bm25")
    assert bm["func_recall@50"] == 1.0

    rows, missing = evaluate_issue(_row(gold=("src/services/user.py:nope",)), fixture_artifact_dir, EmbedderCache())
    assert missing == {("src/services/user.py", "nope")}
    assert all(r["func_acc@5"] == 0.0 for r in rows)  # counted as a miss, not dropped


def test_run_layer1_writes_results_and_summary(tmp_path, fixture_artifact_dir):
    data = tmp_path / "locbench_v1.json"
    data.write_text(json.dumps([_row(), _row("test__sample-2", ("src/services/user.py:nope",))]), encoding="utf-8")
    subset = tmp_path / "subset.json"
    subset.write_text(json.dumps({"layer1": ["test__sample-1", "test__sample-2"], "layer2": [], "repos": ["test/sample_python_repo"], "dataset_sha256": None}), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    arts = tmp_path / "artifacts"
    (arts / "test__sample-1").mkdir(parents=True)
    (arts / "test__sample-2").mkdir(parents=True)
    for iid in ("test__sample-1", "test__sample-2"):
        for f in fixture_artifact_dir.iterdir():
            (arts / iid / f.name).write_bytes(f.read_bytes())
        manifest.open("a", encoding="utf-8").write(json.dumps({"instance_id": iid, "status": "ok", "repo": "test/sample_python_repo"}) + "\n")
    out = tmp_path / "layer1"
    summary = run_layer1(subset, data, manifest, arts, out)
    assert (out / "results.jsonl").exists() and (out / "missing_gold.jsonl").exists()
    assert summary["n_issues"] == 2 and summary["n_missing_gold_issues"] == 1
    assert summary["cells"]["title|bm25"]["n"] == 2
    assert 0.0 <= summary["cells"]["title|bm25"]["func_recall@50"] <= 1.0
    assert "Bug Report" in summary["by_category"]["title|bm25"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/benchmark/test_layer1.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `benchmarks/locbench/layer1.py`**

```python
"""Layer 1 — index-only localization on LocBench (no agent, no API cost)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from benchmarks.locbench.dataset import gold_keys_for, load_locbench, title_of
from benchmarks.locbench.indexing import artifact_dir_for, read_manifest
from benchmarks.locbench.score import Key, missing_gold, ranked_keys, score_ranking
from sutra.core.retrieval.channels.bm25_channel import Bm25Channel
from sutra.core.retrieval.query_analyzer import QueryAnalyzer
from sutra.mcp.registry import EmbedderCache, ServingUnit, build_serving_unit

TOP_K = 50
QUERIES = ("full", "title")
RETRIEVERS = ("hybrid", "bm25")
METRICS = ("file_acc@1", "file_acc@3", "file_acc@5", "func_acc@5", "func_acc@10",
           "func_recall@5", "func_recall@10", "func_recall@50", "file_recall@1", "file_recall@5", "func_mrr")


def retrieve(unit: ServingUnit, query: str, retriever: str, top_k: int = TOP_K) -> list[str]:
    if retriever == "hybrid":
        return [r.moniker for r in unit.pipeline.search(query, top_k=top_k, rerank=False)]
    if retriever == "bm25":
        # Lexical baseline: the BM25 channel alone, no fusion, no kind boost, no embedding.
        embedder = EmbedderCache().get(unit.snapshot.embedding_model_id, unit.snapshot.embedding_dims)
        parsed = QueryAnalyzer(embedder=embedder).parse(query, embed=False)
        return [r.moniker for r in Bm25Channel(unit.snapshot).retrieve(parsed, top_k=top_k)]
    raise ValueError(retriever)


def evaluate_issue(row: dict, artifact_dir: Path, embedders: EmbedderCache) -> tuple[list[dict], set[Key]]:
    unit = build_serving_unit(artifact_dir, embedders)
    symbols = unit.snapshot.symbols
    gold = gold_keys_for(row)
    missing = missing_gold(gold, symbols)
    texts = {"full": row["problem_statement"], "title": title_of(row)}
    out = []
    for q in QUERIES:
        for r in RETRIEVERS:
            ranked = ranked_keys(retrieve(unit, texts[q], r), symbols)
            rec = {"instance_id": row["instance_id"], "repo": row["repo"], "category": row["category"],
                   "query": q, "retriever": r, "gold": sorted(gold), "ranked": ranked,
                   "n_missing_gold": len(missing)}
            rec.update(score_ranking(gold, ranked))
            out.append(rec)
    return out, missing


def summarize(results: list[dict]) -> dict:
    cells: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        cells[f"{r['query']}|{r['retriever']}"].append(r)

    def mean_block(rows: list[dict]) -> dict:
        return {"n": len(rows), **{m: sum(x[m] for x in rows) / len(rows) for m in METRICS}}

    by_cat: dict[str, dict] = {}
    by_repo: dict[str, dict] = {}
    for cell, rows in cells.items():
        cats = defaultdict(list)
        repos = defaultdict(list)
        for r in rows:
            cats[r["category"]].append(r)
            repos[r["repo"]].append(r)
        by_cat[cell] = {c: mean_block(v) for c, v in cats.items()}
        by_repo[cell] = {c: mean_block(v) for c, v in repos.items()}
    issues = {r["instance_id"] for r in results}
    return {
        "n_issues": len(issues),
        "n_missing_gold_issues": len({r["instance_id"] for r in results if r["n_missing_gold"]}),
        "cells": {c: mean_block(v) for c, v in cells.items()},
        "by_category": by_cat,
        "by_repo": by_repo,
    }


def run_layer1(subset_path: Path, data_path: Path, manifest_path: Path, artifacts_root: Path, out_dir: Path) -> dict:
    sel = json.loads(subset_path.read_text(encoding="utf-8"))
    rows = {r["instance_id"]: r for r in load_locbench(data_path, sel.get("dataset_sha256"))}
    manifest = read_manifest(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    embedders = EmbedderCache()  # one query-time model shared across all issues
    results: list[dict] = []
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as res, \
         (out_dir / "missing_gold.jsonl").open("w", encoding="utf-8") as miss:
        for n, iid in enumerate(sel["layer1"], 1):
            if manifest.get(iid, {}).get("status") != "ok":
                print(f"[{n}] {iid} skipped: no ok index")
                continue
            recs, missing = evaluate_issue(rows[iid], artifact_dir_for(artifacts_root, iid), embedders)
            for r in recs:
                res.write(json.dumps(r, sort_keys=True) + "\n")
            if missing:
                miss.write(json.dumps({"instance_id": iid, "missing": sorted(missing)}) + "\n")
            results.extend(recs)
            print(f"[{n}/{len(sel['layer1'])}] {iid} full|hybrid func_acc@5={recs[0]['func_acc@5']:.0f}")
    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


if __name__ == "__main__":
    from benchmarks.locbench.prepare import paths
    p = paths()
    s = run_layer1(p["SUBSET"], p["DATA"], p["MANIFEST"], p["ARTIFACTS"], p["ROOT"] / "layer1")
    print(json.dumps(s["cells"], indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/benchmark/test_layer1.py -q`
Expected: 3 passed. If `ServingUnit` is not importable from `sutra.mcp.registry`, import it from wherever `build_serving_unit` returns it (`grep -n "class ServingUnit" sutra/mcp/registry.py`).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/locbench/layer1.py tests/benchmark/test_layer1.py
git commit -m "bench(locbench): layer 1 index-only localization run + summary"
```

- [ ] **Step 6: Index the full Layer 1 set and run Layer 1 (stage-2 gate)**

```bash
python -m benchmarks.locbench.prepare index          # resumable; hours — run in background
python -m benchmarks.locbench.layer1
git add benchmarks/locbench/manifest.jsonl benchmarks/locbench/layer1/results.jsonl benchmarks/locbench/layer1/missing_gold.jsonl benchmarks/locbench/layer1/summary.json
git commit -m "bench(locbench): layer 1 results on the full subset"
```
**Report back to the designer** with `summary["cells"]`, `n_missing_gold_issues`, and the count of `status: failed` indexes before starting Task 6.

---

### Task 6: `transcript.py` — parse Claude Code `stream-json`

**Files:**
- Create: `benchmarks/locbench/transcript.py`
- Test: `tests/benchmark/test_transcript.py` (uses the real recorded fixture `tests/benchmark/fixtures/stream_json_grep_sutra_connected.jsonl`)

**Interfaces:**
- Produces:
  - `@dataclass RunRecord`: `mcp_status: str | None` (`"connected"`, `"failed"`, or `None` when no `sutra` server was configured), `tools_available: list[str]`, `tool_calls: list[tuple[str, dict]]`, `sutra_calls: int`, `num_turns: int`, `total_cost_usd: float`, `duration_ms: int`, `duration_api_ms: int`, `usage: dict` (input/output/cache_read/cache_creation ints), `is_error: bool`, `result_text: str`, `answer: dict | None` (parsed structured output)
  - `parse_stream_json(path: Path) -> RunRecord`
  - `parse_answer(result_msg: dict) -> dict | None` — takes `structured_output` if present, else tries `json.loads(result)`, else `None`

- [ ] **Step 1: Write the failing tests**

`tests/benchmark/test_transcript.py`:
```python
import json
from pathlib import Path

from benchmarks.locbench.transcript import parse_answer, parse_stream_json

FIX = Path(__file__).resolve().parent / "fixtures" / "stream_json_grep_sutra_connected.jsonl"


def test_parses_real_connected_transcript():
    rec = parse_stream_json(FIX)
    assert rec.mcp_status == "connected"
    assert "mcp__sutra__sutra_search" in rec.tools_available
    assert [name for name, _ in rec.tool_calls] == ["Grep", "Grep"]
    assert rec.sutra_calls == 0
    assert rec.num_turns == 3 and rec.is_error is False
    assert rec.total_cost_usd > 0 and rec.duration_api_ms > 0
    assert set(rec.usage) == {"input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"}
    assert "run_stdio" in rec.result_text


def test_failed_server_and_missing_server(tmp_path):
    lines = [l for l in FIX.read_text(encoding="utf-8").splitlines() if l.strip()]
    init = json.loads(lines[0])
    init["mcp_servers"] = [{"name": "sutra", "status": "failed", "source": "dynamic"}]
    p = tmp_path / "failed.jsonl"
    p.write_text("\n".join([json.dumps(init), *lines[1:]]), encoding="utf-8")
    assert parse_stream_json(p).mcp_status == "failed"
    init["mcp_servers"] = []
    p.write_text("\n".join([json.dumps(init), *lines[1:]]), encoding="utf-8")
    assert parse_stream_json(p).mcp_status is None


def test_parse_answer_prefers_structured_output_then_json_text():
    assert parse_answer({"structured_output": {"functions": []}, "result": "x"}) == {"functions": []}
    assert parse_answer({"result": '{"functions": [{"path": "a.py", "name": "f"}], "files": ["a.py"]}'})["files"] == ["a.py"]
    assert parse_answer({"result": "not json"}) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/benchmark/test_transcript.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `benchmarks/locbench/transcript.py`**

```python
"""Parse a Claude Code `--output-format stream-json --verbose` transcript into one RunRecord."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


@dataclass
class RunRecord:
    mcp_status: str | None = None
    tools_available: list[str] = field(default_factory=list)
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    sutra_calls: int = 0
    num_turns: int = 0
    total_cost_usd: float = 0.0
    duration_ms: int = 0
    duration_api_ms: int = 0
    usage: dict = field(default_factory=dict)
    is_error: bool = True
    result_text: str = ""
    answer: dict | None = None
    saw_result: bool = False


def parse_answer(result_msg: dict) -> dict | None:
    so = result_msg.get("structured_output")
    if isinstance(so, dict):
        return so
    text = result_msg.get("result")
    if isinstance(text, str):
        try:
            val = json.loads(text)
            return val if isinstance(val, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def parse_stream_json(path: Path) -> RunRecord:
    rec = RunRecord()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = m.get("type")
        if t == "system" and m.get("subtype") == "init":
            servers = {s["name"]: s.get("status") for s in m.get("mcp_servers", [])}
            rec.mcp_status = servers.get("sutra")
            rec.tools_available = list(m.get("tools", []))
        elif t == "assistant":
            for block in m.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    rec.tool_calls.append((block["name"], block.get("input", {})))
        elif t == "result":
            rec.saw_result = True
            rec.num_turns = int(m.get("num_turns", 0))
            rec.total_cost_usd = float(m.get("total_cost_usd", 0.0))
            rec.duration_ms = int(m.get("duration_ms", 0))
            rec.duration_api_ms = int(m.get("duration_api_ms", 0))
            usage = m.get("usage", {})
            rec.usage = {k: int(usage.get(k, 0)) for k in USAGE_KEYS}
            rec.is_error = bool(m.get("is_error", False))
            rec.result_text = str(m.get("result") or "")
            rec.answer = parse_answer(m)
    rec.sutra_calls = sum(1 for name, _ in rec.tool_calls if name.startswith("mcp__sutra__"))
    return rec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/benchmark/test_transcript.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add benchmarks/locbench/transcript.py tests/benchmark/test_transcript.py tests/benchmark/fixtures/stream_json_grep_sutra_connected.jsonl
git commit -m "bench(locbench): stream-json transcript parser with real recorded fixture"
```

---

### Task 7: `claude_runner.py` — build the verified command, run one cell

**Files:**
- Create: `benchmarks/locbench/claude_runner.py`, `benchmarks/locbench/prompt.md`, `benchmarks/locbench/answer.schema.json`
- Test: `tests/benchmark/test_claude_runner.py`

**Interfaces:**
- Consumes: `parse_stream_json, RunRecord` (Task 6).
- Produces:
  - `ARMS = ("grep", "grep_sutra")`
  - `render_prompt(problem_statement: str) -> str` — `prompt.md` with `{{PROBLEM_STATEMENT}}` substituted
  - `write_mcp_json(dest: Path, artifacts_dir: Path, sutra_exe: str) -> Path`
  - `build_command(arm: str, prompt: str, schema_path: Path, mcp_json: Path | None, *, model="claude-sonnet-5", max_turns=30, max_budget_usd=0.75, claude_exe: str | None = None) -> list[str]`
  - `run_cell(arm, prompt, cwd: Path, schema_path, mcp_json, transcript_out: Path, *, timeout_s=1200, **kw) -> RunRecord` — runs the command with `stdin=DEVNULL`, `env["MCP_TIMEOUT"]="120000"`, stdout to `transcript_out`, stderr to `transcript_out.with_suffix(".stderr")`; returns the parsed record
  - `is_valid(arm: str, rec: RunRecord) -> tuple[bool, str]` — grep arm: `rec.saw_result and mcp_status is None and sutra_calls == 0`; grep_sutra: `rec.saw_result and mcp_status == "connected" and all six mcp__sutra__* tools in tools_available`

- [ ] **Step 1: Create `prompt.md` and `answer.schema.json`**

`benchmarks/locbench/prompt.md` (byte-identical across arms; never mentions Sutra/MCP):
```
You are working in a repository checkout (the current working directory) at the commit a GitHub issue was filed against.

Task: identify the functions or methods that must be edited to fix the issue below. Investigate the code as needed, then answer with up to 5 candidate functions, most likely first. For each give `path` (file path relative to the repository root, forward slashes) and `name` (the qualified name: `Class.method` for methods, `function_name` for module-level functions). Also list the `files` you would edit.

Issue:
"""
{{PROBLEM_STATEMENT}}
"""
```

`benchmarks/locbench/answer.schema.json`:
```json
{
  "type": "object",
  "properties": {
    "functions": {
      "type": "array",
      "maxItems": 5,
      "items": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "name": {"type": "string"}},
        "required": ["path", "name"],
        "additionalProperties": false
      }
    },
    "files": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["functions", "files"],
  "additionalProperties": false
}
```

- [ ] **Step 2: Write the failing tests**

`tests/benchmark/test_claude_runner.py`:
```python
import json
import os
import shutil
from pathlib import Path

import pytest

from benchmarks.locbench.claude_runner import (
    ARMS,
    build_command,
    is_valid,
    render_prompt,
    run_cell,
    write_mcp_json,
)
from benchmarks.locbench.transcript import RunRecord, parse_stream_json

ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "locbench"
FIX = Path(__file__).resolve().parent / "fixtures" / "stream_json_grep_sutra_connected.jsonl"
LIVE = bool(os.environ.get("SUTRA_BENCH_LIVE"))


def test_render_prompt_substitutes_and_never_mentions_sutra():
    p = render_prompt("Crash on save\n\ndetails")
    assert "Crash on save" in p and "{{" not in p
    assert "sutra" not in p.lower() and "mcp" not in p.lower()


def test_build_command_prompt_first_and_verified_flags(tmp_path):
    schema = ROOT / "answer.schema.json"
    mcp = write_mcp_json(tmp_path / "mcp.json", tmp_path / "arts", "C:/x/sutra.exe")
    cmd = build_command("grep_sutra", "PROMPT", schema, mcp, claude_exe="claude")
    assert cmd[:3] == ["claude", "-p", "PROMPT"]
    joined = " ".join(cmd)
    assert "--bare" not in joined and "--safe-mode" not in joined
    assert "--setting-sources" in cmd and cmd[cmd.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in cmd and "--disable-slash-commands" in cmd
    assert "--json-schema" in cmd and "--output-format" in cmd and "stream-json" in cmd
    assert cmd[cmd.index("--allowedTools") + 1] == "Read,Grep,Glob,mcp__sutra__*"
    grep_cmd = build_command("grep", "PROMPT", schema, None, claude_exe="claude")
    assert grep_cmd[grep_cmd.index("--allowedTools") + 1] == "Read,Grep,Glob"
    assert grep_cmd[grep_cmd.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert json.loads(mcp.read_text(encoding="utf-8"))["mcpServers"]["sutra"]["args"][0] == "serve"


def test_is_valid_rules():
    ok = parse_stream_json(FIX)
    assert is_valid("grep_sutra", ok) == (True, "")
    assert is_valid("grep", ok)[0] is False  # grep arm must not have a sutra server at all
    failed = RunRecord(mcp_status="failed", saw_result=True)
    assert is_valid("grep_sutra", failed)[0] is False
    grep_ok = RunRecord(mcp_status=None, saw_result=True, sutra_calls=0)
    assert is_valid("grep", grep_ok) == (True, "")


@pytest.mark.skipif(not LIVE or shutil.which("claude") is None, reason="set SUTRA_BENCH_LIVE=1 (costs ~$0.05)")
def test_live_grep_arm_smoke(tmp_path):
    cwd = Path(__file__).resolve().parents[1] / "fixtures" / "sample_python_repo"
    rec = run_cell("grep", render_prompt("create_user returns the wrong id; UserService.create_user should call _generate_id"),
                   cwd, ROOT / "answer.schema.json", None, tmp_path / "t.jsonl", max_turns=6, max_budget_usd=0.10)
    assert rec.saw_result and not rec.is_error
    assert rec.answer is not None and "functions" in rec.answer, rec.result_text[:300]
    assert any(f["name"].endswith("create_user") for f in rec.answer["functions"])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/benchmark/test_claude_runner.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `benchmarks/locbench/claude_runner.py`**

```python
"""Headless Claude Code invocation for one benchmark cell — the spec §4.1 recipe, verbatim."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from benchmarks.locbench.transcript import RunRecord, parse_stream_json

ARMS = ("grep", "grep_sutra")
HERE = Path(__file__).resolve().parent
SUTRA_TOOLS = ("sutra_list_repos", "sutra_search", "sutra_get_symbol",
               "sutra_get_callers", "sutra_get_callees", "sutra_expand_neighbors")
BASE_ALLOWED = "Read,Grep,Glob"
DISALLOWED = "Edit,Write,Bash,WebFetch,WebSearch,Agent,NotebookEdit"
EMPTY_MCP = '{"mcpServers":{}}'


def render_prompt(problem_statement: str) -> str:
    return (HERE / "prompt.md").read_text(encoding="utf-8").replace("{{PROBLEM_STATEMENT}}", problem_statement)


def write_mcp_json(dest: Path, artifacts_dir: Path, sutra_exe: str) -> Path:
    cfg = {"mcpServers": {"sutra": {"command": str(sutra_exe),
                                    "args": ["serve", "--artifacts-dir", str(artifacts_dir.resolve())]}}}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return dest


def build_command(arm: str, prompt: str, schema_path: Path, mcp_json: Path | None, *,
                  model: str = "claude-sonnet-5", max_turns: int = 30, max_budget_usd: float = 0.75,
                  claude_exe: str | None = None) -> list[str]:
    if arm not in ARMS:
        raise ValueError(arm)
    exe = claude_exe or shutil.which("claude")
    if not exe:
        raise RuntimeError("claude CLI not found on PATH")
    allowed = BASE_ALLOWED + (",mcp__sutra__*" if arm == "grep_sutra" else "")
    mcp_arg = str(mcp_json) if (arm == "grep_sutra" and mcp_json) else EMPTY_MCP
    # Prompt FIRST: --allowedTools/--disallowedTools are variadic and would swallow it.
    return [
        exe, "-p", prompt,
        "--setting-sources", "", "--disable-slash-commands",
        "--model", model, "--output-format", "stream-json", "--verbose",
        "--permission-mode", "dontAsk", "--max-turns", str(max_turns), "--max-budget-usd", str(max_budget_usd),
        "--json-schema", schema_path.read_text(encoding="utf-8"),
        "--strict-mcp-config", "--mcp-config", mcp_arg,
        "--allowedTools", allowed,
        "--disallowedTools", DISALLOWED,
    ]


def run_cell(arm: str, prompt: str, cwd: Path, schema_path: Path, mcp_json: Path | None,
             transcript_out: Path, *, timeout_s: int = 1200, **kw) -> RunRecord:
    cmd = build_command(arm, prompt, schema_path, mcp_json, **kw)
    env = {**os.environ, "MCP_TIMEOUT": "120000"}
    transcript_out.parent.mkdir(parents=True, exist_ok=True)
    with transcript_out.open("wb") as out, transcript_out.with_suffix(".stderr").open("wb") as err:
        try:
            subprocess.run(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            err.write(b"\n[benchmark] timeout\n")
    return parse_stream_json(transcript_out)


def is_valid(arm: str, rec: RunRecord) -> tuple[bool, str]:
    if not rec.saw_result:
        return False, "no result message"
    if arm == "grep":
        if rec.mcp_status is not None:
            return False, "grep arm had a sutra server configured"
        if rec.sutra_calls:
            return False, "grep arm called sutra"
        return True, ""
    if rec.mcp_status != "connected":
        return False, f"sutra server status={rec.mcp_status}"
    missing = [t for t in SUTRA_TOOLS if f"mcp__sutra__{t}" not in rec.tools_available]
    if missing:
        return False, f"sutra tools missing: {missing}"
    return True, ""
```

- [ ] **Step 5: Run tests to verify they pass, then the live smoke test**

Run: `python -m pytest tests/benchmark/test_claude_runner.py -q` → Expected: 3 passed, 1 skipped.
Run: `SUTRA_BENCH_LIVE=1 python -m pytest tests/benchmark/test_claude_runner.py -q -k live` → Expected: 1 passed (~$0.05).
**If `rec.answer is None` in the live test**: print `rec.result_text` and the raw `result` message keys (`python - <<EOF ... EOF` over the transcript) — the structured-output field name may differ from `structured_output` in this Claude Code version. Fix `parse_answer` to read the real field, add its name to the docstring, and re-run. Do not weaken the assertion.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/locbench/claude_runner.py benchmarks/locbench/prompt.md benchmarks/locbench/answer.schema.json tests/benchmark/test_claude_runner.py
git commit -m "bench(locbench): headless Claude Code runner with the verified flag recipe"
```

---

### Task 8: `stats.py` — cluster bootstrap, paired Δ, pass^k

**Files:**
- Create: `benchmarks/locbench/stats.py`
- Test: `tests/benchmark/test_stats.py`

**Interfaces:**
- Produces:
  - `cluster_bootstrap_mean(values: list[float], clusters: list[str], *, n_boot=10000, seed=0) -> dict` → `{"mean", "ci_lo", "ci_hi", "n", "n_clusters"}` (resample clusters with replacement; statistic = mean over all values in the resampled clusters)
  - `paired_delta(a: dict[str, float], b: dict[str, float], cluster_of: dict[str, str], **kw) -> dict` → same shape for `b[i] − a[i]` over shared ids, plus `"n_pairs"`
  - `pass_pow_k(per_id_trials: dict[str, list[float]]) -> float` — fraction of ids whose every trial == 1.0

- [ ] **Step 1: Write the failing tests**

`tests/benchmark/test_stats.py`:
```python
import random

from benchmarks.locbench.stats import cluster_bootstrap_mean, paired_delta, pass_pow_k


def test_bootstrap_mean_recovers_known_mean_and_tightens_with_n():
    rng = random.Random(1)
    small = [rng.random() for _ in range(20)]
    big = [rng.random() for _ in range(2000)]
    s = cluster_bootstrap_mean(small, [f"c{i % 5}" for i in range(20)], n_boot=2000, seed=1)
    b = cluster_bootstrap_mean(big, [f"c{i % 50}" for i in range(2000)], n_boot=2000, seed=1)
    assert abs(s["mean"] - sum(small) / 20) < 1e-9 and s["ci_lo"] <= s["mean"] <= s["ci_hi"]
    assert (b["ci_hi"] - b["ci_lo"]) < (s["ci_hi"] - s["ci_lo"])
    assert s["n"] == 20 and s["n_clusters"] == 5


def test_cluster_bootstrap_widens_ci_vs_iid_when_clusters_differ():
    # Two clusters with very different means: cluster resampling must show the uncertainty.
    vals = [0.0] * 50 + [1.0] * 50
    clustered = cluster_bootstrap_mean(vals, ["a"] * 50 + ["b"] * 50, n_boot=2000, seed=0)
    iid = cluster_bootstrap_mean(vals, [f"i{i}" for i in range(100)], n_boot=2000, seed=0)
    assert (clustered["ci_hi"] - clustered["ci_lo"]) > (iid["ci_hi"] - iid["ci_lo"])


def test_paired_delta_and_pass_pow_k():
    a = {"i1": 0.0, "i2": 1.0, "i3": 0.0}
    b = {"i1": 1.0, "i2": 1.0, "i3": 1.0, "i9": 1.0}  # i9 unpaired → ignored
    d = paired_delta(a, b, {"i1": "r", "i2": "r", "i3": "s"}, n_boot=500, seed=0)
    assert d["n_pairs"] == 3 and abs(d["mean"] - 2 / 3) < 1e-9
    assert pass_pow_k({"i1": [1.0, 1.0, 1.0], "i2": [1.0, 0.0, 1.0]}) == 0.5
    assert pass_pow_k({}) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/benchmark/test_stats.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `benchmarks/locbench/stats.py`**

```python
"""Repo-clustered bootstrap CIs, paired deltas, pass^k — spec §5."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def cluster_bootstrap_mean(values: list[float], clusters: list[str], *, n_boot: int = 10000, seed: int = 0) -> dict:
    if not values:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0, "n_clusters": 0}
    groups: dict[str, list[float]] = defaultdict(list)
    for v, c in zip(values, clusters):
        groups[c].append(v)
    names = sorted(groups)
    sums = np.array([sum(groups[c]) for c in names], dtype=float)
    counts = np.array([len(groups[c]) for c in names], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(names), size=(n_boot, len(names)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(np.mean(values)), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": len(values), "n_clusters": len(names)}


def paired_delta(a: dict[str, float], b: dict[str, float], cluster_of: dict[str, str], **kw) -> dict:
    ids = sorted(set(a) & set(b))
    deltas = [b[i] - a[i] for i in ids]
    out = cluster_bootstrap_mean(deltas, [cluster_of[i] for i in ids], **kw)
    out["n_pairs"] = len(ids)
    return out


def pass_pow_k(per_id_trials: dict[str, list[float]]) -> float:
    if not per_id_trials:
        return 0.0
    return sum(1 for t in per_id_trials.values() if t and all(x == 1.0 for x in t)) / len(per_id_trials)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/benchmark/test_stats.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add benchmarks/locbench/stats.py tests/benchmark/test_stats.py
git commit -m "bench(locbench): repo-clustered bootstrap, paired delta, pass^k"
```

---

### Task 9: `layer2.py` — cell planning, pilot decision rule, resumable runs, scoring

**Files:**
- Create: `benchmarks/locbench/layer2.py`, `benchmarks/locbench/PREREG.md`
- Test: `tests/benchmark/test_layer2.py`

**Interfaces:**
- Consumes: `paths()` (Task 4); `gold_keys_for, load_locbench` (Task 2); `read_manifest, artifact_dir_for, ensure_checkout` (Task 3); `score_ranking, normalize_gold` (Task 1); `render_prompt, write_mcp_json, run_cell, is_valid, ARMS` (Task 7); `parse_stream_json` (Task 6); `cluster_bootstrap_mean, paired_delta, pass_pow_k` (Task 8).
- Produces:
  - `plan_cells(layer2_ids: list[str], trials: int, seed: int) -> list[dict]` — `{"cell_id": f"{iid}|{arm}|t{n}", "instance_id", "arm", "trial"}`; arm order randomized per issue with `random.Random(seed + hash of id)` — use `random.Random(f"{seed}:{iid}")`
  - `answer_keys(answer: dict | None) -> list[Key]` — `functions[].{path,name}` → keys via `normalize_gold(f"{path}:{name}")`
  - `score_run(row: dict, rec: RunRecord) -> dict` — `func_acc@5, file_acc@1, func_recall@5, file_recall@1` from the answer (answer `None` → all 0)
  - `adoption_rate(results: list[dict]) -> float` — fraction of valid `grep_sutra` runs with `sutra_calls ≥ 1`
  - `run_layer2(mode: "pilot"|"full", *, trials, pilot_issues=5, concurrency=1, ...) -> dict` — writes `layer2/runs/<cell_id>.jsonl` (transcript), `layer2/results.jsonl` (one row per run: cell fields + validity + metrics + cost fields + sutra_calls + tool names), `layer2/summary.json`; in pilot mode also `layer2/pilot_decision.json` `{"adoption": x, "stop": bool}` and `layer2/prereg.sha256`; returns the summary
  - `summarize_layer2(results, cluster_of) -> dict` — per-arm means with CIs, paired Δ for `func_acc@5, file_acc@1, total_cost_usd, tokens_total, num_turns, duration_api_ms`, `pass^3` per arm, adoption metrics, invalid-run counts by reason

- [ ] **Step 1: Write `PREREG.md`** (committed BEFORE any Layer 2 run)

```markdown
# Pre-registration — Sutra LocBench Layer 2 (agent A/B)

Written before any Layer 2 run. `layer2.py --mode pilot` records this file's SHA-256 in
`layer2/prereg.sha256`; `report.py` refuses to render if the hash no longer matches.

- Dataset: LocBench V1 subset in `subset.json` (`layer2`, 30 issues; seed 20260924; ≤3 per repo; stratified by category).
- Arms: `grep` (Read/Grep/Glob) vs `grep_sutra` (same + `mcp__sutra__*`). Prompt byte-identical, never mentions Sutra.
- Model: claude-sonnet-5. Trials: 3 per cell. `--max-turns 30`, `--max-budget-usd 0.75`. Arm order randomized per issue.
- Primary outcomes: paired Δ (grep_sutra − grep) in `func_acc@5`, `file_acc@1`, `total_cost_usd`, total tokens, `num_turns`.
- Secondary: `pass^3` per arm; `duration_api_ms`.
- Guard: adoption = fraction of valid `grep_sutra` runs with ≥1 `mcp__sutra__*` call.
- CIs: 95 % cluster bootstrap over repos, 10 000 draws, seed 0.
- Validity: a run is excluded (and counted) if it has no `result` message, the `grep` arm had a Sutra server configured or called Sutra, or the `grep_sutra` arm's server was not `connected` with all six tools listed. Excluded cells are re-run once.
- Pilot decision rule: 5 issues × 2 arms × 1 trial first. If adoption < 0.50 → STOP; fix tool descriptions in the product; re-pilot. Pilot runs are never merged into the final 180.
- Claim strength: with n=30 the Δ is reported as directional, always with its CI in the same sentence.
```

- [ ] **Step 2: Write the failing tests**

`tests/benchmark/test_layer2.py`:
```python
import json
from pathlib import Path

from benchmarks.locbench.layer2 import (
    adoption_rate,
    answer_keys,
    plan_cells,
    score_run,
    summarize_layer2,
)
from benchmarks.locbench.transcript import RunRecord


def test_plan_cells_randomizes_arm_order_per_issue_deterministically():
    cells = plan_cells(["a", "b", "c", "d", "e", "f"], trials=3, seed=7)
    assert len(cells) == 36
    assert cells == plan_cells(["a", "b", "c", "d", "e", "f"], trials=3, seed=7)
    first_arm: dict[str, str] = {}
    for c in cells:  # first cell listed per issue = the arm that runs first for it
        first_arm.setdefault(c["instance_id"], c["arm"])
    # Not every issue starts with the same arm (6 issues, P(all same) = 1/32 — with seed 7 this holds; keep it).
    assert len(set(first_arm.values())) == 2
    assert {c["cell_id"] for c in cells} >= {"a|grep|t1", "a|grep_sutra|t3"}


def test_answer_keys_and_score_run():
    row = {"edit_functions": ["pkg/a.py:C.m", "pkg/b.py:f"]}
    rec = RunRecord(answer={"functions": [{"path": "pkg/a.py", "name": "C.m"}, {"path": "pkg/b.py", "name": "f"}], "files": ["pkg/a.py"]})
    assert answer_keys(rec.answer) == [("pkg/a.py", "C.m"), ("pkg/b.py", "f")]
    s = score_run(row, rec)
    assert s["func_acc@5"] == 1.0 and s["file_acc@1"] == 0.0 and s["func_recall@5"] == 1.0
    assert score_run(row, RunRecord(answer=None))["func_acc@5"] == 0.0


def test_adoption_and_summary_exclude_invalid_runs():
    base = {"total_cost_usd": 0.1, "tokens_total": 100, "num_turns": 5, "duration_api_ms": 1000,
            "file_acc@1": 1.0, "valid": True, "invalid_reason": ""}
    results = []
    for iid, repo in (("i1", "r"), ("i2", "r"), ("i3", "s")):
        for t in (1, 2, 3):
            results.append({**base, "instance_id": iid, "repo": repo, "arm": "grep", "trial": t, "sutra_calls": 0, "func_acc@5": 0.0})
            results.append({**base, "instance_id": iid, "repo": repo, "arm": "grep_sutra", "trial": t, "sutra_calls": 2, "func_acc@5": 1.0})
    results.append({**base, "instance_id": "i1", "repo": "r", "arm": "grep_sutra", "trial": 4, "sutra_calls": 0,
                    "func_acc@5": 0.0, "valid": False, "invalid_reason": "sutra server status=failed"})
    assert adoption_rate(results) == 1.0
    s = summarize_layer2(results, {"i1": "r", "i2": "r", "i3": "s"})
    assert s["delta"]["func_acc@5"]["mean"] == 1.0 and s["delta"]["func_acc@5"]["n_pairs"] == 3
    assert s["pass^k"]["grep_sutra"] == 1.0 and s["pass^k"]["grep"] == 0.0
    assert s["invalid"] == {"sutra server status=failed": 1}
    assert s["arms"]["grep"]["func_acc@5"]["n"] == 9
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/benchmark/test_layer2.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `benchmarks/locbench/layer2.py`**

```python
"""Layer 2 — paired grep vs grep+Sutra Claude Code A/B (spec §4, §5)."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmarks.locbench.claude_runner import ARMS, is_valid, render_prompt, run_cell, write_mcp_json
from benchmarks.locbench.dataset import gold_keys_for, load_locbench
from benchmarks.locbench.indexing import artifact_dir_for, ensure_checkout, read_manifest
from benchmarks.locbench.prepare import paths, sutra_exe
from benchmarks.locbench.score import Key, acc_at_k, file_ranking, normalize_gold, recall_at_k
from benchmarks.locbench.stats import cluster_bootstrap_mean, paired_delta, pass_pow_k
from benchmarks.locbench.transcript import RunRecord, parse_stream_json

DELTA_METRICS = ("func_acc@5", "file_acc@1", "total_cost_usd", "tokens_total", "num_turns", "duration_api_ms")


def plan_cells(layer2_ids: list[str], trials: int, seed: int) -> list[dict]:
    cells = []
    for iid in layer2_ids:
        arms = list(ARMS)
        random.Random(f"{seed}:{iid}").shuffle(arms)
        for t in range(1, trials + 1):
            for arm in arms:
                cells.append({"cell_id": f"{iid}|{arm}|t{t}", "instance_id": iid, "arm": arm, "trial": t})
    return cells


def answer_keys(answer: dict | None) -> list[Key]:
    if not answer:
        return []
    out: list[Key] = []
    for f in answer.get("functions", [])[:5]:
        try:
            k = normalize_gold(f"{f['path']}:{f['name']}")
        except (KeyError, TypeError):
            continue
        if k not in out:
            out.append(k)
    return out


def score_run(row: dict, rec: RunRecord) -> dict:
    gold = gold_keys_for(row)
    ranked = answer_keys(rec.answer)
    files = file_ranking(ranked)
    gold_files = {p for p, _ in gold}
    return {"func_acc@5": acc_at_k(gold, ranked, 5), "file_acc@1": acc_at_k(gold_files, files, 1),
            "func_recall@5": recall_at_k(gold, ranked, 5), "file_recall@1": recall_at_k(gold_files, files, 1)}


def adoption_rate(results: list[dict]) -> float:
    runs = [r for r in results if r["arm"] == "grep_sutra" and r["valid"]]
    return sum(1 for r in runs if r["sutra_calls"] >= 1) / len(runs) if runs else 0.0


def summarize_layer2(results: list[dict], cluster_of: dict[str, str]) -> dict:
    valid = [r for r in results if r["valid"]]
    per_issue: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in valid:
        for m in DELTA_METRICS:
            per_issue[r["arm"]][m][r["instance_id"]].append(r[m])
    arms = {}
    for arm in ARMS:
        arms[arm] = {m: cluster_bootstrap_mean([r[m] for r in valid if r["arm"] == arm],
                                               [r["repo"] for r in valid if r["arm"] == arm]) for m in DELTA_METRICS}
    delta = {}
    for m in DELTA_METRICS:
        a = {i: sum(v) / len(v) for i, v in per_issue["grep"][m].items()}
        b = {i: sum(v) / len(v) for i, v in per_issue["grep_sutra"][m].items()}
        delta[m] = paired_delta(a, b, cluster_of)
    sutra_runs = [r for r in valid if r["arm"] == "grep_sutra"]
    tool_counts = Counter(name for r in sutra_runs for name in r.get("tool_names", []) if name.startswith("mcp__sutra__"))
    return {
        "n_runs": len(results), "n_valid": len(valid),
        "invalid": dict(Counter(r["invalid_reason"] for r in results if not r["valid"])),
        "arms": arms, "delta": delta,
        "pass^k": {arm: pass_pow_k(per_issue[arm]["func_acc@5"]) for arm in ARMS},
        "adoption": {"rate": adoption_rate(results),
                     "mean_sutra_calls": (sum(r["sutra_calls"] for r in sutra_runs) / len(sutra_runs)) if sutra_runs else 0.0,
                     "by_tool": dict(tool_counts)},
    }


def _run_one(cell: dict, row: dict, p: dict, out_dir: Path, mcp_json: Path | None, checkout: Path, **kw) -> dict:
    transcript = out_dir / "runs" / f"{cell['cell_id'].replace('|', '__')}.jsonl"
    if transcript.exists():
        rec = parse_stream_json(transcript)
        ok, _ = is_valid(cell["arm"], rec)
        if ok:
            return _row_from(cell, row, rec, True, "", resumed=True)
    rec = run_cell(cell["arm"], render_prompt(row["problem_statement"]), checkout, p["ROOT"] / "answer.schema.json",
                   mcp_json, transcript, **kw)
    ok, why = is_valid(cell["arm"], rec)
    if not ok:  # one retry per spec
        rec = run_cell(cell["arm"], render_prompt(row["problem_statement"]), checkout, p["ROOT"] / "answer.schema.json",
                       mcp_json, transcript, **kw)
        ok, why = is_valid(cell["arm"], rec)
    return _row_from(cell, row, rec, ok, why)


def _row_from(cell: dict, row: dict, rec: RunRecord, valid: bool, why: str, resumed: bool = False) -> dict:
    tokens = rec.usage
    return {**cell, "repo": row["repo"], "category": row["category"], "valid": valid, "invalid_reason": why,
            "resumed": resumed, "sutra_calls": rec.sutra_calls, "tool_names": [n for n, _ in rec.tool_calls],
            "num_turns": rec.num_turns, "total_cost_usd": rec.total_cost_usd, "duration_ms": rec.duration_ms,
            "duration_api_ms": rec.duration_api_ms, **{f"tok_{k}": v for k, v in tokens.items()},
            "tokens_total": sum(tokens.values()), "is_error": rec.is_error, "answer": rec.answer,
            **score_run(row, rec)}


def run_layer2(mode: str, *, trials: int = 3, pilot_issues: int = 5, concurrency: int = 1, seed: int = 20260924,
               max_turns: int = 30, max_budget_usd: float = 0.75) -> dict:
    p = paths()
    sel = json.loads(p["SUBSET"].read_text(encoding="utf-8"))
    rows = {r["instance_id"]: r for r in load_locbench(p["DATA"], sel.get("dataset_sha256"))}
    manifest = read_manifest(p["MANIFEST"])
    ids = [i for i in sel["layer2"] if manifest.get(i, {}).get("status") == "ok"]
    if mode == "pilot":
        ids, trials = ids[:pilot_issues], 1
    out_dir = p["ROOT"] / "layer2" / ("pilot" if mode == "pilot" else "full")
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "pilot":
        (p["ROOT"] / "layer2" / "prereg.sha256").write_text(
            hashlib.sha256((p["ROOT"] / "PREREG.md").read_bytes()).hexdigest(), encoding="utf-8")
    cells = plan_cells(ids, trials, seed)
    mcp_jsons = {i: write_mcp_json(out_dir / "mcp" / f"{i}.json", artifact_dir_for(p["ARTIFACTS"], i), sutra_exe()) for i in ids}
    checkouts = {i: ensure_checkout(rows[i]["repo"], rows[i]["base_commit"], p["REPOS"]) for i in ids}

    def work(cell):
        r = _run_one(cell, rows[cell["instance_id"]], p, out_dir, mcp_jsons[cell["instance_id"]], checkouts[cell["instance_id"]],
                     max_turns=max_turns, max_budget_usd=max_budget_usd)
        print(f"{cell['cell_id']} valid={r['valid']} sutra_calls={r['sutra_calls']} acc5={r['func_acc@5']:.0f} ${r['total_cost_usd']:.3f}")
        return r

    with ThreadPoolExecutor(max_workers=max(1, min(2, concurrency))) as ex:
        results = list(ex.map(work, cells))
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    summary = summarize_layer2(results, {i: rows[i]["repo"] for i in ids})
    if mode == "pilot":
        decision = {"adoption": summary["adoption"]["rate"], "stop": summary["adoption"]["rate"] < 0.5, "n_issues": len(ids)}
        (out_dir / "pilot_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        summary["pilot_decision"] = decision
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1)
    a = ap.parse_args()
    s = run_layer2(a.mode, trials=a.trials, concurrency=a.concurrency)
    print(json.dumps({k: s[k] for k in ("n_valid", "invalid", "adoption", "delta")}, indent=2, default=str))
    if s.get("pilot_decision", {}).get("stop"):
        raise SystemExit("PILOT STOP: adoption < 0.50 — fix tool descriptions before spending the budget")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/benchmark/test_layer2.py -q`
Expected: 3 passed. If the arm-order assertion fails for seed 7, print `first_arm` — if all six issues truly start with the same arm under this hashing, change the test's seed to one where they don't and keep the assertion (it guards that randomization exists at all).

- [ ] **Step 6: Commit**

```bash
git add benchmarks/locbench/layer2.py benchmarks/locbench/PREREG.md tests/benchmark/test_layer2.py
git commit -m "bench(locbench): layer 2 runner with pilot decision rule, resumable cells, PREREG"
```

- [ ] **Step 7: Index the Layer 2 issues and run the PILOT (stage-3 gate)**

```bash
python -m benchmarks.locbench.prepare index --layer2-only      # no-op for ids already indexed in Task 5
python -m benchmarks.locbench.layer2 --mode pilot              # 5 issues × 2 arms × 1 trial ≈ $3–5
python -m benchmarks.locbench.layer2 --mode pilot              # resume check: must finish in seconds, every row resumed=true
python - <<'EOF'
import json; rows=[json.loads(l) for l in open("benchmarks/locbench/layer2/pilot/results.jsonl", encoding="utf-8")]
assert all(r["resumed"] for r in rows if r["valid"]), "resume path re-ran a valid cell"
print("resume OK", len(rows), "rows")
EOF
git add benchmarks/locbench/layer2/prereg.sha256 benchmarks/locbench/layer2/pilot/results.jsonl benchmarks/locbench/layer2/pilot/summary.json benchmarks/locbench/layer2/pilot/pilot_decision.json benchmarks/locbench/layer2/pilot/runs
git commit -m "bench(locbench): layer 2 adoption pilot"
```
**Report back to the designer** with `pilot_decision.json`, `adoption.by_tool`, `invalid`, and per-run cost. **If `stop` is true, do not run `--mode full`** — the next work item is product-side (Sutra tool descriptions), decided by the designer.

---

### Task 10: `report.py` — PREREG hash gate + REPORT.md + README snippet

**Files:**
- Create: `benchmarks/locbench/report.py`
- Test: `tests/benchmark/test_report.py`

**Interfaces:**
- Consumes: `layer1/summary.json` + `layer1/results.jsonl` (Task 5), `layer2/full/summary.json` (Task 9), `cluster_bootstrap_mean` (Task 8), `paths()` (Task 4).
- Produces:
  - `check_prereg(root: Path) -> None` — raises `RuntimeError` if `layer2/prereg.sha256` is missing or ≠ sha256(`PREREG.md`)
  - `layer1_cis(results_path: Path) -> dict` — for each `query|retriever` cell and metric, `cluster_bootstrap_mean` over issues clustered by repo
  - `render_report(l1_summary: dict, l1_cis: dict, l2_summary: dict | None, meta: dict) -> str` — Markdown with sections, in this order: `# Sutra on LocBench`, `## Headline`, `## Guard metric: adoption`, `## Layer 1 — index-only localization`, `## Layer 2 — Claude Code paired A/B`, `## Method`, `## Caveats`, `## README snippet`
  - `main()` — writes `benchmarks/locbench/REPORT.md`

- [ ] **Step 1: Write the failing tests**

`tests/benchmark/test_report.py`:
```python
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.locbench.report import check_prereg, layer1_cis, render_report


def test_check_prereg_requires_matching_hash(tmp_path):
    (tmp_path / "PREREG.md").write_text("v1", encoding="utf-8")
    (tmp_path / "layer2").mkdir()
    with pytest.raises(RuntimeError):
        check_prereg(tmp_path)  # missing hash file
    (tmp_path / "layer2" / "prereg.sha256").write_text(hashlib.sha256(b"v1").hexdigest(), encoding="utf-8")
    check_prereg(tmp_path)
    (tmp_path / "PREREG.md").write_text("v2 edited after the pilot", encoding="utf-8")
    with pytest.raises(RuntimeError):
        check_prereg(tmp_path)


def test_layer1_cis_from_results(tmp_path):
    rows = []
    for i, repo in (("a", "r"), ("b", "r"), ("c", "s")):
        rows.append({"instance_id": i, "repo": repo, "query": "full", "retriever": "hybrid", "func_acc@5": 1.0 if i != "c" else 0.0, "file_acc@1": 1.0})
    p = tmp_path / "results.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    cis = layer1_cis(p)
    cell = cis["full|hybrid"]
    assert abs(cell["func_acc@5"]["mean"] - 2 / 3) < 1e-9 and cell["func_acc@5"]["n_clusters"] == 2
    assert cell["file_acc@1"]["ci_lo"] == 1.0 == cell["file_acc@1"]["ci_hi"]


def test_render_report_has_every_section_and_flags_directional_delta():
    l1 = {"n_issues": 3, "n_missing_gold_issues": 1, "cells": {"full|hybrid": {"n": 3, "func_acc@5": 0.67, "file_acc@1": 1.0}}, "by_category": {}, "by_repo": {}}
    cis = {"full|hybrid": {"func_acc@5": {"mean": 0.67, "ci_lo": 0.3, "ci_hi": 1.0, "n": 3, "n_clusters": 2},
                           "file_acc@1": {"mean": 1.0, "ci_lo": 1.0, "ci_hi": 1.0, "n": 3, "n_clusters": 2}}}
    l2 = {"n_runs": 180, "n_valid": 178, "invalid": {"sutra server status=failed": 2},
          "arms": {"grep": {"func_acc@5": {"mean": 0.5, "ci_lo": 0.4, "ci_hi": 0.6}, "total_cost_usd": {"mean": 0.2, "ci_lo": 0.1, "ci_hi": 0.3}},
                   "grep_sutra": {"func_acc@5": {"mean": 0.55, "ci_lo": 0.45, "ci_hi": 0.65}, "total_cost_usd": {"mean": 0.25, "ci_lo": 0.15, "ci_hi": 0.35}}},
          "delta": {"func_acc@5": {"mean": 0.05, "ci_lo": -0.05, "ci_hi": 0.15, "n_pairs": 30},
                    "total_cost_usd": {"mean": 0.05, "ci_lo": 0.01, "ci_hi": 0.09, "n_pairs": 30}},
          "pass^k": {"grep": 0.3, "grep_sutra": 0.35},
          "adoption": {"rate": 0.6, "mean_sutra_calls": 1.4, "by_tool": {"mcp__sutra__sutra_search": 40}}}
    md = render_report(l1, cis, l2, {"sutra_git_sha": "abc", "date": "2026-09-24", "embedder": "BAAI/bge-base-en-v1.5", "model": "claude-sonnet-5"})
    for h in ("# Sutra on LocBench", "## Headline", "## Guard metric: adoption", "## Layer 1", "## Layer 2", "## Method", "## Caveats", "## README snippet"):
        assert h in md
    assert "directional" in md and "[−5%, +15%]" in md  # U+2212 minus, percent-formatted Δ
    assert "adoption 60%" in md.replace("**", "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/benchmark/test_report.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `benchmarks/locbench/report.py`**

```python
"""REPORT.md generator — fixed template, PREREG hash gate, repo-clustered CIs for Layer 1."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path

from benchmarks.locbench.stats import cluster_bootstrap_mean

PUBLISHED = [  # label, file Acc@1, source — "published; different setup"
    ("LocAgent (Claude-3.5, paper)", "78%", "arXiv 2503.09089"),
    ("FastCode (paper)", "86%", "arXiv 2603.01012"),
]
L1_METRICS = ("file_acc@1", "file_acc@3", "file_acc@5", "func_acc@5", "func_acc@10", "func_recall@50", "func_mrr")


def check_prereg(root: Path) -> None:
    h = root / "layer2" / "prereg.sha256"
    if not h.exists():
        raise RuntimeError("layer2/prereg.sha256 missing — run the pilot before reporting")
    actual = hashlib.sha256((root / "PREREG.md").read_bytes()).hexdigest()
    if actual != h.read_text(encoding="utf-8").strip():
        raise RuntimeError("PREREG.md changed after the pilot — refusing to report")


def layer1_cis(results_path: Path) -> dict:
    cells: dict[str, list[dict]] = defaultdict(list)
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            cells[f"{r['query']}|{r['retriever']}"].append(r)
    out = {}
    for cell, rows in cells.items():
        metrics = [m for m in L1_METRICS if m in rows[0]]
        out[cell] = {m: cluster_bootstrap_mean([r[m] for r in rows], [r["repo"] for r in rows]) for m in metrics}
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _ci(d: dict) -> str:
    """Percent point estimate with its 95% CI, e.g. `67% [30%, 100%]`."""
    return f"{_pct(d['mean'])} [{_pct(d['ci_lo'])}, {_pct(d['ci_hi'])}]"


def _signed(v: float, pct: bool) -> str:
    s = f"{100 * v:+.0f}%" if pct else f"{v:+.3f}"
    return s.replace("-", "−")


def render_report(l1: dict, cis: dict, l2: dict | None, meta: dict) -> str:
    L = [f"# Sutra on LocBench", "",
         f"Generated {meta['date']} · sutra `{meta['sutra_git_sha'][:10]}` · embedder `{meta['embedder']}` · agent `{meta['model']}`", ""]
    best = cis.get("full|hybrid") or next(iter(cis.values()))
    L += ["## Headline", "",
          "| Metric | Sutra (hybrid, full issue text) | Published (different setup) |", "|---|---|---|",
          f"| file Acc@1 | **{_ci(best['file_acc@1'])}** | " + "; ".join(f"{n} {v}" for n, v, _ in PUBLISHED) + " |",
          f"| function Acc@5 | **{_ci(best['func_acc@5'])}** | — |", ""]
    if l2:
        d = l2["delta"]["func_acc@5"]
        L += [f"Claude Code paired A/B (n={d['n_pairs']} issues): Δ function Acc@5 = {_signed(d['mean'], True)} "
              f"[{_signed(d['ci_lo'], True)}, {_signed(d['ci_hi'], True)}] — **directional** at this n; "
              f"Δ cost/run {_signed(l2['delta']['total_cost_usd']['mean'], False)} USD.", ""]
    L += ["## Guard metric: adoption", ""]
    if l2:
        a = l2["adoption"]
        L += [f"With Sutra merely available (prompt never mentions it): **adoption {_pct(a['rate'])}** of valid runs, "
              f"{a['mean_sutra_calls']:.1f} calls/run. By tool: " + ", ".join(f"`{k}` {v}" for k, v in sorted(a["by_tool"].items())) + ".", ""]
    else:
        L += ["Layer 2 not run.", ""]
    L += ["## Layer 1 — index-only localization", "",
          f"{l1['n_issues']} issues; {l1['n_missing_gold_issues']} had ≥1 gold function absent from the index (counted as misses).", "",
          "| query × retriever | " + " | ".join(L1_METRICS) + " |", "|---|" + "---|" * len(L1_METRICS)]
    for cell in sorted(cis):
        L.append(f"| {cell} | " + " | ".join(_ci(cis[cell][m]) if m in cis[cell] else "—" for m in L1_METRICS) + " |")
    L += ["", "## Layer 2 — Claude Code paired A/B", ""]
    if l2:
        L += [f"{l2['n_valid']}/{l2['n_runs']} valid runs; excluded: " + (", ".join(f"{k} ×{v}" for k, v in l2["invalid"].items()) or "none") + ".", "",
              "| metric | grep | grep+Sutra | paired Δ (95% CI, repo-clustered) |", "|---|---|---|---|"]
        for m, pct in (("func_acc@5", True), ("file_acc@1", True), ("total_cost_usd", False), ("tokens_total", False), ("num_turns", False), ("duration_api_ms", False)):
            if m in l2["delta"]:
                g, s, d = l2["arms"]["grep"].get(m), l2["arms"]["grep_sutra"].get(m), l2["delta"][m]
                fmt = (lambda v: _pct(v)) if pct else (lambda v: f"{v:.3g}")
                L.append(f"| {m} | {fmt(g['mean']) if g else '—'} | {fmt(s['mean']) if s else '—'} | "
                         f"{_signed(d['mean'], pct)} [{_signed(d['ci_lo'], pct)}, {_signed(d['ci_hi'], pct)}] |")
        L += ["", f"pass^3 (all 3 trials correct): grep {_pct(l2['pass^k']['grep'])}, grep+Sutra {_pct(l2['pass^k']['grep_sutra'])}.", ""]
    else:
        L += ["Not run.", ""]
    L += ["## Method", "",
          "- Dataset: LocBench V1 (`czlll/Loc-Bench_V1`), repos with ≥8 issues; one Sutra index per issue `base_commit`; gold = `edit_functions`.",
          "- Acc@k = 1 only if every gold location is in the top-k (LocBench definition). CIs: 95% cluster bootstrap over repos, 10 000 draws.",
          "- Layer 2: `claude -p` headless, `claude-sonnet-5`, arms `grep` (Read/Grep/Glob) vs `grep+Sutra` (+`mcp__sutra__*`), identical prompt that never mentions Sutra, 3 trials/cell, arm order randomized per issue, pre-registered in `PREREG.md`.", "",
          "## Caveats", "",
          "- Python-only (LocBench); 20 repos; Layer 2 n=30 → Δ is directional, not a significance claim.",
          "- Published numbers are from different systems and setups; only the metric definition is shared.",
          "- Adoption is a property of Sutra's tool descriptions as shipped; a low rate makes Layer 2 Δ uninformative about index quality.", "",
          "## README snippet", "", "```",
          f"LocBench (Python, {l1['n_issues']} issues): file Acc@1 {_ci(best['file_acc@1'])}, function Acc@5 {_ci(best['func_acc@5'])} (index-only).", "```", ""]
    return "\n".join(L)


def main() -> None:
    from benchmarks.locbench.prepare import paths
    p = paths()
    root = p["ROOT"]
    l1 = json.loads((root / "layer1" / "summary.json").read_text(encoding="utf-8"))
    cis = layer1_cis(root / "layer1" / "results.jsonl")
    l2 = None
    full = root / "layer2" / "full" / "summary.json"
    if full.exists():
        check_prereg(root)
        l2 = json.loads(full.read_text(encoding="utf-8"))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root).stdout.strip()
    md = render_report(l1, cis, l2, {"date": date.today().isoformat(), "sutra_git_sha": sha,
                                     "embedder": "BAAI/bge-base-en-v1.5", "model": "claude-sonnet-5"})
    (root / "REPORT.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/benchmark/test_report.py -q`
Expected: 3 passed. The `[−5%, +15%]` assertion needs the U+2212 minus that `_signed` emits — keep it; it is what the report shows.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/locbench/report.py tests/benchmark/test_report.py
git commit -m "bench(locbench): REPORT.md generator with PREREG gate and clustered CIs"
```

- [ ] **Step 6: Full Layer 2 run + report (stage-4 gate; only if the pilot said go)**

```bash
python -m benchmarks.locbench.layer2 --mode full --trials 3 --concurrency 1     # ≈ 180 runs, $60–85, several hours
python -m benchmarks.locbench.report
git add benchmarks/locbench/layer2/full/results.jsonl benchmarks/locbench/layer2/full/summary.json benchmarks/locbench/layer2/full/runs benchmarks/locbench/REPORT.md
git commit -m "bench(locbench): full layer 2 run and REPORT.md"
```
**Report back to the designer** with `REPORT.md` and the raw `summary.json` for both layers.

---

### Task 11: Whole-suite check and README pointer

**Files:**
- Modify: `README.md` (add one bullet under "Testing" or "Roadmap" pointing at `benchmarks/locbench/REPORT.md`)

- [ ] **Step 1: Run the entire test suite**

Run: `python -m pytest -q -p no:warnings`
Expected: all pass except the pre-existing Windows LSP failure `tests/test_backend_bridge_integration.py::test_crashing_lsp_aborts_index_and_publishes_nothing` (`select()` on a pipe — known, out of scope). Anything else failing is a regression from this plan — fix before committing.

- [ ] **Step 2: Add the README pointer**

Under the `## Testing` section of `README.md` add:
```markdown
- **Launch benchmark** (LocBench localization + Claude Code paired A/B): see `benchmarks/locbench/REPORT.md`; method in `docs/superpowers/specs/2026-09-24-locbench-benchmark-design.md`.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: point README at the LocBench benchmark report"
```
