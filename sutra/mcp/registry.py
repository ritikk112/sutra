from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sutra.core.artifact import ArtifactError, ArtifactLoader, ArtifactSnapshot
from sutra.core.embedder.base import Embedder
from sutra.core.graph.traversal import RustworkxTraversal
from sutra.core.retrieval.pipeline import RetrievalPipeline
from sutra.core.retrieval.query_analyzer import QueryAnalyzer


@dataclass(frozen=True, slots=True)
class ServingUnit:
    """Everything needed to serve one repo: immutable, swapped as a whole."""

    repo_name: str
    snapshot: ArtifactSnapshot
    pipeline: RetrievalPipeline = field(repr=False)
    traversal: RustworkxTraversal = field(repr=False)
    artifact_dir: Optional[Path] = None


class EmbedderCache:
    """
    One query-time embedder per model_id, shared across repos.

    Loading sentence-transformers weights costs seconds — repos indexed
    with the same model share one instance.  Fixture artifacts resolve to
    FixtureEmbedder (tests; no ML deps needed).
    """

    def __init__(self) -> None:
        self._cache: dict[str, Embedder] = {}
        self._lock = threading.Lock()

    def get(self, model_id: Optional[str], dims: Optional[int] = None) -> Embedder:
        key = model_id or "fixture-384"
        with self._lock:
            if key not in self._cache:
                self._cache[key] = self._build(key, dims)
            return self._cache[key]

    @staticmethod
    def _build(model_id: str, dims: Optional[int] = None) -> Embedder:
        if model_id.startswith("fixture-"):
            from sutra.core.embedder.fixture import FixtureEmbedder
            return FixtureEmbedder(dims=int(model_id.split("-")[1]))
        if model_id.startswith("sentence-transformers/"):
            from sutra.core.embedder.local import LocalEmbedder
            # The artifact's recorded width is authoritative — LocalEmbedder
            # defaults to 384, which rejects every non-MiniLM model at boot.
            kwargs = {"dimensions": dims} if dims else {}
            return LocalEmbedder(model_name=model_id.split("/", 1)[1], **kwargs)
        if model_id.startswith("openai/"):
            import os
            from sutra.core.embedder.openai import OpenAIEmbedder
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError(
                    f"Artifact was embedded with {model_id!r} — set "
                    f"OPENAI_API_KEY so queries can use the same model."
                )
            return OpenAIEmbedder(api_key=api_key, model=model_id.split("/", 1)[1])
        raise ValueError(f"Unknown embedding model_id in artifact: {model_id!r}")


class SnapshotRegistry:
    """
    repo_name → ServingUnit with ref-counted atomic swap semantics.

    Reads are lock-free: `get()` is a single dict lookup under the GIL,
    and the returned ServingUnit is immutable — an in-flight query keeps
    serving from the unit it grabbed even if a swap lands mid-query
    (Python refcounting IS the ref-count; the old unit is freed when the
    last query drops it).  Only the swap itself takes the mutex.
    """

    def __init__(self) -> None:
        self._units: dict[str, ServingUnit] = {}
        self._lock = threading.Lock()

    def get(self, repo_name: str) -> Optional[ServingUnit]:
        return self._units.get(repo_name)          # lock-free read

    def repos(self) -> list[str]:
        return sorted(self._units)

    def swap(self, unit: ServingUnit) -> None:
        with self._lock:
            self._units[unit.repo_name] = unit

    def remove(self, repo_name: str) -> None:
        with self._lock:
            self._units.pop(repo_name, None)


def build_serving_unit(
    artifact_dir: Path,
    embedders: EmbedderCache,
    analyzer_cache: Optional[dict[str, QueryAnalyzer]] = None,
) -> ServingUnit:
    """
    Load + validate one artifact directory into a ready-to-serve unit.
    Raises ArtifactError / ValueError on torn or incompatible artifacts —
    callers decide whether that's fatal (boot) or skippable (hot reload).
    """
    snapshot = ArtifactLoader().load(artifact_dir)
    embedder = embedders.get(snapshot.embedding_model_id, snapshot.embedding_dims)

    analyzer = None
    if analyzer_cache is not None:
        key = snapshot.embedding_model_id or "fixture-384"
        analyzer = analyzer_cache.get(key)
        if analyzer is None:
            analyzer = QueryAnalyzer(embedder=embedder)
            analyzer_cache[key] = analyzer

    pipeline = RetrievalPipeline(snapshot, embedder, analyzer=analyzer)
    traversal = RustworkxTraversal(snapshot)
    return ServingUnit(
        repo_name=snapshot.repo_name,
        snapshot=snapshot,
        pipeline=pipeline,
        traversal=traversal,
        artifact_dir=artifact_dir,
    )


def scan_artifacts_root(
    root: Path,
    registry: SnapshotRegistry,
    embedders: EmbedderCache,
    analyzer_cache: Optional[dict[str, QueryAnalyzer]] = None,
    strict: bool = False,
) -> list[str]:
    """
    Load every artifact subdirectory under `root` into the registry.

    A directory counts as an artifact if it contains graph.json.  Broken
    artifacts are skipped with a stderr warning (strict=False, the boot
    default — one bad repo must not take the server down) or raised
    (strict=True, used by tests).
    Returns the repo names loaded this scan.
    """
    loaded: list[str] = []
    for child in sorted(root.iterdir()):
        if not (child / "graph.json").exists():
            continue
        try:
            unit = build_serving_unit(child, embedders, analyzer_cache)
        except (ArtifactError, ValueError) as exc:
            if strict:
                raise
            print(f"[sutra-mcp] skipping {child.name}: {exc}", file=sys.stderr)
            continue
        registry.swap(unit)
        loaded.append(unit.repo_name)
    return loaded
