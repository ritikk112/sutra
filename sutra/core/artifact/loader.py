from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Schema versions this loader knows how to read.  The exporter stamps
# graph.json with schema_version; loading an unknown version fails loudly
# at boot rather than silently mis-loading.
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1"})


class ArtifactError(Exception):
    """Raised when an artifact directory is missing files, torn, or incompatible."""


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    """
    A fully-loaded, validated, immutable view of one artifact directory.

    Everything the in-memory retrieval stack needs is here:
    symbol dicts keyed by moniker, the relationship list, the embedding
    matrix, and the row ↔ moniker mapping.  Channels and the MCP-side
    traversal build their own derived indices from this object.
    """

    path: Path
    repo_name: str
    repo_url: str
    commit_sha: str
    schema_version: str
    symbols: dict[str, dict]            # moniker → symbol dict (as in graph.json)
    relationships: list[dict] = field(repr=False)
    vectors: np.ndarray = field(repr=False)   # float32 (N, dims) — raw, NOT normalized
    moniker_order: list[str] = field(repr=False)  # row N ↔ moniker_order[N]
    row_of: dict[str, int] = field(repr=False)    # moniker → row index
    embedding_model_id: Optional[str]
    embedding_dims: int

    @property
    def embedding_count(self) -> int:
        return int(self.vectors.shape[0])


class ArtifactLoader:
    """
    Pure loader: artifact directory path → ArtifactSnapshot.

    Boot-time integrity checks (fail loudly, never mis-load):
      1. graph.json / embeddings.npy / embeddings_index.json all present.
      2. schema_version is one we support.
      3. embeddings.npy row count == len(embeddings_index.json) — primary
         torn-artifact check (the two files commit together as a unit).
      4. vectors.shape[1] == graph.json embeddings.dims.
      5. Every symbol with embedding_id == N must be embeddings_index[N]
         and vice versa — the three sources of truth must agree.

    The `.ready` sentinel is NOT required here: the loader can inspect any
    directory.  The P19 ArtifactWatcher is what gates hot-reload on the
    sentinel; integrity is what gates loading.
    """

    def load(self, artifact_dir: Path | str) -> ArtifactSnapshot:
        artifact_dir = Path(artifact_dir)

        graph_path = artifact_dir / "graph.json"
        npy_path = artifact_dir / "embeddings.npy"
        index_path = artifact_dir / "embeddings_index.json"

        for p in (graph_path, npy_path, index_path):
            if not p.exists():
                raise ArtifactError(
                    f"Artifact at {artifact_dir} is missing {p.name} — "
                    f"not a complete Sutra artifact directory."
                )

        try:
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"graph.json is not valid JSON: {exc}") from exc

        schema_version = str(graph.get("schema_version", ""))
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ArtifactError(
                f"Unsupported schema_version {schema_version!r} in {graph_path} "
                f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}). "
                f"Re-index with a matching Sutra version or upgrade the consumer."
            )

        vectors = np.load(npy_path)
        if vectors.ndim != 2:
            raise ArtifactError(
                f"embeddings.npy must be 2-D (N, dims); got shape {vectors.shape}."
            )
        vectors = vectors.astype(np.float32, copy=False)

        moniker_order: list[str] = json.loads(index_path.read_text(encoding="utf-8"))
        if vectors.shape[0] != len(moniker_order):
            raise ArtifactError(
                f"Torn artifact: embeddings.npy has {vectors.shape[0]} rows but "
                f"embeddings_index.json lists {len(moniker_order)} monikers. "
                f"The two files must be written (and copied) together."
            )

        emb_meta = graph.get("embeddings", {}) or {}
        declared_dims = int(emb_meta.get("dims", vectors.shape[1] if vectors.size else 0))
        if vectors.shape[0] > 0 and vectors.shape[1] != declared_dims:
            raise ArtifactError(
                f"Torn artifact: graph.json declares dims={declared_dims} but "
                f"embeddings.npy rows have {vectors.shape[1]} dimensions."
            )

        symbols: dict[str, dict] = {}
        for sym in graph.get("symbols", []):
            symbols[sym["id"]] = sym

        # Cross-check embedding_id ↔ index-file agreement (three sources of truth).
        row_of: dict[str, int] = {m: i for i, m in enumerate(moniker_order)}
        for moniker, sym in symbols.items():
            eid = sym.get("embedding_id")
            if eid is None:
                continue
            if eid >= len(moniker_order) or moniker_order[eid] != moniker:
                actual = (
                    repr(moniker_order[eid]) if eid < len(moniker_order)
                    else "out of range"
                )
                raise ArtifactError(
                    f"Torn artifact: symbol {moniker!r} claims embedding_id={eid} "
                    f"but embeddings_index.json row {eid} is {actual}."
                )
        for moniker in moniker_order:
            if moniker not in symbols:
                raise ArtifactError(
                    f"Torn artifact: embeddings_index.json lists {moniker!r} "
                    f"but graph.json has no such symbol."
                )

        repo = graph.get("repository", {}) or {}

        return ArtifactSnapshot(
            path=artifact_dir,
            repo_name=repo.get("name", ""),
            repo_url=repo.get("url", ""),
            commit_sha=repo.get("commit_sha", ""),
            schema_version=schema_version,
            symbols=symbols,
            relationships=list(graph.get("relationships", [])),
            vectors=vectors,
            moniker_order=moniker_order,
            row_of=row_of,
            embedding_model_id=emb_meta.get("model_id"),
            embedding_dims=declared_dims,
        )
