"""
Shared dependency construction for Sutra pipelines.

Both full_index.py and incremental_update.py need the same set of objects:
adapters, embedder, graph writer/reader/state, pgvector store, etc.  This
module builds them from a config path and environment so the two CLIs don't
duplicate the logic.

Usage
-----
    deps = build_dependencies(config_path=Path("config/sutra.yaml"), pg_url=pg_url)
    # deps.adapters, deps.embedder, deps.graph_writer, deps.reader,
    # deps.state_store, deps.pgvector_store, ...

The caller is responsible for calling setup() on the writers and close() when
done (or using them as context managers).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from sutra.core.embedder.factory import from_config
from sutra.core.extractor.adapters.go import GoAdapter
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.extractor.adapters.typescript import TypeScriptAdapter
from sutra.core.gitignore_filter import GitignoreFilter
from sutra.core.graph.base import GraphWriter, IncrementalReader
from sutra.core.graph.pgvector_store import PGVectorStore
from sutra.core.graph.sql_reader import SqlIncrementalReader
from sutra.core.graph.sql_state import SqlIndexStateStore
from sutra.core.graph.sql_writer import SqlGraphWriter
from sutra.core.output.json_graph_exporter import JsonGraphExporter


@dataclass
class PipelineDeps:
    """All constructed pipeline dependencies."""
    adapters: dict[str, Any]
    embedder: Any            # Embedder
    exporter: JsonGraphExporter
    graph_writer: Optional[GraphWriter]
    reader: Optional[IncrementalReader]
    state_store: Optional[SqlIndexStateStore]
    pgvector_store: Optional[PGVectorStore]

    def close(self) -> None:
        """Close all DB connections.  Idempotent."""
        if self.graph_writer:
            self.graph_writer.close()
        if self.reader:
            self.reader.close()
        if self.state_store:
            self.state_store.close()
        if self.pgvector_store:
            self.pgvector_store.close()


def build_dependencies(
    config_path: Optional[Path] = None,
    pg_url: Optional[str] = None,
    dims: Optional[int] = None,
    recreate_embeddings: bool = False,
) -> PipelineDeps:
    """
    Construct all pipeline dependencies from config.

    Parameters
    ----------
    config_path : Path | None
        Path to sutra.yaml.  None → FixtureEmbedder (CI/test mode).
    pg_url : str | None
        PostgreSQL connection string.  None → no DB sinks (JSON-only mode).
    dims : int | None
        Embedding dimensionality for pgvector.  None → derived from embedder.
    recreate_embeddings : bool
        When True, DROP and recreate the embeddings table at setup time.
        WARNING: destroys all existing embeddings.  Use when switching
        embedder providers that change vector dimensions.
    """
    adapters: dict[str, Any] = {
        "python": PythonAdapter(),
        "typescript": TypeScriptAdapter(tsx=False),
        "go": GoAdapter(),
    }

    embedder = from_config(config_path)
    exporter = JsonGraphExporter()

    graph_writer: Optional[GraphWriter] = None
    reader: Optional[IncrementalReader] = None
    state_store: Optional[SqlIndexStateStore] = None
    pgvector_store: Optional[PGVectorStore] = None

    if pg_url:
        graph_writer = SqlGraphWriter(pg_url)
        graph_writer.setup()

        reader = SqlIncrementalReader(pg_url)
        state_store = SqlIndexStateStore(pg_url)

        vec_dims = dims if dims is not None else embedder.dimensions
        pgvector_store = PGVectorStore(pg_url, dims=vec_dims)
        pgvector_store.setup(recreate=recreate_embeddings)

    return PipelineDeps(
        adapters=adapters,
        embedder=embedder,
        exporter=exporter,
        graph_writer=graph_writer,
        reader=reader,
        state_store=state_store,
        pgvector_store=pgvector_store,
    )
