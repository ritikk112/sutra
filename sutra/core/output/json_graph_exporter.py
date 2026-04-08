from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from sutra.core.extractor.base import (
    ClassSymbol,
    File,
    FunctionSymbol,
    IndexResult,
    MethodSymbol,
    ModuleSymbol,
    Relationship,
    Symbol,
    VariableSymbol,
)

SUTRA_VERSION = "0.1.0"
SCHEMA_VERSION = "1"
DEFAULT_EMBEDDING_DIMS = 384

# Only these symbol types will receive embeddings in the real pipeline.
# Modules and variables don't have meaningful body chunks to embed.
_EMBEDDABLE = (FunctionSymbol, ClassSymbol, MethodSymbol)


class JsonGraphExporter:
    """
    Writes an IndexResult to three files in output_dir:

      graph.json           — full versioned schema (symbols, relationships, files, metadata)
      embeddings.npy       — float32 array of shape (N, dims), one row per embeddable symbol
      embeddings_index.json — list of monikers, index N → moniker for row N in .npy

    In this phase no real embedder is present.  Each embeddable symbol receives a
    deterministic fixture vector seeded from sha256(moniker), so:
      - Two runs on the same IndexResult produce identical bytes.
      - Identical monikers always produce identical vectors.
      - Tests can snapshot the .npy content.

    Contract (three sources of truth that must agree):
      Row N in embeddings.npy
        == embeddings_index.json[N]  (the moniker)
        == graph.json symbol whose embedding_id == N
    All three are built by _build_embeddings() so they cannot drift.
    """

    def __init__(self, embedding_dims: int = DEFAULT_EMBEDDING_DIMS) -> None:
        self.embedding_dims = embedding_dims

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, result: IndexResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Stable ordering for deterministic output
        sorted_symbols = sorted(result.symbols, key=lambda s: s.id)
        sorted_rels = sorted(
            result.relationships,
            key=lambda r: (r.source_id, r.target_name or "", r.target_id or ""),
        )
        sorted_files = sorted(result.files, key=lambda f: f.path)

        # Build all three embedding outputs in one place — they cannot drift
        embeddable = [s for s in sorted_symbols if isinstance(s, _EMBEDDABLE)]
        array, index = self._build_embeddings(embeddable)
        moniker_to_row: dict[str, int] = {m: i for i, m in enumerate(index)}

        # Write embeddings.npy
        np.save(output_dir / "embeddings.npy", array)

        # Write embeddings_index.json
        (output_dir / "embeddings_index.json").write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )

        # Write graph.json
        graph = {
            "sutra_version": SUTRA_VERSION,
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repository": {
                "name": result.repository.name,
                "url": result.repository.url,
                "commit_sha": result.commit_hash,
                "languages": result.languages,
            },
            "symbols": [
                self._symbol_to_dict(s, moniker_to_row.get(s.id))
                for s in sorted_symbols
            ],
            "relationships": [
                self._relationship_to_dict(r) for r in sorted_rels
            ],
            "files": [self._file_to_dict(f) for f in sorted_files],
            "embeddings": {
                "file": "embeddings.npy",
                "index_file": "embeddings_index.json",
                "dims": self.embedding_dims,
                "count": len(index),
                "dtype": "float32",
            },
            "failed_files": [
                {"path": path, "error": error}
                for path, error in result.failed_files
            ],
        }
        (output_dir / "graph.json").write_text(
            json.dumps(graph, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _build_embeddings(
        self, embeddable: list[Symbol]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Return (array, index) where:
          - array  is float32, shape (len(embeddable), self.embedding_dims)
          - index  is a list of monikers in the same order as array rows

        Each row is deterministic: seeded from sha256(moniker) % 2^32.
        """
        rows: list[np.ndarray] = []
        index: list[str] = []
        for sym in embeddable:
            seed = int(hashlib.sha256(sym.id.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            rows.append(rng.random(self.embedding_dims).astype(np.float32))
            index.append(sym.id)
        if rows:
            array = np.stack(rows).astype(np.float32)
        else:
            array = np.empty((0, self.embedding_dims), dtype=np.float32)
        return array, index

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _symbol_to_dict(self, sym: Symbol, embedding_id: Optional[int]) -> dict:
        loc = sym.location
        base: dict = {
            "id": sym.id,
            "name": sym.name,
            "qualified_name": sym.qualified_name,
            "file_path": sym.file_path,
            "location": {
                "line_start": loc.line_start,
                "line_end": loc.line_end,
                "byte_start": loc.byte_start,
                "byte_end": loc.byte_end,
                "column_start": loc.column_start,
                "column_end": loc.column_end,
            },
            "body_hash": sym.body_hash,
            "language": sym.language,
            "visibility": sym.visibility.value,
            "is_exported": sym.is_exported,
            "embedding_id": embedding_id,
        }

        # MethodSymbol must be checked before FunctionSymbol (it's a subtype)
        if isinstance(sym, MethodSymbol):
            base["kind"] = "method"
            base.update(self._function_fields(sym))
            base["enclosing_class_id"] = sym.enclosing_class_id
            base["is_static"] = sym.is_static
            base["is_constructor"] = sym.is_constructor
            if sym.receiver_kind is not None:
                base["receiver_kind"] = sym.receiver_kind
        elif isinstance(sym, FunctionSymbol):
            base["kind"] = "function"
            base.update(self._function_fields(sym))
        elif isinstance(sym, ClassSymbol):
            base["kind"] = "class"
            base["base_classes"] = sym.base_classes
            base["docstring"] = sym.docstring
            base["decorators"] = sym.decorators
            base["is_abstract"] = sym.is_abstract
        elif isinstance(sym, VariableSymbol):
            base["kind"] = "variable"
            base["type_annotation"] = sym.type_annotation
            base["is_constant"] = sym.is_constant
        elif isinstance(sym, ModuleSymbol):
            base["kind"] = "module"
            base["docstring"] = sym.docstring

        return base

    def _function_fields(self, sym: FunctionSymbol) -> dict:
        return {
            "signature": sym.signature,
            "parameters": [
                {
                    "name": p.name,
                    "type_annotation": p.type_annotation,
                    "default_value": p.default_value,
                    "is_variadic": p.is_variadic,
                    "is_keyword_variadic": p.is_keyword_variadic,
                }
                for p in sym.parameters
            ],
            "return_type": sym.return_type,
            "docstring": sym.docstring,
            "decorators": sym.decorators,
            "is_async": sym.is_async,
            "complexity": sym.complexity,
        }

    def _relationship_to_dict(self, rel: Relationship) -> dict:
        loc = rel.location
        return {
            "source_id": rel.source_id,
            "kind": rel.kind.value,
            "is_resolved": rel.is_resolved,
            "target_id": rel.target_id,
            "target_name": rel.target_name,
            "location": {
                "line_start": loc.line_start,
                "line_end": loc.line_end,
                "byte_start": loc.byte_start,
                "byte_end": loc.byte_end,
                "column_start": loc.column_start,
                "column_end": loc.column_end,
            } if loc else None,
            "metadata": rel.metadata,
        }

    def _file_to_dict(self, f: File) -> dict:
        return {
            "path": f.path,
            "language": f.language,
            "size_bytes": f.size_bytes,
            "hash": f.hash,
        }
