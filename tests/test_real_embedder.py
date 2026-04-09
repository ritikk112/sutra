"""
Real embedder integration tests — require live API / installed packages.

All tests in this file are skipped unless the relevant env var or package is
present.  They make loose assertions (shape, dtype, non-zero variance) and
never snapshot actual vector values — OpenAI has silently updated model weights
in the past, which would break snapshot tests.

Run with a real key:
    OPENAI_API_KEY=sk-... pytest tests/test_real_embedder.py -v
"""
from __future__ import annotations

import os

import numpy as np
import pytest

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
requires_openai = pytest.mark.skipif(
    not OPENAI_KEY,
    reason="OPENAI_API_KEY env var not set — skipping real OpenAI tests",
)


# ---------------------------------------------------------------------------
# OpenAI Embedder
# ---------------------------------------------------------------------------

class TestOpenAIEmbedder:
    @requires_openai
    def test_embed_returns_correct_shape(self) -> None:
        from sutra.core.embedder.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder(api_key=OPENAI_KEY, dimensions=1536)
        vectors = embedder.embed(["hello world"])
        assert vectors.shape == (1, 1536)

    @requires_openai
    def test_embed_returns_float32(self) -> None:
        from sutra.core.embedder.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder(api_key=OPENAI_KEY, dimensions=1536)
        vectors = embedder.embed(["hello world"])
        assert vectors.dtype == np.float32

    @requires_openai
    def test_embed_non_zero_variance(self) -> None:
        from sutra.core.embedder.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder(api_key=OPENAI_KEY, dimensions=1536)
        vectors = embedder.embed(["function foo() returns int", "class Bar extends Baz"])
        # Vectors should not be all zeros or identical
        assert vectors.std() > 0.0
        assert not np.allclose(vectors[0], vectors[1])

    @requires_openai
    def test_dimensions_property_matches_config(self) -> None:
        from sutra.core.embedder.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder(api_key=OPENAI_KEY, dimensions=1536)
        assert embedder.dimensions == 1536

    @requires_openai
    def test_custom_dimensions_768(self) -> None:
        """OpenAI supports dimension reduction — 768 is a valid truncated size."""
        from sutra.core.embedder.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder(api_key=OPENAI_KEY, dimensions=768)
        vectors = embedder.embed(["hello world"])
        assert vectors.shape == (1, 768)
        assert embedder.dimensions == 768

    @requires_openai
    def test_embed_empty_returns_zero_rows(self) -> None:
        from sutra.core.embedder.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder(api_key=OPENAI_KEY, dimensions=1536)
        vectors = embedder.embed([])
        assert vectors.shape == (0, 1536)
        assert vectors.dtype == np.float32

    @requires_openai
    def test_batch_size_respected(self) -> None:
        """Embed more chunks than batch_size to exercise batching."""
        from sutra.core.embedder.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder(api_key=OPENAI_KEY, dimensions=1536, batch_size=3)
        chunks = [f"chunk number {i}" for i in range(7)]
        vectors = embedder.embed(chunks)
        assert vectors.shape == (7, 1536)

    @requires_openai
    def test_shape_invariant_after_full_pipeline(self, tmp_path: pytest.TempPathFactory) -> None:
        """chunk_builder + OpenAIEmbedder produces correct shapes end to end."""
        from sutra.core.embedder.chunk_builder import build_chunks
        from sutra.core.embedder.openai import OpenAIEmbedder
        from sutra.core.extractor.base import (
            ClassSymbol, FunctionSymbol, Location, Visibility
        )

        def _loc():
            return Location(line_start=1, line_end=3, byte_start=0, byte_end=30)

        src_path = tmp_path / "pkg" / "a.py"
        src_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.write_text("def foo() -> None:\n    pass\n", encoding="utf-8")

        func = FunctionSymbol(
            id="sutra python repo pkg/a.py foo().",
            name="foo",
            qualified_name="pkg.foo",
            file_path="pkg/a.py",
            location=_loc(),
            body_hash="sha256:x",
            language="python",
            visibility=Visibility.PUBLIC,
            is_exported=True,
            signature="def foo() -> None",
            parameters=[],
            return_type="None",
            docstring=None,
            decorators=[],
            is_async=False,
            complexity=1,
        )

        chunks, monikers = build_chunks([func], tmp_path)
        embedder = OpenAIEmbedder(api_key=OPENAI_KEY, dimensions=1536)
        vectors = embedder.embed(chunks)

        assert len(chunks) == len(monikers) == vectors.shape[0]
        assert vectors.shape[1] == 1536
        assert vectors.dtype == np.float32
