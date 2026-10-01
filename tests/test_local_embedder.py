"""Unit tests for LocalEmbedder helpers that don't require loading a model.

`_wants_progress` is a pure decision function, so it is testable without
sentence-transformers installed (importing the class does not load ST — ST is
imported lazily inside the methods that need it).
"""
from __future__ import annotations

from sutra.core.embedder.local import LocalEmbedder


class TestWantsProgress:
    def test_progress_shown_for_multi_batch_workload(self) -> None:
        # Thousands of chunks over a batch of 100 -> a real, slow workload.
        assert LocalEmbedder._wants_progress(5000, 100) is True

    def test_progress_hidden_for_single_batch(self) -> None:
        # A sub-batch embed (e.g. a validation probe or a tiny file) stays quiet.
        assert LocalEmbedder._wants_progress(50, 100) is False
        assert LocalEmbedder._wants_progress(100, 100) is False

    def test_progress_hidden_for_empty(self) -> None:
        assert LocalEmbedder._wants_progress(0, 32) is False


class TestQueryInstruction:
    """bge-family models want a query-side instruction prefix (documents
    embed bare).  Pure resolution logic — no model load required."""

    def test_bge_models_get_the_bge_prefix(self) -> None:
        prefix = LocalEmbedder.default_query_instruction("BAAI/bge-base-en-v1.5")
        assert prefix == "Represent this sentence for searching relevant passages: "
        assert LocalEmbedder.default_query_instruction("BAAI/bge-small-en-v1.5") == prefix
        assert LocalEmbedder.default_query_instruction("BAAI/bge-large-en") == prefix

    def test_non_bge_models_get_none(self) -> None:
        assert LocalEmbedder.default_query_instruction("all-MiniLM-L6-v2") is None
        assert LocalEmbedder.default_query_instruction("BAAI/bge-m3") is None  # m3 wants no prefix
        assert LocalEmbedder.default_query_instruction("BAAI/bge-reranker-base") is None

    def test_embed_queries_prepends_instruction(self) -> None:
        # Build an instance without running __init__ (no model download).
        emb = LocalEmbedder.__new__(LocalEmbedder)
        emb._query_instruction = "PREFIX: "
        captured = {}

        def fake_embed(chunks):
            captured["chunks"] = chunks
            return "vecs"

        emb.embed = fake_embed
        assert emb.embed_queries(["find the thing"]) == "vecs"
        assert captured["chunks"] == ["PREFIX: find the thing"]

    def test_embed_queries_without_instruction_is_plain_embed(self) -> None:
        emb = LocalEmbedder.__new__(LocalEmbedder)
        emb._query_instruction = None
        captured = {}

        def fake_embed(chunks):
            captured["chunks"] = chunks
            return "vecs"

        emb.embed = fake_embed
        assert emb.embed_queries(["find the thing"]) == "vecs"
        assert captured["chunks"] == ["find the thing"]
