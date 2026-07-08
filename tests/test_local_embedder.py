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
