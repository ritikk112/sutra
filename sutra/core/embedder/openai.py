from __future__ import annotations

import os

import numpy as np
from openai import OpenAI

from sutra.core.embedder.base import Embedder


class OpenAIEmbedder(Embedder):
    """
    Embedding provider backed by the OpenAI Embeddings API.

    Uses the sync client (openai.OpenAI) — no asyncio overhead.
    Batching is internal: the caller passes the full chunk list and the
    embedder batches by `batch_size`.  This means the event loop is never
    involved and one embed() call pays HTTP overhead once per batch, not once
    per call.

    Retry logic: `max_retries=3` on the client constructor — the SDK handles
    exponential back-off for rate limits and transient 5xx errors.

    Partial failure: if any batch fails after retries, the exception propagates
    and embed() raises.  The caller (Indexer) lets the error surface — the user
    re-runs.  This is the Phase 1 policy; partial recovery is deferred.

    The `dimensions` param flows directly to the API as the `dimensions` field,
    which OpenAI supports for truncated embeddings (e.g., 768 instead of 1536
    for text-embedding-3-small).  Config is authoritative for this value.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 100,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._usage_prompt_tokens = 0
        self._usage_total_tokens = 0
        self._usage_request_count = 0
        self._last_estimated_cost_usd = 0.0
        # max_retries=3: SDK handles exponential back-off for rate limits / 5xx.
        self._client = OpenAI(api_key=api_key, max_retries=3)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, chunks: list[str]) -> np.ndarray:
        if not chunks:
            return np.empty((0, self._dimensions), dtype=np.float32)

        self._usage_prompt_tokens = 0
        self._usage_total_tokens = 0
        self._usage_request_count = 0
        self._last_estimated_cost_usd = 0.0

        all_vectors: list[np.ndarray] = []

        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._usage_prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self._usage_total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
            self._usage_request_count += 1
            # Sort by index to guarantee order — API doesn't guarantee it.
            batch_vectors = [
                np.array(item.embedding, dtype=np.float32)
                for item in sorted(response.data, key=lambda x: x.index)
            ]
            all_vectors.extend(batch_vectors)

        result = np.stack(all_vectors).astype(np.float32)
        assert result.shape == (len(chunks), self._dimensions), (
            f"Shape mismatch: expected ({len(chunks)}, {self._dimensions}), "
            f"got {result.shape}"
        )
        self._last_estimated_cost_usd = (
            self._usage_total_tokens / 1_000_000
        ) * self._usd_per_1m_tokens()
        return result

    def usage_stats(self) -> dict[str, float | int | str]:
        return {
            "provider": "openai",
            "model": self._model,
            "prompt_tokens": self._usage_prompt_tokens,
            "total_tokens": self._usage_total_tokens,
            "request_count": self._usage_request_count,
            "usd_per_1m_tokens": self._usd_per_1m_tokens(),
            "estimated_cost_usd": round(self._last_estimated_cost_usd, 8),
        }

    def _usd_per_1m_tokens(self) -> float:
        """
        Return pricing used for cost estimation.

        Override with SUTRA_EMBED_COST_PER_1M_TOKENS to track current billing.
        """
        env = os.environ.get("SUTRA_EMBED_COST_PER_1M_TOKENS", "").strip()
        if env:
            try:
                return float(env)
            except ValueError:
                pass

        default_by_model = {
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13,
        }
        return float(default_by_model.get(self._model, 0.02))
