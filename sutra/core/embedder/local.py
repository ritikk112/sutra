from __future__ import annotations

import numpy as np

from sutra.core.embedder.base import Embedder


class LocalEmbedder(Embedder):
    """
    Embedding provider backed by sentence-transformers (local, no API calls).

    sentence-transformers is not installed by default — this module is only
    imported when the factory resolves provider: local.  A clear ImportError
    with a remediation hint is raised if the package is missing.

    The `dimensions` config value is validated against the model's actual
    output at construction time.  If they disagree, ValueError is raised
    immediately rather than silently producing wrong-shape vectors.

    Batching is delegated to sentence-transformers' encode() which handles
    GPU/CPU memory management internally.
    """

    # bge en-family models are asymmetric: queries embed with an instruction
    # prefix, documents without.  LocalEmbedder previously embedded queries
    # bare — a known confound flagged in KIND_FILTER_AB.md.  bge-m3 and the
    # rerankers deliberately take no prefix.
    _BGE_QUERY_INSTRUCTION = (
        "Represent this sentence for searching relevant passages: "
    )

    @staticmethod
    def default_query_instruction(model_name: str) -> str | None:
        """Query-side instruction the model was trained with, or None."""
        import re  # noqa: PLC0415
        if re.fullmatch(r"BAAI/bge-(small|base|large)-en(-v[\d.]+)?", model_name):
            return LocalEmbedder._BGE_QUERY_INSTRUCTION
        return None

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimensions: int = 384,
        batch_size: int = 32,
        query_instruction: str | None = "auto",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)

        # Validate dimensions match model output — config is authoritative.
        test_vec = self._model.encode(["test"], batch_size=1, show_progress_bar=False)
        actual_dims = int(test_vec.shape[1])
        if actual_dims != dimensions:
            raise ValueError(
                f"Config says dimensions={dimensions} but model {model_name!r} "
                f"returns {actual_dims}-dim vectors. Update config or choose a "
                f"different model."
            )
        self._dimensions = dimensions
        self._model_name = model_name
        self._query_instruction = (
            self.default_query_instruction(model_name)
            if query_instruction == "auto"
            else query_instruction
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_id(self) -> str:
        return f"sentence-transformers/{self._model_name}"

    def embed_queries(self, chunks: list[str]) -> np.ndarray:
        """Query-side embedding: apply the model's instruction prefix, if any.
        Document embedding (embed) is untouched — artifacts stay valid."""
        if self._query_instruction:
            chunks = [self._query_instruction + c for c in chunks]
        return self.embed(chunks)

    @staticmethod
    def native_dimensions(model_name: str) -> int:
        """Load `model_name` and return its native embedding width.

        Uses the same encode-probe LocalEmbedder validates against, so the value
        this returns is guaranteed to satisfy the constructor's dims check. Lets
        the setup wizard auto-detect the correct dimension instead of guessing.
        """
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        model = SentenceTransformer(model_name)
        test_vec = model.encode(["probe"], batch_size=1, show_progress_bar=False)
        return int(test_vec.shape[1])

    @staticmethod
    def _wants_progress(n_chunks: int, batch_size: int) -> bool:
        """Show a progress bar only for a real, multi-batch workload.

        A sub-batch embed (a validation probe, a tiny file) finishes in well
        under a second, so a bar there is just noise. Indexing a repo on CPU is
        the slow, silent case a bar actually helps.
        """
        return n_chunks > batch_size

    def embed(self, chunks: list[str]) -> np.ndarray:
        if not chunks:
            return np.empty((0, self._dimensions), dtype=np.float32)

        vectors = self._model.encode(
            chunks,
            batch_size=self._batch_size,
            show_progress_bar=self._wants_progress(len(chunks), self._batch_size),
        )
        return np.asarray(vectors, dtype=np.float32)
