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

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimensions: int = 384,
        batch_size: int = 32,
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

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, chunks: list[str]) -> np.ndarray:
        if not chunks:
            return np.empty((0, self._dimensions), dtype=np.float32)

        vectors = self._model.encode(
            chunks,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
