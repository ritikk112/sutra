from __future__ import annotations

import os
from pathlib import Path

import yaml

from sutra.core.embedder.base import Embedder
from sutra.core.embedder.fixture import DEFAULT_FIXTURE_DIMS, FixtureEmbedder


class ConfigError(Exception):
    """Raised when the embedder config is invalid or a required env var is missing."""


def from_config(config_path: Path | None = None) -> Embedder:
    """
    Build an Embedder from sutra.yaml config.

    - If config_path is None or the file does not exist, returns FixtureEmbedder.
    - If provider is 'fixture' or missing, returns FixtureEmbedder.
    - If provider is 'openai', validates the API key env var at construction
      time and raises ConfigError if it is unset.  Never falls back silently.
    - If provider is 'local', imports sentence-transformers (raises ImportError
      with a remediation hint if not installed).
    - Unknown provider raises ConfigError immediately.
    """
    if config_path is None or not Path(config_path).exists():
        return FixtureEmbedder()

    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    embedder_cfg: dict = config.get("embedder", {})
    provider: str = embedder_cfg.get("provider", "fixture")

    if provider == "fixture":
        dims = int(embedder_cfg.get("dimensions", DEFAULT_FIXTURE_DIMS))
        return FixtureEmbedder(dims=dims)

    if provider == "openai":
        model = embedder_cfg.get("model", "text-embedding-3-small")
        dimensions = int(embedder_cfg.get("dimensions", 1536))
        batch_size = int(embedder_cfg.get("batch_size", 100))
        api_key_env = embedder_cfg.get("api_key_env", "OPENAI_API_KEY")

        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ConfigError(
                f"OpenAI embedder requires the {api_key_env!r} environment variable "
                f"to be set, but it is missing or empty.  "
                f"Export it before running: export {api_key_env}=sk-..."
            )

        from sutra.core.embedder.openai import OpenAIEmbedder  # noqa: PLC0415
        return OpenAIEmbedder(
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            batch_size=batch_size,
        )

    if provider == "local":
        model_name = embedder_cfg.get("model", "all-MiniLM-L6-v2")
        dimensions = int(embedder_cfg.get("dimensions", DEFAULT_FIXTURE_DIMS))
        batch_size = int(embedder_cfg.get("batch_size", 32))

        from sutra.core.embedder.local import LocalEmbedder  # noqa: PLC0415
        return LocalEmbedder(
            model_name=model_name,
            dimensions=dimensions,
            batch_size=batch_size,
        )

    raise ConfigError(
        f"Unknown embedder provider: {provider!r}. "
        f"Valid options: 'openai', 'local', 'fixture'."
    )
