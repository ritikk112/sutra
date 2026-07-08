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
    Build an Embedder from sutra.yaml on disk.

    Loads the YAML file and delegates the actual embedder construction to
    :func:`from_dict`.  This keeps a single source of truth for the provider
    dispatch and validation logic.

    - If config_path is None or the file does not exist, returns FixtureEmbedder.
    - Otherwise the parsed YAML dict is handed to :func:`from_dict`.
    """
    if config_path is None or not Path(config_path).exists():
        return FixtureEmbedder()

    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    return from_dict(config)


def from_dict(config: dict) -> Embedder:
    """
    Build an Embedder from an in-memory config dict (the parsed ``sutra.yaml``).

    This is the shared construction path used by both :func:`from_config` (which
    loads the YAML first) and the ``sutra init`` wizard, which builds a candidate
    embedder from user answers and calls ``.embed(["probe"])`` to validate the
    endpoint and read ``.dimensions`` *before* any config file is written.

    The dict shape mirrors the YAML: ``{"embedder": {"provider": ..., ...}}``.

    - Provider 'fixture' or missing → FixtureEmbedder.
    - Provider 'openai' → OpenAIEmbedder.  Covers both real OpenAI and any
      OpenAI-compatible endpoint (via optional ``base_url``).  The API key env
      var is validated at construction time and ConfigError is raised if it is
      unset — EXCEPT when ``base_url`` is set and ``api_key_env`` is blank/empty
      (e.g. Ollama, which needs no key).  When ``base_url`` is set, ``dimensions``
      must be provided explicitly (enforced by OpenAIEmbedder).
    - Provider 'local' → LocalEmbedder (imports sentence-transformers lazily;
      raises ImportError with a remediation hint if not installed).
    - Unknown provider raises ConfigError immediately.
    """
    embedder_cfg: dict = config.get("embedder") or {}
    provider: str = embedder_cfg.get("provider", "fixture")

    if provider == "fixture":
        dims = int(embedder_cfg.get("dimensions", DEFAULT_FIXTURE_DIMS))
        return FixtureEmbedder(dims=dims)

    if provider == "openai":
        model = embedder_cfg.get("model", "text-embedding-3-small")
        batch_size = int(embedder_cfg.get("batch_size", 100))
        base_url = str(embedder_cfg.get("base_url") or "").strip() or None
        api_key_env = str(embedder_cfg.get("api_key_env", "OPENAI_API_KEY") or "").strip()

        # dimensions is optional in the dict.  For real OpenAI it defaults to
        # 1536 inside OpenAIEmbedder; for a base_url endpoint OpenAIEmbedder
        # requires it explicitly (raises ConfigError if omitted).
        raw_dims = embedder_cfg.get("dimensions")
        dimensions = int(raw_dims) if raw_dims is not None else None

        # A key is required UNLESS pointing at a compatible endpoint (base_url
        # set) that declares no key env (api_key_env blank) — e.g. Ollama.
        key_required = not (base_url and not api_key_env)
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        if key_required and not api_key:
            raise ConfigError(
                f"OpenAI embedder requires the {api_key_env or 'OPENAI_API_KEY'!r} "
                f"environment variable to be set, but it is missing or empty.  "
                f"Export it before running: export {api_key_env or 'OPENAI_API_KEY'}=sk-..."
            )
        # Compatible endpoints that need no key still want a non-empty token for
        # the SDK client; the endpoint ignores it (Ollama et al.).
        if not api_key:
            api_key = "sutra-no-key-required"

        from sutra.core.embedder.openai import OpenAIEmbedder  # noqa: PLC0415
        return OpenAIEmbedder(
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            batch_size=batch_size,
            base_url=base_url,
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
