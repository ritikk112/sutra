"""
Tests for the cross-agent contract `factory.from_dict(config: dict) -> Embedder`
and the OpenAI-compatible `base_url` support added for `sutra init`.

No mocks — real Embedder instances built from real in-memory config dicts.
The wizard uses `from_dict` to construct a candidate embedder from user answers
and probe it (`.embed([...])` / read `.dimensions`) before any file is written,
so these tests assert construction/validation behavior directly, without a live
server or a written YAML file.

`from_config(path)` now loads YAML and delegates to `from_dict`, so the existing
`test_embedder.py::TestFactory` suite already covers the YAML-loading path.
"""
from __future__ import annotations

import numpy as np
import pytest

from sutra.core.embedder.factory import ConfigError, from_dict
from sutra.core.embedder.fixture import DEFAULT_FIXTURE_DIMS, FixtureEmbedder
from sutra.core.embedder.openai import OpenAIEmbedder


# ---------------------------------------------------------------------------
# from_dict — fixture branch (no deps, no keys)
# ---------------------------------------------------------------------------

class TestFromDictFixture:
    def test_empty_dict_returns_fixture(self) -> None:
        embedder = from_dict({})
        assert isinstance(embedder, FixtureEmbedder)
        assert embedder.dimensions == DEFAULT_FIXTURE_DIMS

    def test_missing_provider_defaults_to_fixture(self) -> None:
        embedder = from_dict({"embedder": {}})
        assert isinstance(embedder, FixtureEmbedder)

    def test_none_embedder_block_defaults_to_fixture(self) -> None:
        # embedder: (null) in YAML parses to None
        embedder = from_dict({"embedder": None})
        assert isinstance(embedder, FixtureEmbedder)

    def test_fixture_custom_dimensions(self) -> None:
        embedder = from_dict({"embedder": {"provider": "fixture", "dimensions": 256}})
        assert isinstance(embedder, FixtureEmbedder)
        assert embedder.dimensions == 256

    def test_fixture_embeds_probe_end_to_end(self) -> None:
        # Mirrors the wizard's probe: build then embed to read real dimensions.
        embedder = from_dict({"embedder": {"provider": "fixture", "dimensions": 128}})
        vectors = embedder.embed(["probe"])
        assert vectors.shape == (1, 128)
        assert vectors.dtype == np.float32


# ---------------------------------------------------------------------------
# from_dict — unknown provider
# ---------------------------------------------------------------------------

class TestFromDictUnknownProvider:
    def test_unknown_provider_raises_config_error(self) -> None:
        with pytest.raises(ConfigError, match="magic"):
            from_dict({"embedder": {"provider": "magic"}})


# ---------------------------------------------------------------------------
# from_dict — openai (real OpenAI, no base_url)
# ---------------------------------------------------------------------------

class TestFromDictOpenAI:
    def test_openai_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            from_dict({"embedder": {"provider": "openai"}})

    def test_openai_empty_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "")
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            from_dict({"embedder": {"provider": "openai"}})

    def test_openai_with_key_builds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-from-dict")
        embedder = from_dict(
            {"embedder": {"provider": "openai", "dimensions": 1536, "batch_size": 50}}
        )
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.dimensions == 1536
        assert embedder.model_id == "openai/text-embedding-3-small"

    def test_openai_default_dimensions_1536_when_omitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No base_url + no dimensions => legacy 1536 default preserved.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-from-dict")
        embedder = from_dict({"embedder": {"provider": "openai"}})
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.dimensions == 1536

    def test_openai_custom_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("MY_CUSTOM_KEY", "sk-custom")
        embedder = from_dict(
            {"embedder": {"provider": "openai", "api_key_env": "MY_CUSTOM_KEY"}}
        )
        assert isinstance(embedder, OpenAIEmbedder)

    def test_openai_custom_api_key_env_missing_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MY_KEY", raising=False)
        with pytest.raises(ConfigError, match="MY_KEY"):
            from_dict({"embedder": {"provider": "openai", "api_key_env": "MY_KEY"}})


# ---------------------------------------------------------------------------
# from_dict — OpenAI-compatible endpoint (base_url) : the new rules
# ---------------------------------------------------------------------------

class TestFromDictOpenAICompatible:
    def test_base_url_blank_key_env_needs_no_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ollama-style: base_url set, api_key_env blank => no key required.
        # No key in the environment at all — construction must still succeed.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = from_dict(
            {
                "embedder": {
                    "provider": "openai",
                    "base_url": "http://localhost:11434/v1",
                    "api_key_env": "",
                    "model": "nomic-embed-text",
                    "dimensions": 768,
                }
            }
        )
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.dimensions == 768
        assert embedder.model_id == "openai/nomic-embed-text"

    def test_base_url_missing_api_key_env_key_defaults_needs_no_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # api_key_env key omitted entirely (not just blank) with base_url set:
        # falls back to blank => still no key required for a compatible endpoint.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = from_dict(
            {
                "embedder": {
                    "provider": "openai",
                    "base_url": "http://localhost:11434/v1",
                    "api_key_env": None,
                    "dimensions": 384,
                }
            }
        )
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.dimensions == 384

    def test_base_url_requires_explicit_dimensions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # base_url set but dimensions omitted => ConfigError (raised by OpenAIEmbedder).
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="dimensions"):
            from_dict(
                {
                    "embedder": {
                        "provider": "openai",
                        "base_url": "http://localhost:11434/v1",
                        "api_key_env": "",
                    }
                }
            )

    def test_base_url_with_declared_key_env_still_requires_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Compatible endpoint that DOES need a key (Together/Azure-style):
        # base_url set AND api_key_env non-blank => key still required.
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="TOGETHER_API_KEY"):
            from_dict(
                {
                    "embedder": {
                        "provider": "openai",
                        "base_url": "https://api.together.xyz/v1",
                        "api_key_env": "TOGETHER_API_KEY",
                        "dimensions": 768,
                    }
                }
            )

    def test_base_url_with_declared_key_env_builds_when_key_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TOGETHER_API_KEY", "sk-together-fake")
        embedder = from_dict(
            {
                "embedder": {
                    "provider": "openai",
                    "base_url": "https://api.together.xyz/v1",
                    "api_key_env": "TOGETHER_API_KEY",
                    "dimensions": 768,
                }
            }
        )
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.dimensions == 768


# ---------------------------------------------------------------------------
# OpenAIEmbedder direct construction — base_url validation unit
# ---------------------------------------------------------------------------

class TestOpenAIEmbedderBaseURL:
    def test_base_url_without_dimensions_raises(self) -> None:
        with pytest.raises(ConfigError, match="dimensions"):
            OpenAIEmbedder(api_key="x", base_url="http://localhost:11434/v1")

    def test_base_url_with_dimensions_builds(self) -> None:
        embedder = OpenAIEmbedder(
            api_key="x", base_url="http://localhost:11434/v1", dimensions=512
        )
        assert embedder.dimensions == 512

    def test_no_base_url_defaults_dimensions_to_1536(self) -> None:
        # Legacy behavior preserved: dimensions optional when base_url is None.
        embedder = OpenAIEmbedder(api_key="x")
        assert embedder.dimensions == 1536

    def test_no_base_url_explicit_dimensions_respected(self) -> None:
        embedder = OpenAIEmbedder(api_key="x", dimensions=768)
        assert embedder.dimensions == 768


# ---------------------------------------------------------------------------
# from_dict — local branch (guarded: skip if sentence-transformers absent)
# ---------------------------------------------------------------------------

def _sentence_transformers_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


@pytest.mark.skipif(
    not _sentence_transformers_installed(),
    reason="sentence-transformers not installed — skipping local embedder test",
)
class TestFromDictLocal:
    def test_local_builds_and_reports_dimensions(self) -> None:
        from sutra.core.embedder.local import LocalEmbedder

        embedder = from_dict(
            {
                "embedder": {
                    "provider": "local",
                    "model": "all-MiniLM-L6-v2",
                    "dimensions": 384,
                    "batch_size": 8,
                }
            }
        )
        assert isinstance(embedder, LocalEmbedder)
        assert embedder.dimensions == 384
