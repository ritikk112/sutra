"""Tests for `sutra.cli.embed_setup` validation.

No mocks.  Two real paths:
* FixtureEmbedder probe (no deps, no network) via `probe_embedder`.
* A REAL in-process OpenAI-compatible HTTP server (threaded http.server on an
  ephemeral 127.0.0.1 port) that serves `/v1/embeddings`, exercising both
  `discover_dimensions` (auto-discovery) and `probe_embedder` end-to-end.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from sutra.cli import embed_setup

# Native width our dummy endpoint "returns" — the value discovery must find.
SERVER_DIMS = 16


class _EmbeddingsHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silence test output
        pass

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        if not self.path.endswith("/embeddings"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        inputs = body.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        # A deterministic, non-zero vector of the server's native width per input.
        data = [
            {
                "object": "embedding",
                "index": i,
                "embedding": [float((i + j + 1) % 7) / 7.0 for j in range(SERVER_DIMS)],
            }
            for i, _ in enumerate(inputs)
        ]
        payload = {
            "object": "list",
            "data": data,
            "model": body.get("model", "dummy-embed"),
            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
        }
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def dummy_openai_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbeddingsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}/v1"
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# FixtureEmbedder path
# ---------------------------------------------------------------------------

class TestProbeFixture:
    def test_fixture_probe_ok(self) -> None:
        result = embed_setup.probe_embedder(
            {"provider": "fixture", "dimensions": 128}
        )
        assert result.ok is True
        assert result.dimensions == 128
        assert result.vector_len == 128
        assert result.model_id == "fixture-128"

    def test_probe_failure_is_captured_not_raised(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # openai provider with no key -> ConfigError inside from_dict, captured.
        result = embed_setup.probe_embedder({"provider": "openai"})
        assert result.ok is False
        assert result.dimensions is None
        assert "OPENAI_API_KEY" in (result.error or "")


# ---------------------------------------------------------------------------
# Real dummy OpenAI-compatible server path
# ---------------------------------------------------------------------------

class TestCompatibleEndpoint:
    def test_discover_dimensions_from_live_server(self, dummy_openai_server: str) -> None:
        dims = embed_setup.discover_dimensions(dummy_openai_server, "dummy-embed")
        assert dims == SERVER_DIMS

    def test_probe_end_to_end_against_server(self, dummy_openai_server: str) -> None:
        # Mirror the wizard: discover dims, then validate via from_dict + embed.
        dims = embed_setup.discover_dimensions(dummy_openai_server, "dummy-embed")
        cfg = {
            "provider": "openai",
            "base_url": dummy_openai_server,
            "api_key_env": "",
            "model": "dummy-embed",
            "dimensions": dims,
        }
        result = embed_setup.probe_embedder(cfg)
        assert result.ok is True
        assert result.dimensions == SERVER_DIMS
        assert result.vector_len == SERVER_DIMS
        assert result.model_id == "openai/dummy-embed"

    def test_direct_embed_shape_via_from_dict(self, dummy_openai_server: str) -> None:
        from sutra.core.embedder.factory import from_dict

        embedder = from_dict(
            {
                "embedder": {
                    "provider": "openai",
                    "base_url": dummy_openai_server,
                    "api_key_env": "",
                    "model": "dummy-embed",
                    "dimensions": SERVER_DIMS,
                }
            }
        )
        vectors = embedder.embed(["alpha", "beta", "gamma"])
        assert vectors.shape == (3, SERVER_DIMS)
        assert vectors.dtype == np.float32
