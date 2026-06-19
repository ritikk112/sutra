import sys
from pathlib import Path

import pytest

from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.core.resolver.lsp_resolver import LspResolver, LspResolverError, _LspClient

_FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_python_repo"
_FAKE = [sys.executable, str(Path(__file__).parent / "helpers" / "fake_lsp_server.py")]


def _fixture_graph(tmp_path):
    """Index the fixture WITHOUT a resolver to get unresolved Python CALLS."""
    out = tmp_path / "repo"
    result = Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
    ).index(root=_FIXTURE_REPO, repo_url="https://github.com/test/svc", output_dir=out)
    return result.symbols, result.relationships


def test_definition_timeout_is_skipped_not_hung(tmp_path):
    symbols, rels = _fixture_graph(tmp_path)
    resolver = LspResolver(
        root=_FIXTURE_REPO, command=_FAKE + ["stall"], definition_timeout=0.3
    )
    stats = resolver.resolve(symbols, rels)   # must return, not hang
    assert stats.resolved == 0                # nothing resolved by the stub


def test_pyright_crash_fails_the_index(tmp_path):
    symbols, rels = _fixture_graph(tmp_path)
    resolver = LspResolver(root=_FIXTURE_REPO, command=_FAKE + ["crash"])
    with pytest.raises(LspResolverError):
        resolver.resolve(symbols, rels)


def test_close_reaps_process_even_when_shutdown_hangs():
    client = _LspClient(_FAKE + ["stall_shutdown"], _FIXTURE_REPO)
    client.close()
    assert client._proc.poll() is not None   # process is reaped, no leak
