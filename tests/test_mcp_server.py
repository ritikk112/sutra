"""
Priority 19 (server half) — the assembled SutraServer exercised through a
REAL MCP client session over the SDK's in-memory transport.  No mocks: the
server boots from real artifacts on disk (fixture repo + FixtureEmbedder +
resolver), the client speaks actual MCP, every tool is called end-to-end,
and the audit log is checked afterwards.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mcp.shared.memory import create_connected_server_and_client_session

from sutra.core.artifact.atomic_writer import ARTIFACT_FILES, AtomicArtifactWriter
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.core.resolver import HeuristicResolver
from sutra.mcp.server import SutraServer, _docstring_summary

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE_REPO = _FIXTURES / "sample_python_repo"


def _index_fixture(out: Path) -> None:
    Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
        resolver=HeuristicResolver(),
    ).index(
        root=_FIXTURE_REPO,
        repo_url="https://github.com/test/sample_python_repo",
        output_dir=out,
    )


@pytest.fixture(scope="module")
def server(tmp_path_factory) -> SutraServer:
    root = tmp_path_factory.mktemp("mcp_artifacts")
    _index_fixture(root / "sample_python_repo")
    audit_db = tmp_path_factory.mktemp("audit") / "audit.db"
    # watch=False: hot reload is driven manually in its own test below.
    return SutraServer(artifacts_root=root, watch=False, audit_db=audit_db)


def _call(server: SutraServer, tool: str, args: dict):
    """One MCP round-trip over the in-memory transport."""
    async def run():
        async with create_connected_server_and_client_session(
            server.mcp, raise_exceptions=True
        ) as session:
            return await session.call_tool(tool, args)
    return asyncio.run(run())


def _payload(result):
    """structuredContent if present, else parsed text content."""
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(result.content[0].text)


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_lists_all_six_tools(self, server) -> None:
        async def run():
            async with create_connected_server_and_client_session(
                server.mcp, raise_exceptions=True
            ) as session:
                return await session.list_tools()
        tools = {t.name for t in asyncio.run(run()).tools}
        assert tools == {
            "sutra_list_repos",
            "sutra_search",
            "sutra_get_symbol",
            "sutra_get_callers",
            "sutra_get_callees",
            "sutra_expand_neighbors",
        }


# ---------------------------------------------------------------------------
# Tools, end to end
# ---------------------------------------------------------------------------

class TestTools:
    def test_list_repos(self, server) -> None:
        result = _call(server, "sutra_list_repos", {})
        repos = _payload(result)["result"]
        assert len(repos) == 1
        assert repos[0]["repo"] == "test/sample_python_repo"
        assert repos[0]["symbols"] == 8
        assert repos[0]["embedding_model"] == "fixture-384"

    def test_search_returns_ranked_symbols(self, server) -> None:
        result = _call(server, "sutra_search", {
            "query": "create_user", "repo": "test/sample_python_repo", "top_k": 5,
        })
        hits = _payload(result)["result"]
        assert hits
        assert "create_user" in hits[0]["moniker"]
        assert hits[0]["repo"] == "test/sample_python_repo"
        assert hits[0]["file_path"] == "src/services/user.py"
        assert hits[0]["line_start"] is not None

    def test_search_omits_provenance_by_default(self, server) -> None:
        """Default search payload is trimmed: no per-channel provenance (opt-in),
        but the locating fields an agent needs are all present."""
        result = _call(server, "sutra_search", {
            "query": "create_user", "repo": "test/sample_python_repo", "top_k": 5,
        })
        hits = _payload(result)["result"]
        assert hits
        assert "provenance" not in hits[0]
        for field in ("moniker", "score", "file_path", "line_start", "signature", "docstring"):
            assert field in hits[0]

    def test_search_include_provenance_returns_channels(self, server) -> None:
        """include_provenance=True restores the per-channel scores for callers
        (e.g. benchmarking) that want retrieval transparency."""
        result = _call(server, "sutra_search", {
            "query": "create_user", "repo": "test/sample_python_repo",
            "top_k": 5, "include_provenance": True,
        })
        hits = _payload(result)["result"]
        assert "rrf" in hits[0]["provenance"]

    def test_search_all_repos_when_repo_omitted(self, server) -> None:
        result = _call(server, "sutra_search", {"query": "create_user"})
        assert _payload(result)["result"]

    def test_search_unknown_repo_is_an_error(self, server) -> None:
        result = _call(server, "sutra_search", {
            "query": "x", "repo": "not_indexed",
        })
        assert result.isError
        assert "Unknown repo" in result.content[0].text

    def test_get_symbol_with_callers_callees(self, server) -> None:
        unit = server.registry.get("test/sample_python_repo")
        create_user = next(m for m in unit.snapshot.symbols if "create_user" in m)
        generate_id = next(m for m in unit.snapshot.symbols if "_generate_id" in m)

        payload = _payload(_call(server, "sutra_get_symbol", {"moniker": create_user}))
        assert payload["indexed"] is True
        assert payload["kind"] == "method"
        assert payload["signature"]
        assert generate_id in payload["callees"]

    def test_get_callers_and_callees(self, server) -> None:
        unit = server.registry.get("test/sample_python_repo")
        create_user = next(m for m in unit.snapshot.symbols if "create_user" in m)
        generate_id = next(m for m in unit.snapshot.symbols if "_generate_id" in m)

        callees = _payload(_call(server, "sutra_get_callees", {"moniker": create_user}))["result"]
        assert generate_id in [c["moniker"] for c in callees]

        callers = _payload(_call(server, "sutra_get_callers", {"moniker": generate_id}))["result"]
        assert create_user in [c["moniker"] for c in callers]

    def test_expand_neighbors_with_kinds(self, server) -> None:
        unit = server.registry.get("test/sample_python_repo")
        create_user = next(m for m in unit.snapshot.symbols if "create_user" in m)
        neighbors = _payload(_call(server, "sutra_expand_neighbors", {
            "moniker": create_user, "depth": 2, "kinds": ["calls"],
        }))["result"]
        assert neighbors
        assert all(n["edge_kind"] == "calls" for n in neighbors)
        assert all(n["depth"] in (1, 2) for n in neighbors)

    def test_audit_log_recorded_every_call(self, server) -> None:
        before = len(server.audit.recent(limit=1000))
        _call(server, "sutra_list_repos", {})
        rows = server.audit.recent(limit=1000)
        assert len(rows) == before + 1
        assert rows[0]["tool"] == "sutra_list_repos"
        assert rows[0]["duration_ms"] >= 0
        assert rows[0]["error"] is None

    def test_audit_log_records_errors(self, server) -> None:
        _call(server, "sutra_search", {"query": "x", "repo": "not_indexed"})
        rows = server.audit.recent(limit=5)
        errored = [r for r in rows if r["error"]]
        assert errored
        assert "Unknown repo" in errored[0]["error"]


# ---------------------------------------------------------------------------
# Boot validation + hot reload
# ---------------------------------------------------------------------------

class TestBootAndReload:
    def test_refuses_empty_artifacts_root(self, tmp_path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No loadable artifacts"):
            SutraServer(artifacts_root=empty, watch=False,
                        audit_db=tmp_path / "a.db")

    def test_refuses_missing_root(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            SutraServer(artifacts_root=tmp_path / "nope", watch=False,
                        audit_db=tmp_path / "a.db")

    def test_hot_reload_swaps_snapshot(self, tmp_path) -> None:
        root = tmp_path / "artifacts"
        served = root / "sample_python_repo"
        _index_fixture(served)
        server = SutraServer(artifacts_root=root, watch=True,
                             audit_db=tmp_path / "a.db")
        try:
            old_unit = server.registry.get("test/sample_python_repo")

            # Re-commit the same artifact through the atomic writer —
            # the .ready sentinel appears, the watcher must swap units.
            staged = tmp_path / "rebuild"
            _index_fixture(staged)

            def write(staging: Path) -> None:
                for name in ARTIFACT_FILES:
                    (staging / name).write_bytes((staged / name).read_bytes())

            AtomicArtifactWriter().commit(served, write, generation="gen-2")
            fired = server.watcher.check_once()
            assert served in fired

            new_unit = server.registry.get("test/sample_python_repo")
            assert new_unit is not old_unit
            assert new_unit.snapshot.repo_name == "test/sample_python_repo"
        finally:
            server.watcher.stop()

    def test_torn_reload_keeps_old_snapshot(self, tmp_path) -> None:
        root = tmp_path / "artifacts"
        served = root / "sample_python_repo"
        _index_fixture(served)
        server = SutraServer(artifacts_root=root, watch=True,
                             audit_db=tmp_path / "a.db")
        try:
            old_unit = server.registry.get("test/sample_python_repo")

            # Tear the on-disk artifact, then stamp .ready by hand (a
            # buggy producer).  The reload must fail INSIDE the callback
            # and the registry must keep serving the old snapshot.
            index = json.loads((served / "embeddings_index.json").read_text())
            (served / "embeddings_index.json").write_text(json.dumps(index[:-1]))
            (served / ".ready").write_text("gen-torn")

            fired = server.watcher.check_once()
            assert served in fired   # watcher fired and survived the error
            assert server.registry.get("test/sample_python_repo") is old_unit
        finally:
            server.watcher.stop()

    def test_removed_artifact_dir_is_unloaded_live(self, tmp_path) -> None:
        """`sutra remove` deletes an artifact dir; a running server's watcher
        must hot-unload that repo from the registry within one poll."""
        import shutil
        root = tmp_path / "artifacts"
        served = root / "sample_python_repo"
        _index_fixture(served)
        server = SutraServer(artifacts_root=root, watch=True,
                             audit_db=tmp_path / "a.db")
        try:
            assert "test/sample_python_repo" in server.registry.repos()

            shutil.rmtree(served)                 # what `sutra remove` does
            server.watcher.check_once()           # the running server's poll
            assert "test/sample_python_repo" not in server.registry.repos()
        finally:
            server.watcher.stop()


class TestHttpTransport:
    """The streamable-HTTP team server must serve a REMOTE client, exercising
    two fixes at once:
      - the mounted FastMCP session-manager lifespan must run (else every
        request 500s 'Task group is not initialized'); and
      - the SDK's default Host allowlist is localhost-only, so a LAN client's
        Host header (e.g. 192.168.x.x:8765) must not be 421'd.
    An `initialize` POST carrying a non-localhost Host must therefore succeed.
    """

    def test_http_app_serves_remote_lan_host(self, server, monkeypatch):
        monkeypatch.delenv("SUTRA_MCP_TOKEN", raising=False)  # auth off
        from starlette.testclient import TestClient

        app = server._build_http_app()
        # Entering the TestClient context runs the app's lifespan (session
        # manager). The Host header simulates a LAN client, not localhost.
        with TestClient(app) as client:
            r = client.post(
                "/mcp/",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "host": "192.168.7.94:8765",
                },
                json={
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "smoke", "version": "0"},
                    },
                },
            )
        # 500 = lifespan bug; 421 = Host-allowlist bug. Real remote init = 200.
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        assert "Task group is not initialized" not in r.text


# ---------------------------------------------------------------------------
# _docstring_summary — the search-payload trim helper (pure, no server)
# ---------------------------------------------------------------------------

class TestDocstringSummary:
    def test_takes_first_nonempty_line(self) -> None:
        doc = "Create a user and persist it.\n\nLong details follow.\nmore"
        assert _docstring_summary(doc) == "Create a user and persist it."

    def test_skips_leading_blank_lines(self) -> None:
        assert _docstring_summary("\n\n   Summary here.\nrest") == "Summary here."

    def test_none_and_empty_pass_through(self) -> None:
        assert _docstring_summary(None) is None
        assert _docstring_summary("") == ""

    def test_caps_a_very_long_first_line(self) -> None:
        out = _docstring_summary("x" * 300)
        assert len(out) == 200
        assert out.endswith("…")
