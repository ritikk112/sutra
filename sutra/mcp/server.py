from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from sutra.core.retrieval.types import SearchResult
from sutra.mcp.audit import AuditLog
from sutra.mcp.registry import (
    EmbedderCache,
    ServingUnit,
    SnapshotRegistry,
    build_serving_unit,
    scan_artifacts_root,
)
from sutra.mcp.watcher import ArtifactWatcher

INSTRUCTIONS = """\
Sutra serves a team's indexed repositories for code-aware retrieval.

Start with sutra_list_repos to see what is indexed and to get the exact repo
names the other tools expect.

Two things an index does that a per-repo file search cannot — reach for Sutra
when:
  - The answer may span MULTIPLE repositories.  Call sutra_search with no
    `repo` argument to search every indexed repo in one call; results are
    merged and ordered by score as a pragmatic cut (per-repo scores are not
    directly comparable, so treat cross-repo rank as approximate).
  - You need the call/type graph.  sutra_get_callers / sutra_get_callees return
    a symbol's resolved callers / callees in one call, and
    sutra_expand_neighbors walks resolved callers, callees, type hierarchy and
    references out to `depth` hops — a call chain / partial blast radius (only
    resolved edges are in the graph, so treat it as a lower bound).

sutra_search answers natural-language questions ("which function saves the
meeting in db") with symbols ranked by relevance, each carrying file/line,
signature and docstring.  sutra_get_symbol returns full metadata for one
moniker (the id sutra_search returns), including its resolved callers and
callees.
"""

_DOCSTRING_SUMMARY_MAX = 200
_SIGNATURE_SUMMARY_MAX = 200


def _docstring_summary(doc: Optional[str]) -> Optional[str]:
    """First non-empty line of a docstring, length-capped.

    A search hit only needs the one-line "what does this do" to be picked out
    of a ranked list; the full docstring (and parameters, complexity,
    relationships) stays available via sutra_get_symbol.  Summarising here keeps
    every sutra_search result light without dropping the information — it moves
    the bulk from pushed-on-every-hit to pulled-on-demand.
    """
    if not doc:
        return doc
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            if len(line) > _DOCSTRING_SUMMARY_MAX:
                return line[: _DOCSTRING_SUMMARY_MAX - 1] + "…"
            return line
    return ""


def _signature_summary(sig: Optional[str]) -> Optional[str]:
    """Whitespace-collapsed, length-capped signature for search payloads.

    Raw signatures can be enormous — fastapi's Annotated[..., Doc("...")]
    parameter style put 115K of a 126K two-query search payload in this one
    field, and the vs-control benchmark measured sutra-arm agents growing
    context ~30% faster than grep-only agents largely on tool-result bulk
    (SUTRA_VS_CONTROL.md).  The full signature stays available via
    sutra_get_symbol — same pushed-vs-pulled split as _docstring_summary.
    """
    if not sig:
        return sig
    collapsed = " ".join(sig.split())
    if len(collapsed) > _SIGNATURE_SUMMARY_MAX:
        return collapsed[: _SIGNATURE_SUMMARY_MAX - 1] + "…"
    return collapsed


class SutraServer:
    """
    The assembled product: registry + watcher + audit + FastMCP tools.

    Construction loads every artifact under `artifacts_root`; `watch=True`
    hot-reloads a repo when its `.ready` sentinel changes (ref-counted
    swap — in-flight queries finish on the old snapshot).
    """

    def __init__(
        self,
        artifacts_root: Path | str,
        watch: bool = True,
        audit_db: Optional[Path] = None,
        strict: bool = False,
    ) -> None:
        self.artifacts_root = Path(artifacts_root)
        if not self.artifacts_root.is_dir():
            raise FileNotFoundError(
                f"Artifacts root {self.artifacts_root} does not exist or is "
                f"not a directory."
            )

        self.registry = SnapshotRegistry()
        self.embedders = EmbedderCache()
        self._analyzers: dict[str, Any] = {}
        self.audit = AuditLog(audit_db) if audit_db else AuditLog()

        loaded = scan_artifacts_root(
            self.artifacts_root, self.registry, self.embedders,
            self._analyzers, strict=strict,
        )
        if not loaded:
            raise FileNotFoundError(
                f"No loadable artifacts under {self.artifacts_root} — "
                f"expected subdirectories each containing graph.json + "
                f"embeddings.npy + embeddings_index.json."
            )

        self.watcher: Optional[ArtifactWatcher] = None
        if watch:
            self.watcher = ArtifactWatcher(
                self.artifacts_root, self._reload_artifact,
                on_removed=self._remove_artifact,
            )
            self.watcher.start()

        self.mcp = FastMCP(name="sutra", instructions=INSTRUCTIONS)
        self._register_tools()

    # ------------------------------------------------------------------
    # Hot reload
    # ------------------------------------------------------------------

    def _reload_artifact(self, artifact_dir: Path) -> None:
        """Watcher callback: rebuild the unit, swap atomically.  A torn or
        incompatible artifact raises inside build → watcher logs, old
        snapshot keeps serving."""
        unit = build_serving_unit(artifact_dir, self.embedders, self._analyzers)
        self.registry.swap(unit)

    def _remove_artifact(self, artifact_dir: Path) -> None:
        """Watcher callback: an artifact directory disappeared (e.g. `sutra
        remove`) — hot-unload the repo it served so a running server stops
        answering for it."""
        # Match on the artifact directory's basename (the repo slug), which is
        # unique per artifacts root — robust to path-normalization differences
        # (symlinked/relative root) between the watcher and the loader.
        for repo in self.registry.repos():
            unit = self.registry.get(repo)
            if (
                unit is not None
                and unit.artifact_dir is not None
                and unit.artifact_dir.name == artifact_dir.name
            ):
                self.registry.remove(repo)
                return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unit(self, repo: str) -> ServingUnit:
        unit = self.registry.get(repo)
        if unit is None:
            raise ValueError(
                f"Unknown repo {repo!r}. Indexed repos: {self.registry.repos()}"
            )
        return unit

    def _unit_for_moniker(self, moniker: str) -> ServingUnit:
        # moniker format: "sutra <lang> <repo> <file> <descriptor>"
        parts = moniker.split(" ", 3)
        if len(parts) >= 3:
            unit = self.registry.get(parts[2])
            if unit is not None:
                return unit
        for repo in self.registry.repos():
            unit = self.registry.get(repo)
            if unit is not None and moniker in unit.snapshot.symbols:
                return unit
        raise ValueError(f"No indexed repo contains moniker {moniker!r}.")

    def _symbol_payload(self, unit: ServingUnit, moniker: str) -> dict[str, Any]:
        sym = unit.snapshot.symbols.get(moniker)
        if sym is None:
            return {"moniker": moniker, "indexed": False}
        return {**sym, "moniker": moniker, "repo": unit.repo_name, "indexed": True}

    def _result_payload(
        self, unit: ServingUnit, r: SearchResult, *, include_provenance: bool = False
    ) -> dict[str, Any]:
        sym = unit.snapshot.symbols.get(r.moniker, {})
        loc = sym.get("location") or {}
        payload = {
            "moniker": r.moniker,
            "score": round(r.score, 6),
            "kind": sym.get("kind"),
            "name": sym.get("name"),
            "qualified_name": sym.get("qualified_name"),
            "file_path": sym.get("file_path"),
            "line_start": loc.get("line_start"),
            "line_end": loc.get("line_end"),
            "signature": _signature_summary(sym.get("signature")),
            "docstring": _docstring_summary(sym.get("docstring")),
        }
        if include_provenance:
            payload["provenance"] = {k: round(v, 6) for k, v in r.provenance.items()}
        return payload

    def _audited(self, tool: str, args: dict[str, Any], repo: Optional[str], fn):
        t0 = time.time()
        try:
            result = fn()
        except Exception as exc:
            self.audit.record(
                tool, args, (time.time() - t0) * 1000, repo=repo,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        count = len(result) if isinstance(result, list) else 1
        self.audit.record(
            tool, args, (time.time() - t0) * 1000, repo=repo, result_count=count
        )
        return result

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _register_tools(self) -> None:
        server = self

        @self.mcp.tool(
            name="sutra_list_repos",
            description="List the indexed repositories this server is serving, "
                        "with symbol counts and the commit each was indexed at. "
                        "Call this first to learn which repos are available and "
                        "the exact repo names the other tools expect.",
        )
        def sutra_list_repos() -> list[dict[str, Any]]:
            def run():
                out = []
                for repo in server.registry.repos():
                    unit = server.registry.get(repo)
                    if unit is None:
                        continue
                    snap = unit.snapshot
                    out.append({
                        "repo": repo,
                        "url": snap.repo_url,
                        "commit_sha": snap.commit_sha,
                        "symbols": len(snap.symbols),
                        "embeddings": snap.embedding_count,
                        "embedding_model": snap.embedding_model_id,
                    })
                return out
            return server._audited("sutra_list_repos", {}, None, run)

        @self.mcp.tool(
            name="sutra_search",
            description="Natural-language search over indexed code.  Returns "
                        "symbols (functions, classes, methods …) ranked by "
                        "relevance, each with file/line location, signature and "
                        "a one-line docstring summary (call sutra_get_symbol for "
                        "the full docstring + relationships).  Omit `repo` to "
                        "search every indexed repo in one call — results are "
                        "merged and ordered by score as a pragmatic cut (per-repo "
                        "scores aren't directly comparable); pass `repo` to scope "
                        "to one.  Set include_provenance=true for per-channel "
                        "retrieval scores, rerank=true for a slower, "
                        "higher-precision cross-encoder pass.",
        )
        def sutra_search(
            query: str,
            repo: Optional[str] = None,
            top_k: int = 10,
            rerank: bool = False,
            include_provenance: bool = False,
        ) -> list[dict[str, Any]]:
            args = {"query": query, "repo": repo, "top_k": top_k,
                    "rerank": rerank, "include_provenance": include_provenance}

            def run():
                repos = [repo] if repo else server.registry.repos()
                out: list[dict[str, Any]] = []
                for name in repos:
                    unit = server._unit(name)
                    for r in unit.pipeline.search(query, top_k=top_k, rerank=rerank):
                        payload = server._result_payload(
                            unit, r, include_provenance=include_provenance
                        )
                        payload["repo"] = name
                        out.append(payload)
                # Multi-repo: order by score within equal provenance shapes is
                # not meaningful across repos; sort by score as a pragmatic cut.
                if not repo:
                    out.sort(key=lambda p: -p["score"])
                return out[:top_k]
            return server._audited("sutra_search", args, repo, run)

        @self.mcp.tool(
            name="sutra_get_symbol",
            description="Full metadata for one symbol by its moniker (the id "
                        "returned by sutra_search): signature, docstring, "
                        "location, parameters, complexity, relationships, plus "
                        "its resolved callers and callees.",
        )
        def sutra_get_symbol(moniker: str) -> dict[str, Any]:
            def run():
                unit = server._unit_for_moniker(moniker)
                payload = server._symbol_payload(unit, moniker)
                # is_local is already present in payload when the symbol is found
                # ({**sym} in _symbol_payload includes it). Do not add it on the
                # not-found path ({"moniker": ..., "indexed": False}) — misleading.
                payload["callers"] = [
                    n.moniker for n in unit.traversal.get_callers(moniker)
                ]
                payload["callees"] = [
                    n.moniker for n in unit.traversal.get_callees(moniker)
                ]
                return payload
            return server._audited(
                "sutra_get_symbol", {"moniker": moniker}, None, run
            )

        @self.mcp.tool(
            name="sutra_get_callers",
            description="Symbols that call the given symbol: its resolved "
                        "incoming CALLS edges, returned in one call.",
        )
        def sutra_get_callers(moniker: str) -> list[dict[str, Any]]:
            def run():
                unit = server._unit_for_moniker(moniker)
                return [
                    server._symbol_payload(unit, n.moniker)
                    for n in unit.traversal.get_callers(moniker)
                ]
            return server._audited(
                "sutra_get_callers", {"moniker": moniker}, None, run
            )

        @self.mcp.tool(
            name="sutra_get_callees",
            description="Symbols the given symbol calls: its resolved outgoing "
                        "CALLS edges, returned in one call.",
        )
        def sutra_get_callees(moniker: str) -> list[dict[str, Any]]:
            def run():
                unit = server._unit_for_moniker(moniker)
                return [
                    server._symbol_payload(unit, n.moniker)
                    for n in unit.traversal.get_callees(moniker)
                ]
            return server._audited(
                "sutra_get_callees", {"moniker": moniker}, None, run
            )

        @self.mcp.tool(
            name="sutra_expand_neighbors",
            description="Walk the relationship graph around a symbol out to "
                        "`depth` hops over resolved edges: callers, callees, "
                        "type hierarchy and references — a call chain / partial "
                        "blast radius (only resolved edges are in the graph). "
                        "`kinds` restricts edge kinds (calls, extends, "
                        "implements, references, contains, imports, "
                        "returns_type, parameter_type).",
        )
        def sutra_expand_neighbors(
            moniker: str,
            depth: int = 1,
            kinds: Optional[list[str]] = None,
        ) -> list[dict[str, Any]]:
            args = {"moniker": moniker, "depth": depth, "kinds": kinds}

            def run():
                unit = server._unit_for_moniker(moniker)
                out = []
                for n in unit.traversal.expand_neighbors(
                    moniker, depth=depth, kinds=kinds
                ):
                    payload = server._symbol_payload(unit, n.moniker)
                    payload["edge_kind"] = n.edge_kind
                    payload["direction"] = n.direction
                    payload["depth"] = n.depth
                    out.append(payload)
                return out
            return server._audited("sutra_expand_neighbors", args, None, run)

    # ------------------------------------------------------------------
    # Transports
    # ------------------------------------------------------------------

    def run_stdio(self) -> None:
        """Local zero-infra mode — what an agent on this machine connects to."""
        self.mcp.run(transport="stdio")

    def run_http(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        """
        Shared team-server mode: streamable HTTP + bearer-token auth.

        Auth: if SUTRA_MCP_TOKEN is set, every request must carry
        `Authorization: Bearer <token>`.  Unset = no auth (loopback dev).
        """
        import uvicorn

        uvicorn.run(self._build_http_app(), host=host, port=port)

    def _build_http_app(self):
        """Build the streamable-HTTP ASGI app: the FastMCP app mounted under
        /mcp behind bearer-token auth.

        The outer Starlette app MUST run the mounted FastMCP app's lifespan.
        A Starlette ``Mount`` does NOT propagate a sub-app's lifespan, and
        FastMCP starts its session-manager task group there — without it every
        request 500s with "Task group is not initialized. Make sure to use
        run()." So we forward the mounted app's lifespan to the outer app.
        """
        from starlette.applications import Starlette
        from starlette.middleware import Middleware
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse
        from starlette.routing import Mount

        from mcp.server.transport_security import TransportSecuritySettings

        token = os.environ.get("SUTRA_MCP_TOKEN", "")

        class BearerAuth(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if token and request.headers.get("authorization") != f"Bearer {token}":
                    return JSONResponse({"error": "unauthorized"}, status_code=401)
                return await call_next(request)

        self.mcp.settings.streamable_http_path = "/"
        # The SDK's DNS-rebinding protection defaults to a localhost-only Host
        # allowlist, which 421s every LAN/remote client. This team server is
        # reached over arbitrary host addresses and the bearer token above is
        # the auth boundary, so relax the Host/Origin check here. ALWAYS set
        # SUTRA_MCP_TOKEN when binding a non-localhost interface.
        self.mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        )
        mcp_app = self.mcp.streamable_http_app()
        return Starlette(
            routes=[Mount("/mcp", app=mcp_app)],
            middleware=[Middleware(BearerAuth)],
            lifespan=lambda _outer: mcp_app.router.lifespan_context(_outer),
        )
