"""
P20-full — LSP-backed CALLS resolver (pyright, Python only for now).

Where the heuristic gives up (same method name on several classes, no
disambiguating import), the language server's type inference answers
exactly: textDocument/definition at the callee token resolves through
receiver types, aliases, and re-exports.

Design:
- One warm `pyright-langserver --stdio` process per resolve() run, shared
  across every file (process startup dominates; requests are cheap).
- Only CALLS edges still unresolved after cheaper strategies are queried —
  compose as HeuristicResolver → LspResolver via ChainResolver, so the
  LSP only pays for the hard residue.
- Definition responses are mapped back to monikers via a per-file interval
  index over symbol locations (narrowest containing callable/class wins).
"""
from __future__ import annotations

import json
import os
import re
import select
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from sutra.core.extractor.base import (
    ClassSymbol,
    FunctionSymbol,
    RelationKind,
    Relationship,
    Symbol,
)
from sutra.core.extractor.moniker import parse_moniker
from sutra.core.resolver.base import ResolutionStats, Resolver

_DEFINITION_TIMEOUT_S = 30.0


class LspResolverError(RuntimeError):
    """Fatal pyright failure mid-resolve — aborts the index (nothing published)."""


class _LspClient:
    """Minimal LSP JSON-RPC client over stdio (Content-Length framing)."""

    def __init__(self, command: list[str], root: Path) -> None:
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._next_id = 1
        self._root = root

        self.request("initialize", {
            "processId": os.getpid(),
            "rootUri": root.as_uri(),
            "capabilities": {},
            "workspaceFolders": [
                {"uri": root.as_uri(), "name": root.name}
            ],
        })
        self.notify("initialized", {})

    # ------------------------------------------------------------------

    def request(self, method: str, params: dict, timeout: Optional[float] = None) -> Any:
        msg_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method,
                    "params": params})
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"LSP {method} timed out after {timeout}s")
                ready, _, _ = select.select([self._proc.stdout], [], [], remaining)
                if not ready:
                    raise TimeoutError(f"LSP {method} timed out after {timeout}s")
            msg = self._recv()
            if msg is None:
                raise RuntimeError(
                    f"pyright-langserver died while waiting for {method!r}"
                )
            if msg.get("id") == msg_id and ("result" in msg or "error" in msg):
                if "error" in msg:
                    raise RuntimeError(f"LSP {method} error: {msg['error']}")
                return msg["result"]
            if "id" in msg and "method" in msg:
                self._send({"jsonrpc": "2.0", "id": msg["id"], "result": None})

    def notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def close(self) -> None:
        try:
            self.request("shutdown", {}, timeout=5)
            self.notify("exit", {})
            self._proc.wait(timeout=5)
        except Exception:   # noqa: BLE001 — best-effort teardown
            self._proc.kill()
            self._proc.wait()

    # ------------------------------------------------------------------

    def _send(self, msg: dict) -> None:
        data = json.dumps(msg).encode("utf-8")
        self._proc.stdin.write(
            f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data
        )
        self._proc.stdin.flush()

    def _recv(self) -> Optional[dict]:
        headers: dict[str, str] = {}
        while True:
            line = self._proc.stdout.readline()
            if not line:
                return None
            line = line.decode("ascii").strip()
            if not line:
                break
            key, _, value = line.partition(":")
            headers[key.lower()] = value.strip()
        length = int(headers.get("content-length", 0))
        body = self._proc.stdout.read(length)
        return json.loads(body.decode("utf-8"))


class LspResolver(Resolver):
    """
    pyright-backed resolver for Python CALLS edges.

    Parameters
    ----------
    root : Path
        Repository root on disk — pyright analyzes real files, so this
        resolver runs at INDEX time (the only moment source is guaranteed
        present; the artifact alone cannot power it).
    command : list[str] | None
        Language-server command.  Defaults to pyright-langserver from the
        running interpreter's environment.
    """

    def __init__(
        self,
        root: Path | str,
        command: Optional[list[str]] = None,
        definition_timeout: float = _DEFINITION_TIMEOUT_S,
    ) -> None:
        self._root = Path(root).resolve()
        self._definition_timeout = definition_timeout
        self._command = command or [
            str(Path(sys.executable).parent / "pyright-langserver"), "--stdio",
        ]

    # ------------------------------------------------------------------

    def resolve(
        self,
        symbols: list[Symbol],
        relationships: list[Relationship],
    ) -> ResolutionStats:
        stats = ResolutionStats()

        # Interval index: file → [(line_start, line_end, symbol)] for
        # callables + classes, used to map definition hits → monikers.
        by_file: dict[str, list[tuple[int, int, Symbol]]] = defaultdict(list)
        names = set()
        for sym in symbols:
            if isinstance(sym, (FunctionSymbol, ClassSymbol)):
                by_file[sym.file_path].append(
                    (sym.location.line_start, sym.location.line_end, sym)
                )
                names.add(sym.name)

        # The work list: unresolved Python CALLS with a usable location.
        work: dict[str, list[Relationship]] = defaultdict(list)
        for rel in relationships:
            if rel.kind != RelationKind.CALLS:
                continue
            stats.total_calls += 1
            if rel.is_resolved:
                continue
            stats.unresolved_before += 1
            if not rel.target_name or rel.location is None:
                continue
            caller_file = parse_moniker(rel.source_id).file_path
            if not caller_file.endswith(".py"):
                continue   # pyright is Python-only; other languages later
            if rel.target_name in names:
                stats.matchable += 1
                work[caller_file].append(rel)

        if not work:
            return stats

        client = _LspClient(self._command, self._root)
        try:
            for file_path, rels in sorted(work.items()):
                abs_path = self._root / file_path
                try:
                    text = abs_path.read_text(encoding="utf-8")
                except OSError:
                    continue
                uri = abs_path.as_uri()
                client.notify("textDocument/didOpen", {
                    "textDocument": {
                        "uri": uri, "languageId": "python",
                        "version": 1, "text": text,
                    }
                })
                lines = text.splitlines()
                for rel in rels:
                    pos = self._callee_position(rel, lines)
                    if pos is None:
                        continue
                    try:
                        result = client.request(
                            "textDocument/definition",
                            {
                                "textDocument": {"uri": uri},
                                "position": {"line": pos[0], "character": pos[1]},
                            },
                            timeout=self._definition_timeout,
                        )
                    except TimeoutError:
                        continue   # abandon this one edge; it stays heuristic-resolved
                    except RuntimeError as exc:
                        raise LspResolverError(
                            f"pyright failed mid-resolve: {exc}"
                        ) from exc
                    target = self._map_definition(result, by_file)
                    if target is None:
                        continue
                    rel.target_id = target.id
                    rel.is_resolved = True
                    rel.metadata["resolved_by"] = "lsp"
                    stats.resolved += 1
                    stats.by_rule["lsp"] = stats.by_rule.get("lsp", 0) + 1
                client.notify("textDocument/didClose", {
                    "textDocument": {"uri": uri}
                })
        finally:
            client.close()

        return stats

    # ------------------------------------------------------------------

    def _callee_position(
        self, rel: Relationship, lines: list[str]
    ) -> Optional[tuple[int, int]]:
        """
        0-based (line, character) of the callee NAME token at the call
        site.  The stored location spans the whole call expression
        (`db.insert_one(args…)`); pyright wants the cursor ON the name.
        """
        loc = rel.location
        name = rel.target_name or ""
        line_idx = loc.line_start - 1
        if not (0 <= line_idx < len(lines)) or not name:
            return None
        line = lines[line_idx]
        # Search from the call's start column; whole-word, followed by '('
        # or as an attribute — first occurrence wins.
        for m in re.finditer(rf"\b{re.escape(name)}\b", line):
            if m.start() >= max(0, loc.column_start - 1):
                return line_idx, m.start()
        m = re.search(rf"\b{re.escape(name)}\b", line)
        return (line_idx, m.start()) if m else None

    def _map_definition(
        self,
        result: Any,
        by_file: dict[str, list[tuple[int, int, Symbol]]],
    ) -> Optional[Symbol]:
        """LSP definition response → narrowest containing indexed symbol."""
        if not result:
            return None
        locations = result if isinstance(result, list) else [result]
        for loc in locations:
            uri = loc.get("uri") or loc.get("targetUri", "")
            rng = loc.get("range") or loc.get("targetSelectionRange") or {}
            if not uri.startswith("file://"):
                continue
            path = Path(uri[len("file://"):])
            try:
                rel_path = str(path.resolve().relative_to(self._root)).replace(
                    "\\", "/"
                )
            except ValueError:
                continue   # definition outside the repo (third-party)
            def_line = rng.get("start", {}).get("line", -1) + 1   # 1-based
            best: Optional[tuple[int, Symbol]] = None
            for line_start, line_end, sym in by_file.get(rel_path, ()):
                if line_start <= def_line <= line_end:
                    span = line_end - line_start
                    if best is None or span < best[0]:
                        best = (span, sym)
            if best is not None:
                return best[1]
        return None


class ChainResolver(Resolver):
    """
    Compose resolvers: each stage only sees what the previous left
    unresolved.  The production chain is Heuristic → LSP — cheap rules
    first, type inference for the residue.
    """

    def __init__(self, *resolvers: Resolver) -> None:
        self._resolvers = resolvers

    def resolve(
        self,
        symbols: list[Symbol],
        relationships: list[Relationship],
    ) -> ResolutionStats:
        total = ResolutionStats()
        for i, resolver in enumerate(self._resolvers):
            stats = resolver.resolve(symbols, relationships)
            if i == 0:
                total.total_calls = stats.total_calls
                total.unresolved_before = stats.unresolved_before
                total.matchable = stats.matchable
            total.resolved += stats.resolved
            for rule, n in stats.by_rule.items():
                total.by_rule[rule] = total.by_rule.get(rule, 0) + n
        return total
