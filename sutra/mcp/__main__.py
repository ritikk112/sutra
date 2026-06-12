"""
Run the Sutra MCP server.

    # Local stdio (what Claude Code / Claude Desktop / Cursor spawn):
    python -m sutra.mcp --artifacts-dir ~/sutra-artifacts

    # Shared team server over HTTP (set SUTRA_MCP_TOKEN for auth):
    python -m sutra.mcp --artifacts-dir ~/sutra-artifacts --http --port 8765

Environment:
    SUTRA_ARTIFACTS_DIR   default for --artifacts-dir
    SUTRA_MCP_TOKEN       bearer token required on every HTTP request
    OPENAI_API_KEY        only if artifacts were embedded with an OpenAI model
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sutra.mcp.server import SutraServer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sutra.mcp")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=os.environ.get("SUTRA_ARTIFACTS_DIR"),
        help="Directory containing one artifact subdirectory per repo "
             "(or set SUTRA_ARTIFACTS_DIR).",
    )
    parser.add_argument("--http", action="store_true",
                        help="Serve streamable HTTP instead of stdio.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-watch", action="store_true",
                        help="Disable hot-reload on .ready sentinel changes.")
    parser.add_argument("--audit-db", type=Path, default=None,
                        help="SQLite audit log path (default ~/.sutra/mcp_audit.db).")
    args = parser.parse_args(argv)

    if args.artifacts_dir is None:
        parser.error("--artifacts-dir (or SUTRA_ARTIFACTS_DIR) is required")

    try:
        server = SutraServer(
            artifacts_root=args.artifacts_dir,
            watch=not args.no_watch,
            audit_db=args.audit_db,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"[sutra-mcp] serving {len(server.registry.repos())} repo(s): "
        f"{', '.join(server.registry.repos())}",
        file=sys.stderr,
    )

    if args.http:
        server.run_http(host=args.host, port=args.port)
    else:
        server.run_stdio()
    return 0


if __name__ == "__main__":
    sys.exit(main())
