from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

# Same home directory as the local UI's jobs.db (~/.sutra) — one place to
# look for everything Sutra writes on a machine.
DEFAULT_AUDIT_DB = Path.home() / ".sutra" / "mcp_audit.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    tool        TEXT    NOT NULL,
    args        TEXT    NOT NULL,
    repo        TEXT,
    duration_ms REAL    NOT NULL,
    result_count INTEGER,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS tool_calls_ts ON tool_calls (ts);
"""


class AuditLog:
    """
    Every MCP tool call → one SQLite row.  Append-only, thread-safe via a
    short-lived connection per write (SQLite handles its own locking; the
    audit path must never contend with query serving).
    """

    def __init__(self, db_path: Path | str = DEFAULT_AUDIT_DB) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def record(
        self,
        tool: str,
        args: dict[str, Any],
        duration_ms: float,
        repo: Optional[str] = None,
        result_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        row = (
            time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            tool,
            json.dumps(args, default=str),
            repo,
            duration_ms,
            result_count,
            error,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO tool_calls "
                "(ts, tool, args, repo, duration_ms, result_count, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                row,
            )

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM tool_calls ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)
