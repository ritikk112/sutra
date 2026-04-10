from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sqlite3
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JOB_STATUSES_FINISHED = {"succeeded", "failed", "cancelled"}


@dataclass
class JobRuntime:
    id: str
    log_buffer: deque[dict[str, str]]
    subscribers: set[asyncio.Queue[dict[str, Any]]]
    process: asyncio.subprocess.Process | None


class JobManager:
    def __init__(self, repo_root: Path, pg_url: str, sutra_home: Path) -> None:
        self.repo_root = repo_root
        self.pg_url = pg_url
        self.sutra_home = sutra_home
        self.jobs_root = sutra_home / "jobs"
        self.db_path = sutra_home / "jobs.db"

        self._pending: deque[str] = deque()
        self._wake = asyncio.Event()
        self._runtime: dict[str, JobRuntime] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._init_db)
        await asyncio.to_thread(self._mark_stale_running_jobs)
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

    async def enqueue(self, repo_url: str, replace: bool) -> dict[str, Any]:
        now = _now_iso()
        repo_slug = _repo_slug(repo_url)
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        job_id = f"{repo_slug}-{stamp}"

        clone_path = Path("/tmp") / "sutra-jobs" / job_id / "repo"
        output_path = self.jobs_root / job_id / "out"
        output_path.mkdir(parents=True, exist_ok=True)

        async with self._lock:
            self._pending.append(job_id)
            queue_position = len(self._pending)
            await asyncio.to_thread(
                self._insert_job,
                job_id,
                repo_url,
                int(replace),
                "queued",
                queue_position,
                str(clone_path),
                str(output_path),
                now,
            )
            runtime = JobRuntime(
                id=job_id,
                log_buffer=deque(maxlen=10000),
                subscribers=set(),
                process=None,
            )
            self._runtime[job_id] = runtime
            await self._recompute_queue_positions_locked()
            self._wake.set()

        return {"jobId": job_id, "queuePosition": queue_position}

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_job, job_id)

    async def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_jobs, limit)

    async def cost_summary(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._cost_summary)

    async def subscribe(self, job_id: str) -> tuple[JobRuntime | None, asyncio.Queue[dict[str, Any]]]:
        async with self._lock:
            runtime = self._runtime.get(job_id)
            q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
            if runtime is None:
                return None, q
            runtime.subscribers.add(q)
            return runtime, q

    async def unsubscribe(self, job_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            runtime = self._runtime.get(job_id)
            if runtime is not None:
                runtime.subscribers.discard(q)

    async def cancel(self, job_id: str) -> tuple[int, dict[str, Any]]:
        async with self._lock:
            job = await asyncio.to_thread(self._get_job, job_id)
            if not job:
                return 404, {"detail": "job not found"}
            if job["status"] in JOB_STATUSES_FINISHED:
                return 409, {"detail": "job already finished"}

            if job_id in self._pending:
                self._pending = deque([j for j in self._pending if j != job_id])
                await asyncio.to_thread(
                    self._update_job_status,
                    job_id,
                    "cancelled",
                    1,
                    "Cancelled while queued",
                    None,
                )
                runtime = self._runtime.get(job_id)
                if runtime:
                    await self._publish(runtime, "completed", await asyncio.to_thread(self._get_job, job_id))
                await self._recompute_queue_positions_locked()
                return 200, {"status": "cancelled"}

            runtime = self._runtime.get(job_id)
            if not runtime or not runtime.process or runtime.process.returncode is not None:
                return 409, {"detail": "job is not running"}

            runtime.process.terminate()
            asyncio.create_task(self._force_kill_if_needed(job_id, runtime.process, timeout_sec=10))
            self._append_log_locked(runtime, "system", "Cancellation requested: SIGTERM sent")
            return 200, {"status": "cancelling"}

    async def _force_kill_if_needed(
        self, job_id: str, process: asyncio.subprocess.Process, timeout_sec: int
    ) -> None:
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            async with self._lock:
                runtime = self._runtime.get(job_id)
                if runtime:
                    self._append_log_locked(runtime, "system", "Process did not stop in 10s: SIGKILL sent")

    async def _worker_loop(self) -> None:
        while True:
            await self._wake.wait()
            while True:
                async with self._lock:
                    if not self._pending:
                        self._wake.clear()
                        break
                    job_id = self._pending.popleft()
                    await self._recompute_queue_positions_locked()

                await self._run_job(job_id)

    async def _run_job(self, job_id: str) -> None:
        job = await asyncio.to_thread(self._get_job, job_id)
        if not job:
            return
        if job["status"] == "cancelled":
            return

        started_at = _now_iso()
        await asyncio.to_thread(self._set_running, job_id, started_at)

        async with self._lock:
            runtime = self._runtime[job_id]
            await self._publish(runtime, "started", await asyncio.to_thread(self._get_job, job_id))

        clone_path = Path(job["clone_path"])
        output_path = Path(job["output_path"])
        repo_url = job["repo_url"]
        replace = bool(job["replace"])

        clone_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)

        exit_code = 1
        error = None
        error_detail = None
        summary_json = None

        try:
            runtime = self._runtime[job_id]
            clone_cmd = ["git", "clone", repo_url, str(clone_path)]
            clone_rc = await self._run_cmd(job_id, runtime, clone_cmd)
            if clone_rc != 0:
                raise RuntimeError(f"git clone failed with exit code {clone_rc}")

            cmd = [
                sys.executable,
                "-m",
                "pipelines.full_index",
                "--root",
                str(clone_path),
                "--repo-url",
                repo_url,
                "--output-dir",
                str(output_path),
                "--config",
                str(self.repo_root / "config" / "sutra.yaml"),
                "--pg-url",
                self.pg_url,
            ]
            if replace:
                cmd.append("--replace")

            exit_code = await self._run_cmd(job_id, runtime, cmd)
            graph_path = output_path / "graph.json"
            if graph_path.exists():
                summary_json = self._parse_summary_json(graph_path)
            if exit_code == 0:
                status = "succeeded"
                if summary_json:
                    summary = json.loads(summary_json)
                else:
                    summary = {}
                fallback = self._extract_usage_from_logs(runtime.log_buffer)
                if fallback:
                    summary.setdefault("embedding_total_tokens", fallback.get("embedding_total_tokens"))
                    summary.setdefault("embedding_estimated_cost_usd", fallback.get("embedding_estimated_cost_usd"))
                    summary_json = json.dumps(summary)
                if summary_json:
                    with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
                        summary = json.loads(summary_json)
                        tokens = summary.get("embedding_total_tokens")
                        cost = summary.get("embedding_estimated_cost_usd")
                        if tokens is not None:
                            async with self._lock:
                                self._append_log_locked(
                                    runtime,
                                    "system",
                                    f"Embedding tokens used: {tokens}",
                                )
                        if cost is not None:
                            async with self._lock:
                                self._append_log_locked(
                                    runtime,
                                    "system",
                                    f"Estimated embedding cost (USD): {float(cost):.8f}",
                                )
            elif exit_code < 0:
                status = "cancelled"
                error = f"terminated by signal {-exit_code}"
            else:
                status = "failed"
                error = f"pipeline exited with code {exit_code}"
                error_detail = self._extract_failure_detail(runtime.log_buffer)
        except Exception as exc:
            status = "failed"
            error = str(exc)
            runtime = self._runtime.get(job_id)
            if runtime:
                error_detail = self._extract_failure_detail(runtime.log_buffer)
        finally:
            shutil.rmtree(clone_path.parent, ignore_errors=True)

        finished_at = _now_iso()
        await asyncio.to_thread(
            self._set_finished,
            job_id,
            status,
            finished_at,
            exit_code,
            error,
            error_detail,
            summary_json,
        )

        async with self._lock:
            runtime = self._runtime.get(job_id)
            if runtime:
                await self._publish(runtime, "completed", await asyncio.to_thread(self._get_job, job_id))

    async def _run_cmd(self, job_id: str, runtime: JobRuntime, cmd: list[str]) -> int:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )

        await asyncio.to_thread(self._set_pid, job_id, proc.pid)

        async with self._lock:
            runtime.process = proc
            self._append_log_locked(runtime, "system", "$ " + " ".join(cmd))

        async def pump(stream: asyncio.StreamReader, source: str) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                if text:
                    out = sys.stderr if source == "stderr" else sys.stdout
                    print(f"[job {job_id}][{source}] {text}", file=out, flush=True)
                async with self._lock:
                    self._append_log_locked(runtime, source, text)

        stdout_task = asyncio.create_task(pump(proc.stdout, "stdout"))
        stderr_task = asyncio.create_task(pump(proc.stderr, "stderr"))
        await asyncio.gather(stdout_task, stderr_task)
        rc = await proc.wait()

        async with self._lock:
            runtime.process = None

        return rc

    def _parse_summary_json(self, graph_path: Path) -> str | None:
        try:
            with graph_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            repo = data.get("repository") or {}
            usage = data.get("embedding_usage") or {}
            summary = {
                "symbol_count": len(data.get("symbols") or []),
                "file_count": len(data.get("files") or []),
                "languages": repo.get("languages") or {},
                "commit_sha": repo.get("commit_sha") or repo.get("commit_hash"),
                "embedding_provider": usage.get("provider"),
                "embedding_model": usage.get("model"),
                "embedding_prompt_tokens": usage.get("prompt_tokens"),
                "embedding_total_tokens": usage.get("total_tokens"),
                "embedding_estimated_cost_usd": usage.get("estimated_cost_usd"),
            }
            return json.dumps(summary)
        except Exception as exc:  # keep worker resilient on malformed outputs
            print(f"[sutra-ui] failed to parse graph.json: {exc}", file=sys.stderr)
            return None

    def _extract_usage_from_logs(self, log_buffer: deque[dict[str, str]]) -> dict[str, Any] | None:
        tokens = None
        cost = None
        for item in reversed(log_buffer):
            line = item.get("line", "")
            if tokens is None and "Embedding tokens used:" in line:
                with contextlib.suppress(ValueError):
                    tokens = int(line.rsplit(":", 1)[1].strip())
            if cost is None and "Estimated embedding cost (USD):" in line:
                with contextlib.suppress(ValueError):
                    cost = float(line.rsplit(":", 1)[1].strip())
            if tokens is not None and cost is not None:
                break

        if tokens is None and cost is None:
            return None
        return {
            "embedding_total_tokens": tokens,
            "embedding_estimated_cost_usd": cost,
        }

    def _extract_failure_detail(self, log_buffer: deque[dict[str, str]]) -> str | None:
        for item in reversed(log_buffer):
            if item.get("source") != "stderr":
                continue
            line = (item.get("line") or "").strip()
            if not line:
                continue
            if line.startswith("File \"") or line.startswith("Traceback ") or line.startswith("^"):
                continue
            return line
        return None

    async def _recompute_queue_positions_locked(self) -> None:
        for pos, pending_id in enumerate(self._pending, start=1):
            await asyncio.to_thread(self._update_queue_position, pending_id, pos)

    def _append_log_locked(self, runtime: JobRuntime, source: str, line: str) -> None:
        payload = {"source": source, "line": line, "ts": _now_iso()}
        runtime.log_buffer.append(payload)
        for q in list(runtime.subscribers):
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait({"event": "log", "data": payload})

    async def _publish(self, runtime: JobRuntime, event: str, data: Any) -> None:
        payload = {"event": event, "data": data}
        for q in list(runtime.subscribers):
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(payload)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    repo_url TEXT NOT NULL,
                    replace INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    queue_position INTEGER,
                    started_at TEXT,
                    finished_at TEXT,
                    exit_code INTEGER,
                    clone_path TEXT,
                    output_path TEXT,
                    error TEXT,
                    error_detail TEXT,
                    summary_json TEXT,
                    embedding_prompt_tokens INTEGER,
                    embedding_total_tokens INTEGER,
                    embedding_estimated_cost_usd REAL,
                    pid INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )
            existing = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "embedding_prompt_tokens" not in existing:
                conn.execute("ALTER TABLE jobs ADD COLUMN embedding_prompt_tokens INTEGER")
            if "embedding_total_tokens" not in existing:
                conn.execute("ALTER TABLE jobs ADD COLUMN embedding_total_tokens INTEGER")
            if "embedding_estimated_cost_usd" not in existing:
                conn.execute("ALTER TABLE jobs ADD COLUMN embedding_estimated_cost_usd REAL")
            if "error_detail" not in existing:
                conn.execute("ALTER TABLE jobs ADD COLUMN error_detail TEXT")
            conn.commit()

    def _mark_stale_running_jobs(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status='failed', error='API restarted mid-run', finished_at=?
                WHERE status='running'
                """,
                (_now_iso(),),
            )
            conn.commit()

    def _insert_job(
        self,
        job_id: str,
        repo_url: str,
        replace: int,
        status: str,
        queue_position: int,
        clone_path: str,
        output_path: str,
        created_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs
                (id, repo_url, replace, status, queue_position, clone_path, output_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, repo_url, replace, status, queue_position, clone_path, output_path, created_at),
            )
            conn.commit()

    def _set_running(self, job_id: str, started_at: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='running', started_at=?, queue_position=NULL WHERE id=?",
                (started_at, job_id),
            )
            conn.commit()

    def _set_pid(self, job_id: str, pid: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET pid=? WHERE id=?", (pid, job_id))
            conn.commit()

    def _set_finished(
        self,
        job_id: str,
        status: str,
        finished_at: str,
        exit_code: int,
        error: str | None,
        error_detail: str | None,
        summary_json: str | None,
    ) -> None:
        prompt_tokens = None
        total_tokens = None
        estimated_cost = None
        if summary_json:
            with contextlib.suppress(json.JSONDecodeError):
                summary_obj = json.loads(summary_json)
                prompt_tokens = summary_obj.get("embedding_prompt_tokens")
                total_tokens = summary_obj.get("embedding_total_tokens")
                estimated_cost = summary_obj.get("embedding_estimated_cost_usd")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status=?, finished_at=?, exit_code=?, error=?, summary_json=?,
                    error_detail=?,
                    embedding_prompt_tokens=?, embedding_total_tokens=?, embedding_estimated_cost_usd=?,
                    pid=NULL, queue_position=NULL
                WHERE id=?
                """,
                (
                    status,
                    finished_at,
                    exit_code,
                    error,
                    summary_json,
                    error_detail,
                    prompt_tokens,
                    total_tokens,
                    estimated_cost,
                    job_id,
                ),
            )
            conn.commit()

    def _update_job_status(
        self,
        job_id: str,
        status: str,
        exit_code: int | None,
        error: str | None,
        finished_at: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status=?, exit_code=?, error=?, finished_at=?, pid=NULL, queue_position=NULL
                WHERE id=?
                """,
                (status, exit_code, error, finished_at or _now_iso(), job_id),
            )
            conn.commit()

    def _update_queue_position(self, job_id: str, position: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET queue_position=? WHERE id=? AND status='queued'",
                (position, job_id),
            )
            conn.commit()

    def _get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            return _row_to_job(row)

    def _list_jobs(self, limit: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [_row_to_job(r) for r in rows]

    def _cost_summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS jobs_count,
                    COALESCE(SUM(embedding_total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(embedding_estimated_cost_usd), 0) AS total_cost_usd
                FROM jobs
                WHERE status = 'succeeded'
                """
            ).fetchone()
            by_repo_rows = conn.execute(
                """
                SELECT
                    repo_url,
                    COUNT(*) AS jobs_count,
                    COALESCE(SUM(embedding_total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(embedding_estimated_cost_usd), 0) AS total_cost_usd
                FROM jobs
                WHERE status = 'succeeded'
                GROUP BY repo_url
                ORDER BY total_cost_usd DESC, total_tokens DESC
                """
            ).fetchall()

        return {
            "jobs_count": int(row["jobs_count"]),
            "total_tokens": int(row["total_tokens"]),
            "total_cost_usd": float(row["total_cost_usd"]),
            "by_repo": [
                {
                    "repo_url": r["repo_url"],
                    "jobs_count": int(r["jobs_count"]),
                    "total_tokens": int(r["total_tokens"]),
                    "total_cost_usd": float(r["total_cost_usd"]),
                }
                for r in by_repo_rows
            ],
        }


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    summary = None
    if row["summary_json"]:
        with contextlib.suppress(json.JSONDecodeError):
            summary = json.loads(row["summary_json"])

    return {
        "id": row["id"],
        "repo_url": row["repo_url"],
        "clone_path": row["clone_path"],
        "output_path": row["output_path"],
        "repoUrl": row["repo_url"],
        "replace": bool(row["replace"]),
        "status": row["status"],
        "queuePosition": row["queue_position"],
        "startedAt": row["started_at"],
        "finishedAt": row["finished_at"],
        "exitCode": row["exit_code"],
        "clonePath": row["clone_path"],
        "outputPath": row["output_path"],
        "error": row["error"],
        "errorDetail": row["error_detail"],
        "summary": summary,
        "embeddingPromptTokens": row["embedding_prompt_tokens"],
        "embeddingTotalTokens": row["embedding_total_tokens"],
        "embeddingEstimatedCostUsd": row["embedding_estimated_cost_usd"],
        "pid": row["pid"],
        "createdAt": row["created_at"],
    }


def _repo_slug(repo_url: str) -> str:
    raw = repo_url.rstrip("/").split("/")[-1]
    raw = raw.removesuffix(".git")
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw)
    return clean or "repo"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
