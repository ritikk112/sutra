from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

import psycopg2
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from frontend.api.jobs import JobManager
from frontend.api.utils import ARTIFACT_ALLOWLIST, looks_like_git_url


REPO_ROOT = Path(__file__).resolve().parents[2]
SUTRA_HOME = Path.home() / ".sutra"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Sutra Frontend API", version="0.1.0")
manager: JobManager | None = None


class CreateJobRequest(BaseModel):
    repoUrl: str
    replace: bool = True


@app.on_event("startup")
async def startup() -> None:
    pg_url = os.environ.get("SUTRA_PG_URL", "").strip()
    if not pg_url:
        raise RuntimeError("SUTRA_PG_URL is required at startup.")

    SUTRA_HOME.mkdir(parents=True, exist_ok=True)

    _check_postgres(pg_url)
    _check_pipeline_importable()

    needs_openai = _config_requires_openai(REPO_ROOT / "config" / "sutra.yaml")
    openai_key_set = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if needs_openai and not openai_key_set:
        raise RuntimeError("OPENAI_API_KEY is required by config/sutra.yaml (embedder.provider=openai).")

    print(
        "[sutra-ui] startup config: "
        f"repo_root={REPO_ROOT} sutra_home={SUTRA_HOME} "
        f"pg_url_set={bool(pg_url)} openai_required={needs_openai} openai_key_set={openai_key_set}"
    )

    global manager
    manager = JobManager(repo_root=REPO_ROOT, pg_url=pg_url, sutra_home=SUTRA_HOME)
    await manager.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    global manager
    if manager:
        await manager.stop()


@app.get("/api/health")
async def health() -> dict[str, Any]:
    pg_url = os.environ.get("SUTRA_PG_URL", "").strip()
    pg_ok = _check_postgres(pg_url, raise_on_error=False) if pg_url else False
    openai_key_set = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    disk = shutil.disk_usage(SUTRA_HOME)
    disk_free_gb = round(disk.free / (1024 ** 3), 2)
    return {
        "pg_ok": pg_ok,
        "openai_key_set": openai_key_set,
        "disk_free_gb": disk_free_gb,
    }


@app.post("/api/jobs")
async def create_job(req: CreateJobRequest) -> dict[str, Any]:
    _require_manager()
    if not looks_like_git_url(req.repoUrl):
        raise HTTPException(status_code=400, detail="repoUrl must look like a git URL")
    return await manager.enqueue(repo_url=req.repoUrl.strip(), replace=req.replace)


@app.get("/api/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    _require_manager()
    return await manager.list_jobs(limit=50)


@app.get("/api/jobs/cost-summary")
async def jobs_cost_summary() -> dict[str, Any]:
    _require_manager()
    return await manager.cost_summary()


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    _require_manager()
    job = await manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> JSONResponse:
    _require_manager()
    code, payload = await manager.cancel(job_id)
    if code == 404:
        raise HTTPException(status_code=404, detail=payload["detail"])
    if code == 409:
        raise HTTPException(status_code=409, detail=payload["detail"])
    return JSONResponse(payload)


@app.get("/api/jobs/{job_id}/artifacts/{name}")
async def download_artifact(job_id: str, name: str) -> FileResponse:
    _require_manager()
    if name not in ARTIFACT_ALLOWLIST:
        raise HTTPException(status_code=404, detail="artifact not allowed")

    job = await manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    artifact_path = Path(job["outputPath"]) / name
    if not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path=artifact_path, filename=name)


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    _require_manager()
    job = await manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    runtime, subscriber = await manager.subscribe(job_id)

    async def gen() -> Any:
        try:
            if runtime:
                for log_line in list(runtime.log_buffer):
                    yield _sse("log", log_line)

            latest = await manager.get_job(job_id)
            if latest and latest["status"] in {"succeeded", "failed", "cancelled"}:
                yield _sse("completed", latest)
                return

            while True:
                try:
                    item = await asyncio.wait_for(subscriber.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\\n\\n"
                    continue

                yield _sse(item["event"], item["data"])
                if item["event"] == "completed":
                    return
        finally:
            await manager.unsubscribe(job_id, subscriber)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/", include_in_schema=False)
async def root_index() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return FileResponse(_write_missing_index(), media_type="text/html")
    return FileResponse(index)


@app.get("/{full_path:path}", include_in_schema=False)
async def static_or_spa(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404)

    candidate = STATIC_DIR / full_path
    if candidate.is_file():
        return FileResponse(candidate)

    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return FileResponse(_write_missing_index(), media_type="text/html")


def _require_manager() -> None:
    if manager is None:
        raise HTTPException(status_code=503, detail="job manager is not initialized")


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\\ndata: {json.dumps(data)}\\n\\n"


def _check_pipeline_importable() -> None:
    import pipelines.full_index  # noqa: F401


def _config_requires_openai(config_path: Path) -> bool:
    if not config_path.exists():
        return False

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    embedder = cfg.get("embedder") or {}
    return str(embedder.get("provider", "fixture")).lower() == "openai"


def _check_postgres(pg_url: str, raise_on_error: bool = True) -> bool:
    try:
        with psycopg2.connect(pg_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        if raise_on_error:
            raise
        return False


def _write_missing_index() -> Path:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    fallback = STATIC_DIR / "index.html"
    fallback.write_text(
        "<html><body><h3>Frontend not built.</h3>"
        "<p>Run: cd frontend/web && npm install && npm run build</p></body></html>",
        encoding="utf-8",
    )
    return fallback
