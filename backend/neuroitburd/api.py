"""HTTP layer: health, job submission, job status and result images.

Run as a single process (the model is loaded once, in the job worker thread)::

    uvicorn neuroitburd.api:app --host 127.0.0.1 --port 8000

Never use ``--workers N`` (each worker would load its own copy of the model) or
``--reload`` with the real model.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response

from neuroitburd.config import Settings, setup_logging
from neuroitburd.errors import (
    DeviceUnavailableError,
    InvalidImageError,
    InvalidParamsError,
    LoraLoadError,
    LoraNotFoundError,
    ModelLoadError,
    NeuroitburdError,
    QueueFullError,
    UnsupportedImageError,
)
from neuroitburd.image_utils import load_image
from neuroitburd.jobs import Job, JobManager, JobStatus, ModelStatus, StylizerFactory
from neuroitburd.params import GenerationParams
from neuroitburd.pipeline import gpu_info

logger = logging.getLogger(__name__)

GENERATE_PATH = "/api/generate"
MULTIPART_OVERHEAD = 64 * 1024  # form fields and boundaries on top of the image itself

# Checked in order, so subclasses come before their base classes.
_ERROR_STATUS: tuple[tuple[type[NeuroitburdError], int], ...] = (
    (UnsupportedImageError, 415),
    (InvalidImageError, 422),
    (InvalidParamsError, 422),
    (QueueFullError, 503),
    (ModelLoadError, 503),
    (DeviceUnavailableError, 503),
    (LoraNotFoundError, 503),
    (LoraLoadError, 503),
)


def create_app(
    settings: Settings | None = None, stylizer_factory: StylizerFactory | None = None
) -> FastAPI:
    """Build the FastAPI app. The job worker (and model load) starts with the app lifespan."""
    settings = settings or Settings.from_env()
    max_upload_bytes = int(settings.max_upload_mb * 2**20)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        _warn_if_multiple_workers()
        manager = JobManager(settings, stylizer_factory)
        manager.start()
        app.state.jobs = manager
        logger.info(
            "API ready; model loading in the background (output_dir=%s)", settings.output_dir
        )
        try:
            yield
        finally:
            manager.stop()

    app = FastAPI(title="Neuroitburd", version="0.1.0", lifespan=lifespan)

    def jobs(request: Request) -> JobManager:
        manager: JobManager = request.app.state.jobs
        return manager

    @app.middleware("http")
    async def reject_large_uploads(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Reject before the multipart body is read and spooled to disk.
        if request.method == "POST" and request.url.path == GENERATE_PATH:
            length = request.headers.get("content-length", "")
            if length.isdigit() and int(length) > max_upload_bytes + MULTIPART_OVERHEAD:
                return _too_large(settings)
        return await call_next(request)

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        """Service and model state, basic GPU information and queue length."""
        manager = jobs(request)
        status = manager.model_status
        info = manager.model_info
        return {
            "status": {ModelStatus.READY: "ok", ModelStatus.ERROR: "error"}.get(status, "starting"),
            "model_status": status.value,
            "model_error": manager.model_error,
            "model_id": info.get("model_id", settings.model_id),
            "device": info.get("device"),
            "lora": (
                {"name": info["lora"], "weight": info.get("lora_weight")}
                if info.get("lora")
                else None
            ),
            "gpu": gpu_info(),
            "queue_size": manager.queue_size,
        }

    @app.post(GENERATE_PATH, status_code=202)
    def generate(
        request: Request,
        image: Annotated[UploadFile, File(description="JPEG, PNG or WebP image")],
        strength: Annotated[str | None, Form()] = None,
        steps: Annotated[str | None, Form()] = None,
        guidance: Annotated[str | None, Form()] = None,
        seed: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        negative_prompt: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        """Validate the upload and settings, queue a job, and return its id and status."""
        fields = {
            "strength": strength,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
        # Empty form fields mean "use the default" (browsers send "" for blank inputs).
        params = GenerationParams.create(**{k: v for k, v in fields.items() if v not in (None, "")})
        data = image.file.read(max_upload_bytes + 1)
        if len(data) > max_upload_bytes:
            raise HTTPException(413, _too_large_message(settings))
        if not data:
            raise InvalidImageError("The uploaded image is empty")
        source = load_image(data)
        try:
            job = jobs(request).submit(source, params)
        except RuntimeError as exc:  # manager stopping
            raise HTTPException(503, "The server is shutting down; try again shortly") from exc
        return _job_body(job)

    @app.get("/api/jobs/{job_id}")
    def get_job(request: Request, job_id: str) -> dict[str, Any]:
        """Status, progress, error, output URL and parameters of a job."""
        job = jobs(request).get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return _job_body(job)

    @app.get("/api/images/{image_id}")
    def get_image(image_id: str) -> FileResponse:
        """Serve a generated image. Only UUID ids are accepted, so no path tricks."""
        try:
            name = uuid.UUID(image_id).hex
        except ValueError:
            raise HTTPException(404, "Image not found") from None
        path = settings.output_dir / f"{name}.png"
        if not path.is_file():
            raise HTTPException(404, "Image not found")
        return FileResponse(
            path,
            media_type="image/png",
            filename=f"neuroitburd-{name}.png",
            content_disposition_type="inline",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.exception_handler(NeuroitburdError)
    async def handle_app_error(_request: Request, exc: NeuroitburdError) -> JSONResponse:
        status = next((code for cls, code in _ERROR_STATUS if isinstance(exc, cls)), 500)
        if status >= 500 and status != 503:
            logger.error("Request failed: %s", exc)
        return JSONResponse({"detail": str(exc)}, status_code=status)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse({"detail": _format_request_errors(exc)}, status_code=422)

    @app.exception_handler(Exception)
    async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled error", exc_info=exc)
        return JSONResponse({"detail": "Internal server error"}, status_code=500)

    return app


def _job_body(job: Job) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "status": job.status.value,
        "progress": round(job.progress, 3),
        "error": job.error,
        "output_url": f"/api/images/{job.id}" if job.status is JobStatus.COMPLETED else None,
        "params": job.params.model_dump(),
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at),
    }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _too_large_message(settings: Settings) -> str:
    return f"Image is too large; the maximum upload size is {settings.max_upload_mb:g} MB"


def _too_large(settings: Settings) -> JSONResponse:
    return JSONResponse({"detail": _too_large_message(settings)}, status_code=413)


def _format_request_errors(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", ()) if p not in ("body", "query", "path")]
        msg = str(err.get("msg", "invalid value"))
        parts.append(f"{'.'.join(loc)}: {msg}" if loc else msg)
    return "Invalid request: " + "; ".join(parts)


def _warn_if_multiple_workers() -> None:
    workers = os.environ.get("WEB_CONCURRENCY", "1")
    if workers.isdigit() and int(workers) > 1:
        logger.warning(
            "WEB_CONCURRENCY=%s: every worker process loads its own model and GPU jobs are not "
            "serialized across processes. Run a single worker.",
            workers,
        )


app = create_app()
