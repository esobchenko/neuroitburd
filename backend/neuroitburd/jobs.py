"""In-process FIFO job queue with a single worker thread that owns the model.

One ``JobManager`` per process. The worker thread loads the stylizer (so all CUDA work
happens on one thread) and then runs jobs strictly one at a time, in submission order.
Job records live in memory only and are lost on restart.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from PIL import Image

from neuroitburd.config import Settings
from neuroitburd.errors import ModelLoadError, NeuroitburdError, QueueFullError
from neuroitburd.image_utils import load_image, save_image
from neuroitburd.params import GenerationParams
from neuroitburd.pipeline import Stylizer, create_stylizer

logger = logging.getLogger(__name__)

StylizerFactory = Callable[[Settings], Stylizer]

UNEXPECTED_JOB_ERROR = "Generation failed due to an internal error; see the server log."
UNEXPECTED_LOAD_ERROR = "The model failed to load due to an internal error; see the server log."
SHUTDOWN_ERROR = "The server shut down before this job ran."


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelStatus(StrEnum):
    NOT_STARTED = "not_started"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


TERMINAL_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED})


@dataclass
class Job:
    """One generation request. ``JobManager.get`` returns copies, never the live record."""

    id: str
    params: GenerationParams
    input_path: Path
    output_path: Path
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0  # 0.0 to 1.0
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobManager:
    """Owns the job table, the FIFO queue and the worker thread."""

    def __init__(self, settings: Settings, stylizer_factory: StylizerFactory | None = None):
        self._settings = settings
        self._factory = stylizer_factory or create_stylizer
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()  # None = stop sentinel
        self._changed = threading.Condition()  # guards all mutable state below
        self._queued_count = 0
        self._model_status = ModelStatus.NOT_STARTED
        self._model_error: str | None = None
        self._model_info: dict[str, Any] = {}
        self._stopping = False
        self._thread: threading.Thread | None = None

    # --- lifecycle --------------------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread; it loads the model, then processes jobs."""
        with self._changed:
            if self._thread is not None:
                raise RuntimeError("JobManager already started")
            self._model_status = ModelStatus.LOADING
            self._thread = threading.Thread(target=self._worker, name="neuroitburd-worker")
            self._thread.daemon = True  # never block interpreter exit on a stuck model load
            self._thread.start()

    def stop(self, timeout: float | None = 30.0) -> None:
        """Let the current job finish, fail the queued ones, and stop the worker."""
        with self._changed:
            if self._stopping:
                return
            self._stopping = True
            self._changed.notify_all()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.warning("Worker did not stop within %ss", timeout)

    # --- public API -------------------------------------------------------------------

    def submit(self, image: Image.Image, params: GenerationParams) -> Job:
        """Save the input image and queue a job. Returns a snapshot of the new job.

        Raises ``ModelLoadError`` if the model failed to load and ``QueueFullError`` if
        ``max_queue_size`` jobs are already waiting.
        """
        with self._changed:
            self._check_accepting()
        job_id = uuid.uuid4().hex
        params = params.with_resolved_seed()
        input_path = save_image(image, self._settings.output_dir / "uploads" / f"{job_id}.png")
        job = Job(
            id=job_id,
            params=params,
            input_path=input_path,
            output_path=self._settings.output_dir / f"{job_id}.png",
            created_at=_now(),
        )
        with self._changed:
            try:
                self._check_accepting()  # state may have changed while the file was saved
            except NeuroitburdError:
                input_path.unlink(missing_ok=True)
                raise
            self._jobs[job_id] = job
            self._queued_count += 1
            self._queue.put(job_id)
            self._changed.notify_all()
            snapshot = replace(job)
        logger.info(
            "Job %s queued: %dx%d strength=%s steps=%d guidance=%s seed=%s",
            job_id,
            image.width,
            image.height,
            params.strength,
            params.steps,
            params.guidance,
            params.seed,
        )
        return snapshot

    def get(self, job_id: str) -> Job | None:
        """Snapshot of a job, or ``None`` if the id is unknown."""
        with self._changed:
            job = self._jobs.get(job_id)
            return replace(job) if job is not None else None

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        """Block until the job is completed or failed. Raises ``TimeoutError``/``KeyError``."""
        with self._changed:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            done = self._changed.wait_for(
                lambda: self._jobs[job_id].status in TERMINAL_STATUSES, timeout
            )
            if not done:
                raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")
            return replace(self._jobs[job_id])

    def wait_for_model(self, timeout: float | None = None) -> ModelStatus:
        """Block until the model is ready or failed to load; return the status."""
        with self._changed:
            self._changed.wait_for(
                lambda: self._model_status in {ModelStatus.READY, ModelStatus.ERROR}, timeout
            )
            return self._model_status

    @property
    def model_status(self) -> ModelStatus:
        """Current model state, for the health endpoint."""
        with self._changed:
            return self._model_status

    @property
    def model_error(self) -> str | None:
        """User-safe load error message, if the model failed to load."""
        with self._changed:
            return self._model_error

    @property
    def model_info(self) -> dict[str, Any]:
        """``Stylizer.describe()`` of the loaded model; empty until it is ready."""
        with self._changed:
            return dict(self._model_info)

    @property
    def queue_size(self) -> int:
        """Number of jobs waiting (not counting the one being processed)."""
        with self._changed:
            return self._queued_count

    # --- worker thread ----------------------------------------------------------------

    def _check_accepting(self) -> None:
        """Raise if new jobs cannot be queued. Caller holds the lock."""
        if self._stopping:
            raise RuntimeError("JobManager is stopping")
        if self._model_status is ModelStatus.ERROR:
            raise ModelLoadError(self._model_error or UNEXPECTED_LOAD_ERROR)
        if self._queued_count >= self._settings.max_queue_size:
            raise QueueFullError(
                f"The queue is full ({self._settings.max_queue_size} jobs waiting); "
                "try again in a minute."
            )

    def _worker(self) -> None:
        stylizer = self._load_stylizer()
        while True:
            job_id = self._queue.get()
            if job_id is None:
                break
            with self._changed:
                self._queued_count -= 1
                stopping = self._stopping
            if stopping:
                self._finish(job_id, error=SHUTDOWN_ERROR)
            elif stylizer is None:
                self._finish(job_id, error=self._model_error or UNEXPECTED_LOAD_ERROR)
            else:
                self._run(job_id, stylizer)
        self._fail_remaining()
        logger.info("Worker stopped")

    def _load_stylizer(self) -> Stylizer | None:
        try:
            stylizer = self._factory(self._settings)
            info = stylizer.describe()
        except NeuroitburdError as exc:
            logger.error("Model failed to load: %s", exc)
            self._set_model_error(str(exc))
            return None
        except Exception:
            logger.exception("Model failed to load")
            self._set_model_error(UNEXPECTED_LOAD_ERROR)
            return None
        with self._changed:
            self._model_status = ModelStatus.READY
            self._model_info = info
            self._changed.notify_all()
        logger.info("Model ready: %s", info)
        return stylizer

    def _set_model_error(self, message: str) -> None:
        with self._changed:
            self._model_status = ModelStatus.ERROR
            self._model_error = message
            self._changed.notify_all()

    def _run(self, job_id: str, stylizer: Stylizer) -> None:
        with self._changed:
            job = self._jobs[job_id]
            job.status = JobStatus.PROCESSING
            job.started_at = _now()
            params, input_path, output_path = job.params, job.input_path, job.output_path
            self._changed.notify_all()
        logger.info("Job %s processing", job_id)

        def on_progress(step: int, total: int) -> None:
            with self._changed:
                job.progress = min(step / total, 1.0) if total else 0.0
                self._changed.notify_all()

        try:
            image = load_image(input_path)
            result = stylizer.generate(image, params, on_progress=on_progress)
            save_image(result, output_path)
        except NeuroitburdError as exc:
            logger.warning("Job %s failed: %s", job_id, exc)
            self._finish(job_id, error=str(exc))
            return
        except Exception:
            logger.exception("Job %s failed unexpectedly", job_id)
            self._finish(job_id, error=UNEXPECTED_JOB_ERROR)
            return
        self._finish(job_id)

    def _finish(self, job_id: str, error: str | None = None) -> None:
        with self._changed:
            job = self._jobs[job_id]
            job.status = JobStatus.FAILED if error else JobStatus.COMPLETED
            job.error = error
            if not error:
                job.progress = 1.0
            job.finished_at = _now()
            started = job.started_at
            self._changed.notify_all()
        if not error:
            seconds = (job.finished_at - started).total_seconds() if started else 0.0
            logger.info("Job %s completed in %.1fs: %s", job_id, seconds, job.output_path)

    def _fail_remaining(self) -> None:
        """After the stop sentinel: fail jobs still in the queue (unlikely, but possible)."""
        while True:
            try:
                job_id = self._queue.get_nowait()
            except queue.Empty:
                return
            if job_id is not None:
                with self._changed:
                    self._queued_count -= 1
                self._finish(job_id, error=SHUTDOWN_ERROR)


def _now() -> datetime:
    return datetime.now(UTC)
