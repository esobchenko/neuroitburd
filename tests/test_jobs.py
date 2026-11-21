import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from neuroitburd.config import Settings
from neuroitburd.errors import GpuOutOfMemoryError, ModelLoadError, QueueFullError
from neuroitburd.image_utils import fit_size
from neuroitburd.jobs import (
    SHUTDOWN_ERROR,
    UNEXPECTED_JOB_ERROR,
    UNEXPECTED_LOAD_ERROR,
    JobManager,
    JobStatus,
    ModelStatus,
)
from neuroitburd.params import GenerationParams
from neuroitburd.pipeline import FakeStylizer, ProgressCallback, Stylizer

TIMEOUT = 10.0  # generous upper bound; tests synchronize with events, not sleeps


def image(w: int = 320, h: int = 240) -> Image.Image:
    return Image.linear_gradient("L").resize((w, h)).convert("RGB")


def params(**values: Any) -> GenerationParams:
    values.setdefault("steps", 4)
    values.setdefault("strength", 0.5)
    return GenerationParams.create(**values)


class ScriptedStylizer(FakeStylizer):
    """FakeStylizer that records order and concurrency and can be held or made to fail."""

    def __init__(self) -> None:
        super().__init__()
        self.order: list[int | None] = []
        self.active = 0
        self.max_active = 0
        self.gate: threading.Event | None = None  # if set, generate() waits on it
        self.entered = threading.Event()
        self.fail_seeds: dict[int, Exception] = {}
        self.on_step: Callable[[], None] | None = None
        self._lock = threading.Lock()

    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        on_progress: ProgressCallback | None = None,
    ) -> Image.Image:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.order.append(params.seed)
        self.entered.set()
        try:
            if self.gate is not None:
                assert self.gate.wait(TIMEOUT)
            if params.seed in self.fail_seeds:
                raise self.fail_seeds[params.seed]

            def progress(step: int, total: int) -> None:
                if on_progress is not None:
                    on_progress(step, total)
                if self.on_step is not None:
                    self.on_step()

            return super().generate(image, params, progress)
        finally:
            with self._lock:
                self.active -= 1


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return replace(Settings(), output_dir=tmp_path / "out", max_queue_size=20)


@pytest.fixture
def fake() -> ScriptedStylizer:
    return ScriptedStylizer()


@pytest.fixture
def manager(settings: Settings, fake: ScriptedStylizer) -> Iterator[JobManager]:
    mgr = JobManager(settings, stylizer_factory=lambda _s: fake)
    yield mgr
    if fake.gate is not None:
        fake.gate.set()
    mgr.stop(timeout=TIMEOUT)


# --- happy path ---------------------------------------------------------------------


def test_job_lifecycle(manager: JobManager, fake: ScriptedStylizer, settings: Settings) -> None:
    fake.gate = threading.Event()
    job = manager.submit(image(), params(seed=5))

    assert job.status is JobStatus.QUEUED
    assert job.input_path == settings.output_dir / "uploads" / f"{job.id}.png"
    assert job.input_path.is_file()
    assert job.output_path == settings.output_dir / f"{job.id}.png"
    assert job.created_at is not None and job.started_at is None

    manager.start()
    assert fake.entered.wait(TIMEOUT)
    running = manager.get(job.id)
    assert running is not None
    assert running.status is JobStatus.PROCESSING and running.started_at is not None

    fake.gate.set()
    done = manager.wait(job.id, TIMEOUT)
    assert done.status is JobStatus.COMPLETED
    assert done.progress == 1.0 and done.error is None
    assert done.finished_at is not None and done.finished_at >= done.started_at
    with Image.open(done.output_path) as out:
        assert out.size == fit_size(320, 240)


def test_snapshots_are_copies(manager: JobManager) -> None:
    job = manager.submit(image(), params())
    job.status = JobStatus.FAILED
    fresh = manager.get(job.id)
    assert fresh is not None and fresh.status is JobStatus.QUEUED


def test_unknown_job(manager: JobManager) -> None:
    assert manager.get("does-not-exist") is None
    with pytest.raises(KeyError):
        manager.wait("does-not-exist")


def test_random_seed_resolved_at_submit(manager: JobManager) -> None:
    job = manager.submit(image(), params(seed=None))
    assert job.params.seed is not None


def test_model_info_and_status(manager: JobManager) -> None:
    assert manager.model_status is ModelStatus.NOT_STARTED
    manager.start()
    assert manager.wait_for_model(TIMEOUT) is ModelStatus.READY
    assert manager.model_info["backend"] == "fake"
    assert manager.model_error is None


def test_model_loaded_on_worker_thread(settings: Settings) -> None:
    threads: list[str] = []

    def factory(_s: Settings) -> Stylizer:
        threads.append(threading.current_thread().name)
        return FakeStylizer()

    mgr = JobManager(settings, stylizer_factory=factory)
    mgr.start()
    mgr.wait_for_model(TIMEOUT)
    mgr.stop(TIMEOUT)
    assert threads == ["neuroitburd-worker"]


# --- ordering and concurrency -------------------------------------------------------


def test_fifo_one_at_a_time(manager: JobManager, fake: ScriptedStylizer) -> None:
    jobs = [manager.submit(image(), params(seed=i)) for i in range(6)]
    manager.start()
    for job in jobs:
        assert manager.wait(job.id, TIMEOUT).status is JobStatus.COMPLETED
    assert fake.order == list(range(6))
    assert fake.max_active == 1


def test_submit_from_many_threads(manager: JobManager, fake: ScriptedStylizer) -> None:
    manager.start()
    barrier = threading.Barrier(8)
    ids: list[str] = []
    ids_lock = threading.Lock()

    def submit(i: int) -> None:
        barrier.wait(TIMEOUT)
        job = manager.submit(image(), params(seed=i))
        with ids_lock:
            ids.append(job.id)

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)
    for job_id in ids:
        assert manager.wait(job_id, TIMEOUT).status is JobStatus.COMPLETED
    assert len(set(ids)) == 8
    assert sorted(s for s in fake.order if s is not None) == list(range(8))
    assert fake.max_active == 1


def test_progress_increases(manager: JobManager, fake: ScriptedStylizer) -> None:
    seen: list[float] = []
    job = manager.submit(image(), params(steps=10, strength=0.5))

    def record() -> None:
        snap = manager.get(job.id)
        assert snap is not None
        seen.append(snap.progress)

    fake.on_step = record
    manager.start()
    manager.wait(job.id, TIMEOUT)
    assert seen == [0.2, 0.4, 0.6, 0.8, 1.0]


def test_queue_size_and_limit(settings: Settings, fake: ScriptedStylizer) -> None:
    fake.gate = threading.Event()
    mgr = JobManager(replace(settings, max_queue_size=2), stylizer_factory=lambda _s: fake)
    mgr.start()
    first = mgr.submit(image(), params(seed=1))
    assert fake.entered.wait(TIMEOUT)  # first job is processing, not counted as queued
    mgr.submit(image(), params(seed=2))
    mgr.submit(image(), params(seed=3))
    assert mgr.queue_size == 2

    with pytest.raises(QueueFullError, match="queue is full"):
        mgr.submit(image(), params(seed=4))
    uploads = list((settings.output_dir / "uploads").iterdir())
    assert len(uploads) == 3  # the rejected upload was not kept

    fake.gate.set()
    mgr.wait(first.id, TIMEOUT)
    mgr.stop(TIMEOUT)


# --- failures -----------------------------------------------------------------------


def test_failed_job_does_not_block_next(manager: JobManager, fake: ScriptedStylizer) -> None:
    fake.fail_seeds[1] = GpuOutOfMemoryError("GPU ran out of memory. Close other apps.")
    bad = manager.submit(image(), params(seed=1))
    good = manager.submit(image(), params(seed=2))
    manager.start()

    failed = manager.wait(bad.id, TIMEOUT)
    assert failed.status is JobStatus.FAILED
    assert failed.error == "GPU ran out of memory. Close other apps."
    assert not failed.output_path.exists()
    assert manager.wait(good.id, TIMEOUT).status is JobStatus.COMPLETED


def test_unexpected_error_message_is_generic(
    manager: JobManager, fake: ScriptedStylizer, caplog: pytest.LogCaptureFixture
) -> None:
    fake.fail_seeds[1] = RuntimeError("/internal/path/model.py blew up")
    job = manager.submit(image(), params(seed=1))
    manager.start()
    failed = manager.wait(job.id, TIMEOUT)
    assert failed.error == UNEXPECTED_JOB_ERROR
    assert "/internal/path/model.py blew up" in caplog.text  # details only in the log


def test_missing_input_file_fails_job(manager: JobManager) -> None:
    job = manager.submit(image(), params())
    job.input_path.unlink()
    manager.start()
    failed = manager.wait(job.id, TIMEOUT)
    assert failed.status is JobStatus.FAILED
    assert failed.error is not None and "not found" in failed.error


def test_model_load_failure(settings: Settings) -> None:
    release = threading.Event()

    def factory(_s: Settings) -> Stylizer:
        assert release.wait(TIMEOUT)
        raise ModelLoadError("Could not load model 'x'. Check the model ID.")

    mgr = JobManager(settings, stylizer_factory=factory)
    mgr.start()
    queued = mgr.submit(image(), params())  # accepted while the model is still loading
    assert mgr.model_status is ModelStatus.LOADING
    release.set()

    failed = mgr.wait(queued.id, TIMEOUT)
    assert failed.status is JobStatus.FAILED
    assert failed.error == "Could not load model 'x'. Check the model ID."
    assert mgr.model_status is ModelStatus.ERROR
    assert mgr.model_error == failed.error
    with pytest.raises(ModelLoadError, match="Could not load model"):
        mgr.submit(image(), params())  # new jobs fail fast
    mgr.stop(TIMEOUT)


def test_unexpected_load_failure_is_generic(settings: Settings) -> None:
    def factory(_s: Settings) -> Stylizer:
        raise OSError("/home/someone/.cache broken")

    mgr = JobManager(settings, stylizer_factory=factory)
    mgr.start()
    assert mgr.wait_for_model(TIMEOUT) is ModelStatus.ERROR
    assert mgr.model_error == UNEXPECTED_LOAD_ERROR
    mgr.stop(TIMEOUT)


# --- shutdown -----------------------------------------------------------------------


def test_stop_finishes_current_and_fails_queued(
    manager: JobManager, fake: ScriptedStylizer
) -> None:
    fake.gate = threading.Event()
    current = manager.submit(image(), params(seed=1))
    waiting = manager.submit(image(), params(seed=2))
    manager.start()
    assert fake.entered.wait(TIMEOUT)

    stopper = threading.Thread(target=manager.stop, args=(TIMEOUT,))
    stopper.start()
    with manager._changed:  # release the running job only once stop() has taken effect
        assert manager._changed.wait_for(lambda: manager._stopping, TIMEOUT)
    fake.gate.set()
    stopper.join(TIMEOUT)

    assert manager.wait(current.id, 0).status is JobStatus.COMPLETED
    shut = manager.wait(waiting.id, 0)
    assert shut.status is JobStatus.FAILED and shut.error == SHUTDOWN_ERROR
    assert manager.queue_size == 0
    with pytest.raises(RuntimeError):
        manager.submit(image(), params())


def test_start_twice_rejected(manager: JobManager) -> None:
    manager.start()
    with pytest.raises(RuntimeError):
        manager.start()
