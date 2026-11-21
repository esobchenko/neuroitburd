import io
import threading
import uuid
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from neuroitburd import api
from neuroitburd.config import Settings
from neuroitburd.errors import ModelLoadError
from neuroitburd.image_utils import fit_size
from neuroitburd.jobs import JobManager, StylizerFactory
from neuroitburd.params import GenerationParams
from neuroitburd.pipeline import FakeStylizer, ProgressCallback, Stylizer

TIMEOUT = 10.0


def png_bytes(w: int = 320, h: int = 240, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.linear_gradient("L").resize((w, h)).convert("RGB").save(buf, format=fmt)
    return buf.getvalue()


class GatedStylizer(FakeStylizer):
    """Blocks inside generate() until ``gate`` is set; ``entered`` fires when it starts."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.entered = threading.Event()

    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        on_progress: ProgressCallback | None = None,
    ) -> Image.Image:
        self.entered.set()
        assert self.gate.wait(TIMEOUT)
        return super().generate(image, params, on_progress)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return replace(Settings(), output_dir=tmp_path / "out")


@pytest.fixture(autouse=True)
def no_gpu_query(monkeypatch: pytest.MonkeyPatch) -> None:
    # Other tests may import torch; keep health deterministic on machines with a GPU.
    monkeypatch.setattr(api, "gpu_info", lambda: None)


def make_client(
    settings: Settings, factory: StylizerFactory | None = None, **kwargs: Any
) -> TestClient:
    app = api.create_app(settings, factory or (lambda _s: FakeStylizer()))
    return TestClient(app, **kwargs)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with make_client(settings) as c:
        manager(c).wait_for_model(TIMEOUT)
        yield c


def manager(client: TestClient) -> JobManager:
    jobs: JobManager = client.app.state.jobs  # type: ignore[attr-defined]
    return jobs


def post_image(
    client: TestClient, data: bytes | None = None, filename: str = "in.png", **fields: str
) -> Any:
    return client.post(
        "/api/generate",
        files={"image": (filename, png_bytes() if data is None else data, "image/png")},
        data=fields,
    )


# --- health -------------------------------------------------------------------------


def test_health_ready(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body == {
        "status": "ok",
        "model_status": "ready",
        "model_error": None,
        "model_id": "fake",
        "device": "cpu",
        "lora": None,
        "gpu": None,
        "queue_size": 0,
    }


def test_health_while_loading_and_gpu_info(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def slow_factory(_s: Settings) -> Stylizer:
        assert release.wait(TIMEOUT)
        return FakeStylizer()

    gpu = {"name": "RTX", "total_mb": 12282, "free_mb": 11000}
    monkeypatch.setattr(api, "gpu_info", lambda: gpu)
    with make_client(settings, slow_factory) as c:
        body = c.get("/api/health").json()
        assert body["status"] == "starting" and body["model_status"] == "loading"
        assert body["model_id"] == settings.model_id  # from settings until the model reports
        assert body["gpu"] == gpu
        release.set()
        manager(c).wait_for_model(TIMEOUT)
        assert c.get("/api/health").json()["status"] == "ok"


def test_health_reports_lora(settings: Settings) -> None:
    class LoraFake(FakeStylizer):
        def describe(self) -> dict[str, Any]:
            return {**super().describe(), "lora": "roitburd-v1.safetensors", "lora_weight": 0.8}

    with make_client(settings, lambda _s: LoraFake()) as c:
        manager(c).wait_for_model(TIMEOUT)
        lora = c.get("/api/health").json()["lora"]
    assert lora == {"name": "roitburd-v1.safetensors", "weight": 0.8}


def test_model_load_failure(settings: Settings) -> None:
    def broken(_s: Settings) -> Stylizer:
        raise ModelLoadError("Could not load model 'x'. Check the model ID.")

    with make_client(settings, broken) as c:
        manager(c).wait_for_model(TIMEOUT)
        body = c.get("/api/health").json()
        assert body["status"] == "error"
        assert body["model_error"] == "Could not load model 'x'. Check the model ID."
        resp = post_image(c)
    assert resp.status_code == 503
    assert resp.json() == {"detail": "Could not load model 'x'. Check the model ID."}


# --- generate, poll, fetch ------------------------------------------------------------


def test_end_to_end(client: TestClient, settings: Settings) -> None:
    resp = post_image(
        client,
        png_bytes(640, 480),
        strength="0.5",
        steps="20",
        guidance="6",
        seed="42",
        prompt="cubist portrait",
    )
    assert resp.status_code == 202
    created = resp.json()
    assert created["status"] in {"queued", "processing", "completed"}
    assert created["params"] == {
        "strength": 0.5,
        "steps": 20,
        "guidance": 6.0,
        "seed": 42,
        "prompt": "cubist portrait",
        "negative_prompt": None,
    }
    job_id = created["job_id"]

    manager(client).wait(job_id, TIMEOUT)
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "completed"
    assert status["progress"] == 1.0 and status["error"] is None
    assert status["output_url"] == f"/api/images/{job_id}"
    assert status["started_at"] is not None and status["finished_at"] is not None

    img = client.get(status["output_url"])
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"
    assert "immutable" in img.headers["cache-control"]
    with Image.open(io.BytesIO(img.content)) as out:
        assert out.size == fit_size(640, 480)
    assert (settings.output_dir / "uploads" / f"{job_id}.png").is_file()


def test_defaults_and_blank_fields(client: TestClient) -> None:
    resp = post_image(client, seed="", prompt="", strength="")
    assert resp.status_code == 202
    params = resp.json()["params"]
    defaults = GenerationParams()
    assert params["strength"] == defaults.strength
    assert params["seed"] is not None  # random seed chosen and reported
    assert params["prompt"] is None


def test_jpeg_and_webp_accepted(client: TestClient) -> None:
    for fmt in ("JPEG", "WEBP"):
        assert post_image(client, png_bytes(fmt=fmt), filename="x.bin").status_code == 202


def test_hyphenated_uuid_serves_same_image(client: TestClient) -> None:
    job_id = post_image(client).json()["job_id"]
    manager(client).wait(job_id, TIMEOUT)
    resp = client.get(f"/api/images/{uuid.UUID(job_id)}")
    assert resp.status_code == 200


def test_processing_status_and_queue_size(settings: Settings) -> None:
    fake = GatedStylizer()
    with make_client(settings, lambda _s: fake) as c:
        first = post_image(c).json()["job_id"]
        assert fake.entered.wait(TIMEOUT)
        second = post_image(c).json()["job_id"]
        assert c.get(f"/api/jobs/{first}").json()["status"] == "processing"
        queued = c.get(f"/api/jobs/{second}").json()
        assert queued["status"] == "queued" and queued["output_url"] is None
        assert c.get("/api/health").json()["queue_size"] == 1
        fake.gate.set()
        manager(c).wait(second, TIMEOUT)


def test_failed_job_reports_error(settings: Settings) -> None:
    class Failing(FakeStylizer):
        def generate(self, *args: Any, **kwargs: Any) -> Image.Image:
            raise RuntimeError("/internal/secret.py exploded")

    with make_client(settings, lambda _s: Failing()) as c:
        job_id = post_image(c).json()["job_id"]
        manager(c).wait(job_id, TIMEOUT)
        body = c.get(f"/api/jobs/{job_id}").json()
    assert body["status"] == "failed"
    assert body["output_url"] is None
    assert "internal error" in body["error"] and "secret" not in body["error"]


# --- request errors -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "fragment"),
    [
        ({"strength": "2"}, "strength"),
        ({"steps": "abc"}, "steps"),
        ({"steps": "1", "strength": "0.5"}, "steps * strength"),
        ({"seed": "-1"}, "seed"),
        ({"prompt": "x" * 501}, "prompt"),
    ],
)
def test_invalid_params_422(client: TestClient, fields: dict[str, str], fragment: str) -> None:
    resp = post_image(client, **fields)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, str) and fragment in detail


def test_missing_image_422(client: TestClient) -> None:
    resp = client.post("/api/generate", data={"strength": "0.5"})
    assert resp.status_code == 422
    assert resp.json() == {"detail": "Invalid request: image: Field required"}


def test_empty_image_422(client: TestClient) -> None:
    resp = post_image(client, b"")
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"]


@pytest.mark.parametrize(
    "data", [b"hello, not an image", png_bytes(fmt="BMP"), png_bytes(fmt="GIF")]
)
def test_unsupported_type_415(client: TestClient, data: bytes) -> None:
    resp = post_image(client, data)
    assert resp.status_code == 415
    assert "JPEG, PNG or WebP" in resp.json()["detail"]


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        (png_bytes(32, 32), "too small"),
        (png_bytes(1000, 200), "aspect ratio"),
        (png_bytes()[:200], "corrupt or truncated"),
    ],
)
def test_bad_image_422(client: TestClient, data: bytes, fragment: str) -> None:
    resp = post_image(client, data)
    assert resp.status_code == 422
    assert fragment in resp.json()["detail"]


def test_too_large_rejected_by_content_length(settings: Settings) -> None:
    small = replace(settings, max_upload_mb=0.05)  # ~52 KB, plus 64 KB multipart allowance
    with make_client(small) as c:
        resp = post_image(c, b"\0" * 300_000)
    assert resp.status_code == 413
    assert resp.json() == {"detail": "Image is too large; the maximum upload size is 0.05 MB"}


def test_too_large_rejected_while_reading(settings: Settings) -> None:
    small = replace(settings, max_upload_mb=0.05)
    with make_client(small) as c:
        resp = post_image(c, b"\0" * 80_000)  # under the header allowance, over the limit
    assert resp.status_code == 413


def test_queue_full_503(settings: Settings) -> None:
    fake = GatedStylizer()
    with make_client(replace(settings, max_queue_size=1), lambda _s: fake) as c:
        assert post_image(c).status_code == 202
        assert fake.entered.wait(TIMEOUT)
        assert post_image(c).status_code == 202
        resp = post_image(c)
        assert resp.status_code == 503
        assert "queue is full" in resp.json()["detail"]
        fake.gate.set()


def test_unknown_job_404(client: TestClient) -> None:
    resp = client.get(f"/api/jobs/{uuid.uuid4().hex}")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Job not found"}


@pytest.mark.parametrize("image_id", ["not-a-uuid", "..%2F..%2Fetc%2Fpasswd", uuid.uuid4().hex])
def test_image_404(client: TestClient, image_id: str) -> None:
    resp = client.get(f"/api/images/{image_id}")
    assert resp.status_code == 404


def test_image_id_cannot_escape_output_dir(client: TestClient, settings: Settings) -> None:
    secret = settings.output_dir.parent / "secret.png"
    secret.write_bytes(png_bytes())
    for path in ["/api/images/../secret.png", "/api/images/%2E%2E%2Fsecret"]:
        assert client.get(path).status_code == 404


def test_unexpected_error_500_is_generic(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(_data: bytes) -> Image.Image:
        raise RuntimeError("/internal/path detail")

    monkeypatch.setattr(api, "load_image", boom)
    with make_client(settings, raise_server_exceptions=False) as c:
        resp = post_image(c)
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Internal server error"}
    assert "/internal/path detail" in caplog.text


def test_multiple_workers_warning(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    with make_client(settings):
        pass
    assert "WEB_CONCURRENCY=4" in caplog.text


def test_module_level_app_exists() -> None:
    assert api.app.title == "Neuroitburd"
