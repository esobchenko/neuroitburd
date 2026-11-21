import sys
import types
from dataclasses import replace

import pytest
from PIL import Image

from neuroitburd.config import Settings
from neuroitburd.image_utils import fit_size
from neuroitburd.params import GenerationParams
from neuroitburd.pipeline import FakeStylizer, create_stylizer, gpu_info


def gradient(w: int = 320, h: int = 240) -> Image.Image:
    return Image.linear_gradient("L").resize((w, h)).convert("RGB")


def test_output_is_fitted_and_rgb() -> None:
    out = FakeStylizer().generate(gradient(), GenerationParams.create(seed=1))
    assert out.mode == "RGB"
    assert out.size == fit_size(320, 240)


def test_progress_reports_effective_steps() -> None:
    calls: list[tuple[int, int]] = []
    params = GenerationParams.create(steps=10, strength=0.5)
    FakeStylizer().generate(gradient(), params, on_progress=lambda s, t: calls.append((s, t)))
    assert calls == [(i, 5) for i in range(1, 6)]


def test_deterministic_and_seed_sensitive() -> None:
    fake = FakeStylizer()
    colorful = Image.merge("RGB", [gradient().getchannel(0).rotate(a) for a in (0, 90, 180)])
    a = fake.generate(colorful, GenerationParams.create(seed=3))
    b = fake.generate(colorful, GenerationParams.create(seed=3))
    c = fake.generate(colorful, GenerationParams.create(seed=4))
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()


def test_create_stylizer_fake_mode() -> None:
    stylizer = create_stylizer(replace(Settings(), fake_pipeline=True))
    assert isinstance(stylizer, FakeStylizer)
    assert stylizer.describe()["backend"] == "fake"


def test_gpu_info_none_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert gpu_info() is None


def _fake_torch(available: bool, fail: bool = False) -> types.SimpleNamespace:
    def mem_get_info(_idx: int) -> tuple[int, int]:
        if fail:
            raise RuntimeError("driver error")
        return 3 * 2**30, 12 * 2**30

    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        mem_get_info=mem_get_info,
        get_device_name=lambda _idx: "NVIDIA GeForce RTX 4070 SUPER",
    )
    return types.SimpleNamespace(cuda=cuda)


def test_gpu_info_reads_loaded_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True))
    assert gpu_info() == {
        "name": "NVIDIA GeForce RTX 4070 SUPER",
        "total_mb": 12288,
        "free_mb": 3072,
    }


def test_gpu_info_none_without_cuda_or_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
    assert gpu_info() is None
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True, fail=True))
    assert gpu_info() is None
