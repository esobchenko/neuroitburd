"""SDXLStylizer wiring tests: real torch (CPU only), fake ``diffusers`` module, no downloads."""

import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from neuroitburd.config import Settings
from neuroitburd.errors import (
    DeviceUnavailableError,
    GenerationError,
    GpuOutOfMemoryError,
    LoraLoadError,
    LoraNotFoundError,
    ModelLoadError,
)
from neuroitburd.image_utils import fit_size
from neuroitburd.params import GenerationParams
from neuroitburd.pipeline import LORA_ADAPTER_NAME, SDXLStylizer, resolve_device

torch = pytest.importorskip("torch")


@dataclass
class Recorder:
    """Shared state the fake diffusers classes write into, inspected by tests."""

    from_pretrained: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    vae_loads: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    placed_on: str | None = None
    offloaded: bool = False
    lora_load: tuple[Any, ...] | None = None
    adapters: tuple[Any, ...] | None = None
    load_error: Exception | None = None
    no_fp16_variant: bool = False
    lora_error: Exception | None = None
    to_error: Exception | None = None
    call_error: Exception | None = None


@pytest.fixture
def rec(monkeypatch: pytest.MonkeyPatch) -> Iterator[Recorder]:
    r = Recorder()

    class FakeVAE:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeVAE":
            r.vae_loads.append((model_id, kwargs))
            return cls()

    class FakePipe:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakePipe":
            r.from_pretrained.append((model_id, kwargs))
            if r.load_error is not None:
                raise r.load_error
            if r.no_fp16_variant and kwargs.get("variant") == "fp16":
                raise ValueError("You are trying to load model files of the `variant=fp16`")
            return cls()

        def set_progress_bar_config(self, **kwargs: Any) -> None:
            pass

        def load_lora_weights(self, *args: Any, **kwargs: Any) -> None:
            if r.lora_error is not None:
                raise r.lora_error
            r.lora_load = (*args, kwargs)

        def set_adapters(self, *args: Any, **kwargs: Any) -> None:
            r.adapters = (*args, kwargs)

        def to(self, device: str) -> "FakePipe":
            if r.to_error is not None:
                raise r.to_error
            r.placed_on = device
            return self

        def enable_model_cpu_offload(self) -> None:
            r.offloaded = True

        def __call__(self, **kwargs: Any) -> Any:
            r.calls.append(kwargs)
            if r.call_error is not None:
                raise r.call_error
            total = int(kwargs["num_inference_steps"] * kwargs["strength"])
            for step in range(total):
                kwargs["callback_on_step_end"](self, step, 0, {})
            return types.SimpleNamespace(images=[Image.new("RGB", kwargs["image"].size, "gray")])

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.AutoencoderKL = FakeVAE  # type: ignore[attr-defined]
    fake_diffusers.StableDiffusionXLImg2ImgPipeline = FakePipe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    yield r


def set_cuda(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)


def settings(**overrides: Any) -> Settings:
    return replace(Settings(), **overrides)


# --- resolve_device -------------------------------------------------------------


@pytest.mark.parametrize(
    ("setting", "cuda", "expected"),
    [("auto", True, "cuda"), ("auto", False, "cpu"), ("cuda", True, "cuda"), ("cpu", True, "cpu")],
)
def test_resolve_device(
    monkeypatch: pytest.MonkeyPatch, setting: str, cuda: bool, expected: str
) -> None:
    set_cuda(monkeypatch, cuda)
    assert resolve_device(setting) == expected


def test_explicit_cuda_without_cuda_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    set_cuda(monkeypatch, False)
    with pytest.raises(DeviceUnavailableError, match="CUDA was requested"):
        resolve_device("cuda")


# --- load -------------------------------------------------------------------------


def test_load_cuda_uses_fp16_vae_and_full_gpu(
    monkeypatch: pytest.MonkeyPatch, rec: Recorder
) -> None:
    set_cuda(monkeypatch, True)
    stylizer = SDXLStylizer.load(settings())
    model_id, kwargs = rec.from_pretrained[0]
    assert model_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert kwargs["torch_dtype"] is torch.float16
    assert kwargs["variant"] == "fp16"
    assert rec.vae_loads[0][0] == "madebyollin/sdxl-vae-fp16-fix"
    assert rec.vae_loads[0][1]["torch_dtype"] is torch.float16
    assert rec.placed_on == "cuda" and not rec.offloaded
    assert stylizer.describe()["dtype"] == "float16"


def test_load_cpu_uses_fp32_without_variant(monkeypatch: pytest.MonkeyPatch, rec: Recorder) -> None:
    set_cuda(monkeypatch, False)
    SDXLStylizer.load(settings(device="cpu", vae_id=""))
    _, kwargs = rec.from_pretrained[0]
    assert kwargs["torch_dtype"] is torch.float32
    assert "variant" not in kwargs and "vae" not in kwargs
    assert rec.vae_loads == []
    assert rec.placed_on == "cpu"


def test_load_falls_back_when_no_fp16_variant(
    monkeypatch: pytest.MonkeyPatch, rec: Recorder
) -> None:
    set_cuda(monkeypatch, True)
    rec.no_fp16_variant = True
    SDXLStylizer.load(settings())
    assert [kw.get("variant") for _, kw in rec.from_pretrained] == ["fp16", None]


def test_model_offload_does_not_move_to_cuda(
    monkeypatch: pytest.MonkeyPatch, rec: Recorder
) -> None:
    set_cuda(monkeypatch, True)
    SDXLStylizer.load(settings(offload="model"))
    assert rec.offloaded and rec.placed_on is None


def test_model_load_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch, rec: Recorder) -> None:
    set_cuda(monkeypatch, True)
    rec.load_error = OSError("repo not found")
    with pytest.raises(ModelLoadError, match="Could not load model"):
        SDXLStylizer.load(settings())


def test_oom_while_placing_model(monkeypatch: pytest.MonkeyPatch, rec: Recorder) -> None:
    set_cuda(monkeypatch, True)
    rec.to_error = torch.cuda.OutOfMemoryError("CUDA out of memory")
    with pytest.raises(GpuOutOfMemoryError, match="NEUROITBURD_OFFLOAD=model"):
        SDXLStylizer.load(settings())


def test_missing_lora_fails_before_model_load(
    monkeypatch: pytest.MonkeyPatch, rec: Recorder, tmp_path: Path
) -> None:
    set_cuda(monkeypatch, True)
    with pytest.raises(LoraNotFoundError, match="LoRA file not found"):
        SDXLStylizer.load(settings(lora_path=tmp_path / "missing.safetensors"))
    assert rec.from_pretrained == []


def test_lora_loaded_with_weight(
    monkeypatch: pytest.MonkeyPatch, rec: Recorder, tmp_path: Path
) -> None:
    set_cuda(monkeypatch, True)
    lora = tmp_path / "roitburd-v1.safetensors"
    lora.write_bytes(b"")
    stylizer = SDXLStylizer.load(settings(lora_path=lora, lora_weight=0.6))
    assert rec.lora_load == (
        str(tmp_path),
        {"weight_name": "roitburd-v1.safetensors", "adapter_name": LORA_ADAPTER_NAME},
    )
    assert rec.adapters == ([LORA_ADAPTER_NAME], {"adapter_weights": [0.6]})
    assert stylizer.describe()["lora"] == "roitburd-v1.safetensors"


def test_bad_lora_is_wrapped(
    monkeypatch: pytest.MonkeyPatch, rec: Recorder, tmp_path: Path
) -> None:
    set_cuda(monkeypatch, True)
    lora = tmp_path / "bad.safetensors"
    lora.write_bytes(b"")
    rec.lora_error = RuntimeError("size mismatch")
    with pytest.raises(LoraLoadError, match="SDXL 1.0 LoRA"):
        SDXLStylizer.load(settings(lora_path=lora))


# --- generate ---------------------------------------------------------------------


@pytest.fixture
def stylizer(monkeypatch: pytest.MonkeyPatch, rec: Recorder) -> SDXLStylizer:
    set_cuda(monkeypatch, True)
    return SDXLStylizer.load(settings(default_prompt="DEFAULT", default_negative_prompt="NEG"))


def test_generate_passes_params(stylizer: SDXLStylizer, rec: Recorder) -> None:
    progress: list[tuple[int, int]] = []
    params = GenerationParams.create(strength=0.5, steps=20, guidance=6.5, seed=42)
    out = stylizer.generate(
        Image.new("RGB", (400, 300)), params, on_progress=lambda s, t: progress.append((s, t))
    )
    call = rec.calls[0]
    assert call["prompt"] == "DEFAULT" and call["negative_prompt"] == "NEG"
    assert (call["strength"], call["num_inference_steps"], call["guidance_scale"]) == (0.5, 20, 6.5)
    assert call["image"].size == fit_size(400, 300)
    assert call["generator"].initial_seed() == 42
    assert progress == [(i, 10) for i in range(1, 11)]
    assert out.size == fit_size(400, 300)


def test_generate_custom_prompt_and_no_seed(stylizer: SDXLStylizer, rec: Recorder) -> None:
    stylizer.generate(Image.new("RGB", (256, 256)), GenerationParams.create(prompt="a cat"))
    assert rec.calls[0]["prompt"] == "a cat"
    assert rec.calls[0]["generator"] is None


def test_generate_oom(stylizer: SDXLStylizer, rec: Recorder) -> None:
    rec.call_error = torch.cuda.OutOfMemoryError("CUDA out of memory")
    with pytest.raises(GpuOutOfMemoryError, match="ran out of memory"):
        stylizer.generate(Image.new("RGB", (256, 256)), GenerationParams.create())


def test_generate_other_failure(stylizer: SDXLStylizer, rec: Recorder) -> None:
    rec.call_error = RuntimeError("boom")
    with pytest.raises(GenerationError, match="RuntimeError"):
        stylizer.generate(Image.new("RGB", (256, 256)), GenerationParams.create())
