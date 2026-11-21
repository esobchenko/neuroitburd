"""Inference: the ``Stylizer`` interface, the real SDXL img2img implementation and a fake one.

``torch`` and ``diffusers`` are imported lazily inside functions so that the API, job
queue, CLI argument handling and tests work without the ML stack installed.
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from PIL import Image, ImageEnhance, ImageOps

from neuroitburd.config import Settings
from neuroitburd.errors import (
    DeviceUnavailableError,
    GenerationError,
    GpuOutOfMemoryError,
    LoraLoadError,
    LoraNotFoundError,
    ModelLoadError,
)
from neuroitburd.image_utils import fit_to_model
from neuroitburd.params import GenerationParams

if TYPE_CHECKING:
    from types import ModuleType

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
"""Called as ``on_progress(completed_steps, total_steps)`` during generation."""

LORA_ADAPTER_NAME = "style"
OOM_HINT = (
    "Close other GPU applications or set NEUROITBURD_OFFLOAD=model to trade speed for memory."
)


class Stylizer(Protocol):
    """Anything that can turn an input image into a stylized one."""

    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        on_progress: ProgressCallback | None = None,
    ) -> Image.Image:
        """Fit ``image`` to the model size and return the stylized result."""
        ...

    def describe(self) -> dict[str, Any]:
        """Static facts about the loaded model, for logging and the health endpoint."""
        ...


def create_stylizer(settings: Settings) -> Stylizer:
    """Build the stylizer selected by ``settings`` (fake or real SDXL). May take minutes."""
    if settings.fake_pipeline:
        logger.warning("NEUROITBURD_FAKE_PIPELINE is set: using FakeStylizer, no real model")
        return FakeStylizer(settings, step_delay=0.05)
    return SDXLStylizer.load(settings)


# --- device -----------------------------------------------------------------------


def _import_torch() -> ModuleType:
    try:
        import torch
    except ImportError as exc:
        raise ModelLoadError(
            'PyTorch is not installed; install the ML extras with: pip install -e ".[ml]"'
        ) from exc
    module: ModuleType = torch  # typed even when torch is absent (mypy then sees Any)
    return module


def resolve_device(setting: str) -> str:
    """Map ``auto|cuda|cpu`` to a concrete device. Never falls back silently from ``cuda``."""
    torch = _import_torch()
    cuda_ok = bool(torch.cuda.is_available())
    if setting == "cuda":
        if not cuda_ok:
            raise DeviceUnavailableError(
                "CUDA was requested (NEUROITBURD_DEVICE=cuda / --device cuda) but is not "
                "available; check the NVIDIA driver and that a CUDA build of PyTorch is installed"
            )
        return "cuda"
    if setting == "cpu":
        return "cpu"
    if setting == "auto":
        if cuda_ok:
            return "cuda"
        logger.warning("CUDA not available; running on CPU (expect several minutes per image)")
        return "cpu"
    raise DeviceUnavailableError(f"Unknown device setting {setting!r}; use auto, cuda or cpu")


def gpu_info() -> dict[str, Any] | None:
    """Name and memory of CUDA device 0, or ``None`` if torch is not loaded or has no CUDA.

    Never imports torch itself: in fake mode, or before the model loads, this returns
    ``None`` instead of paying a multi-second import on the caller's thread.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(0)
        return {
            "name": str(torch.cuda.get_device_name(0)),
            "total_mb": int(total) // 2**20,
            "free_mb": int(free) // 2**20,
        }
    except Exception:
        logger.debug("Could not query GPU info", exc_info=True)
        return None


def _free_cuda_memory(torch: ModuleType) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- SDXL -------------------------------------------------------------------------


class SDXLStylizer:
    """SDXL img2img pipeline, loaded once and reused for every generation."""

    def __init__(self, pipe: Any, settings: Settings, device: str, dtype: str) -> None:
        self._pipe = pipe
        self._settings = settings
        self._device = device
        self._dtype = dtype

    @classmethod
    def load(cls, settings: Settings) -> SDXLStylizer:
        """Load model, optional VAE override and optional LoRA, then place on the device."""
        lora_path = settings.lora_path
        if lora_path is not None and not lora_path.is_file():
            # Checked first so a typo fails in milliseconds, not after loading 7 GB.
            raise LoraNotFoundError(f"LoRA file not found: {lora_path}")

        device = resolve_device(settings.device)
        torch = _import_torch()
        try:
            from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline
        except ImportError as exc:
            raise ModelLoadError(
                'diffusers is not installed; install the ML extras with: pip install -e ".[ml]"'
            ) from exc

        use_fp16 = device == "cuda"
        dtype = torch.float16 if use_fp16 else torch.float32
        started = time.perf_counter()
        logger.info(
            "Loading model=%s vae=%s device=%s dtype=%s offload=%s",
            settings.model_id,
            settings.vae_id or "(model default)",
            device,
            "float16" if use_fp16 else "float32",
            settings.offload,
        )

        try:
            kwargs: dict[str, Any] = {"torch_dtype": dtype, "use_safetensors": True}
            if settings.vae_id:
                kwargs["vae"] = AutoencoderKL.from_pretrained(settings.vae_id, torch_dtype=dtype)
            pipe = _from_pretrained(
                StableDiffusionXLImg2ImgPipeline, settings.model_id, kwargs, use_fp16
            )
        except Exception as exc:
            logger.exception("Model loading failed")
            raise ModelLoadError(
                f"Could not load model {settings.model_id!r} ({type(exc).__name__}). "
                "Check the model ID, network access or Hugging Face cache, and free disk space."
            ) from exc

        pipe.set_progress_bar_config(disable=True)
        if lora_path is not None:
            _apply_lora(pipe, lora_path, settings.lora_weight)
        _place_on_device(pipe, device, settings.offload, torch)

        logger.info("Model ready in %.1fs", time.perf_counter() - started)
        return cls(pipe, settings, device, "float16" if use_fp16 else "float32")

    def describe(self) -> dict[str, Any]:
        """Model, device and adapter facts for logs and the health endpoint."""
        lora = self._settings.lora_path
        return {
            "backend": "sdxl",
            "model_id": self._settings.model_id,
            "vae_id": self._settings.vae_id or None,
            "device": self._device,
            "dtype": self._dtype,
            "offload": self._settings.offload,
            "lora": lora.name if lora else None,
            "lora_weight": self._settings.lora_weight if lora else None,
        }

    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        on_progress: ProgressCallback | None = None,
    ) -> Image.Image:
        """Run img2img on ``image`` (fitted to ~1 MP) and return the result."""
        torch = _import_torch()
        source = fit_to_model(image)
        total = params.effective_steps
        prompt = params.prompt or self._settings.default_prompt
        negative = params.negative_prompt or self._settings.default_negative_prompt or None
        # A CPU generator gives the same noise for a seed regardless of device or offload mode.
        generator = (
            torch.Generator(device="cpu").manual_seed(params.seed)
            if params.seed is not None
            else None
        )

        def on_step_end(
            _pipe: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]
        ) -> dict[str, Any]:
            if on_progress is not None:
                on_progress(step + 1, total)
            return callback_kwargs

        logger.info(
            "Generating size=%dx%d strength=%s steps=%d (effective %d) guidance=%s seed=%s",
            source.width,
            source.height,
            params.strength,
            params.steps,
            total,
            params.guidance,
            params.seed,
        )
        started = time.perf_counter()
        try:
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=source,
                strength=params.strength,
                num_inference_steps=params.steps,
                guidance_scale=params.guidance,
                generator=generator,
                callback_on_step_end=on_step_end,
            )
            output: Image.Image = result.images[0]
        except torch.cuda.OutOfMemoryError as exc:
            _free_cuda_memory(torch)
            raise GpuOutOfMemoryError(
                f"GPU ran out of memory generating a {source.width}x{source.height} image. "
                + OOM_HINT
            ) from exc
        except Exception as exc:
            logger.exception("Generation failed")
            raise GenerationError(f"Image generation failed ({type(exc).__name__})") from exc

        if output.getextrema() == ((0, 0), (0, 0), (0, 0)):
            logger.warning("Output is completely black; likely fp16 NaNs in the VAE")
        logger.info("Generated in %.1fs", time.perf_counter() - started)
        return output


def _from_pretrained(
    pipeline_cls: Any, model_id: str, kwargs: dict[str, Any], use_fp16: bool
) -> Any:
    """Prefer the smaller fp16 weight variant; fall back if the repo does not provide it."""
    if not use_fp16:
        return pipeline_cls.from_pretrained(model_id, **kwargs)
    try:
        return pipeline_cls.from_pretrained(model_id, variant="fp16", **kwargs)
    except (OSError, ValueError) as exc:
        if "variant" not in str(exc).lower():
            raise
        logger.warning("Model %s has no fp16 variant; loading default weights", model_id)
        return pipeline_cls.from_pretrained(model_id, **kwargs)


def _apply_lora(pipe: Any, path: Path, weight: float) -> None:
    logger.info("Loading LoRA %s with weight %s", path.name, weight)
    try:
        pipe.load_lora_weights(
            str(path.parent), weight_name=path.name, adapter_name=LORA_ADAPTER_NAME
        )
        pipe.set_adapters([LORA_ADAPTER_NAME], adapter_weights=[weight])
    except ImportError as exc:
        raise LoraLoadError("LoRA support requires peft: pip install peft") from exc
    except Exception as exc:
        logger.exception("LoRA loading failed")
        raise LoraLoadError(
            f"Could not apply LoRA {path.name} ({type(exc).__name__}); "
            "it must be an SDXL 1.0 LoRA in .safetensors format"
        ) from exc


def _place_on_device(pipe: Any, device: str, offload: str, torch: ModuleType) -> None:
    try:
        if device == "cuda" and offload == "model":
            # Keeps only the active sub-model on the GPU. Must not be combined with .to("cuda").
            pipe.enable_model_cpu_offload()
        else:
            if offload != "none":
                logger.warning("Offload mode %r ignored on device %s", offload, device)
            pipe.to(device)
    except torch.cuda.OutOfMemoryError as exc:
        _free_cuda_memory(torch)
        raise GpuOutOfMemoryError(
            "GPU ran out of memory while loading the model. " + OOM_HINT
        ) from exc


# --- fake -------------------------------------------------------------------------


class FakeStylizer:
    """Deterministic, GPU-free stand-in for tests and frontend development.

    Produces a posterized, colour-boosted version of the input; the seed rotates colour
    channels so different seeds give visibly different results.
    """

    def __init__(self, settings: Settings | None = None, step_delay: float = 0.0) -> None:
        self._settings = settings or Settings()
        self._step_delay = step_delay

    def describe(self) -> dict[str, Any]:
        """Facts reported in place of a real model."""
        return {
            "backend": "fake",
            "model_id": "fake",
            "vae_id": None,
            "device": "cpu",
            "dtype": None,
            "offload": "none",
            "lora": None,
            "lora_weight": None,
        }

    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        on_progress: ProgressCallback | None = None,
    ) -> Image.Image:
        """Return a cheap stylized version of ``image`` at model size."""
        source = fit_to_model(image)
        total = params.effective_steps
        for step in range(1, total + 1):
            if self._step_delay:
                time.sleep(self._step_delay)
            if on_progress is not None:
                on_progress(step, total)
        boosted = ImageEnhance.Color(source).enhance(1.0 + params.strength * 2)
        out = ImageOps.posterize(boosted, 3)
        shift = (params.seed or 0) % 3
        if shift:
            r, g, b = out.split()
            out = Image.merge("RGB", (g, b, r) if shift == 1 else (b, r, g))
        return out
