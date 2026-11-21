"""Runtime configuration loaded from ``NEUROITBURD_*`` environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DeviceSetting = Literal["auto", "cuda", "cpu"]
OffloadMode = Literal["none", "model"]

DEFAULT_PROMPT = (
    "An expressive postmodern oil painting, bold contrasting colors, "
    "distorted figurative forms, surreal atmosphere, textured brushwork, "
    "dramatic composition."
)
DEFAULT_NEGATIVE_PROMPT = (
    "photo, photorealistic, blurry, low quality, watermark, text, signature, frame"
)

_ENV_PREFIX = "NEUROITBURD_"
MAX_LORA_WEIGHT = 2.0


class ConfigError(ValueError):
    """Raised when an environment variable has an invalid value."""


def _env(name: str, default: str) -> str:
    return os.environ.get(_ENV_PREFIX + name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "1" if default else "0").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off", ""}:
        return False
    raise ConfigError(f"{_ENV_PREFIX}{name} must be a boolean, got {raw!r}")


def _env_float(name: str, default: float) -> float:
    raw = _env(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{_ENV_PREFIX}{name} must be a number, got {raw!r}") from exc


def _env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    raw = _env(name, default).lower()
    if raw not in choices:
        raise ConfigError(f"{_ENV_PREFIX}{name} must be one of {choices}, got {raw!r}")
    return raw


@dataclass(frozen=True)
class Settings:
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_id: str = "madebyollin/sdxl-vae-fp16-fix"  # empty string = use the model's own VAE
    device: DeviceSetting = "auto"
    offload: OffloadMode = "none"
    output_dir: Path = Path("outputs")
    lora_path: Path | None = None
    lora_weight: float = 0.8
    default_prompt: str = DEFAULT_PROMPT
    default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    max_upload_mb: float = 20.0
    max_queue_size: int = 20
    fake_pipeline: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.lora_weight <= MAX_LORA_WEIGHT:
            raise ConfigError(f"LoRA weight must be between 0 and {MAX_LORA_WEIGHT:g}")
        if self.max_upload_mb <= 0:
            raise ConfigError("Maximum upload size must be positive")
        if self.max_queue_size < 1:
            raise ConfigError("Maximum queue size must be at least 1")

    @classmethod
    def from_env(cls) -> Settings:
        lora_raw = _env("LORA_PATH", "")
        return cls(
            model_id=_env("MODEL_ID", cls.model_id),
            vae_id=_env("VAE_ID", cls.vae_id),
            device=_env_choice("DEVICE", "auto", ("auto", "cuda", "cpu")),  # type: ignore[arg-type]
            offload=_env_choice("OFFLOAD", "none", ("none", "model")),  # type: ignore[arg-type]
            output_dir=Path(_env("OUTPUT_DIR", "outputs")).expanduser().resolve(),
            lora_path=Path(lora_raw).expanduser().resolve() if lora_raw else None,
            lora_weight=_env_float("LORA_WEIGHT", cls.lora_weight),
            default_prompt=_env("DEFAULT_PROMPT", DEFAULT_PROMPT) or DEFAULT_PROMPT,
            default_negative_prompt=_env("DEFAULT_NEGATIVE_PROMPT", DEFAULT_NEGATIVE_PROMPT),
            max_upload_mb=_env_float("MAX_UPLOAD_MB", cls.max_upload_mb),
            max_queue_size=int(_env_float("MAX_QUEUE_SIZE", cls.max_queue_size)),
            fake_pipeline=_env_bool("FAKE_PIPELINE", False),
        )


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once with a compact key=value-friendly format."""
    logging.basicConfig(
        level=(level or os.environ.get("NEUROITBURD_LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
