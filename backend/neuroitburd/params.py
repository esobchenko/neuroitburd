"""Generation parameters: the single place where they are validated (used by CLI and API)."""

from __future__ import annotations

import secrets
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from neuroitburd.errors import InvalidParamsError

MAX_PROMPT_CHARS = 500
MAX_SEED = 2**32 - 1


class GenerationParams(BaseModel):
    """Validated settings for one img2img generation.

    ``prompt`` / ``negative_prompt`` of ``None`` mean "use the configured default".
    ``seed`` of ``None`` means "pick a random seed" (the caller records which one).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    strength: float = Field(default=0.65, gt=0.0, le=1.0)
    steps: int = Field(default=30, ge=1, le=100)
    guidance: float = Field(default=7.0, ge=0.0, le=20.0)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    prompt: str | None = Field(default=None, max_length=MAX_PROMPT_CHARS)
    negative_prompt: str | None = Field(default=None, max_length=MAX_PROMPT_CHARS)

    @field_validator("prompt", "negative_prompt")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def _check_effective_steps(self) -> Self:
        # img2img runs only int(steps * strength) denoising steps; zero makes diffusers fail.
        if self.effective_steps < 1:
            raise ValueError(
                f"steps * strength must be at least 1 (got {self.steps} * {self.strength}); "
                "increase steps or strength"
            )
        return self

    @property
    def effective_steps(self) -> int:
        """Number of denoising steps img2img actually runs."""
        return int(self.steps * self.strength)

    def with_resolved_seed(self) -> GenerationParams:
        """Return a copy with a concrete random seed if none was given (for reproducibility)."""
        if self.seed is not None:
            return self
        return self.model_copy(update={"seed": secrets.randbelow(MAX_SEED + 1)})

    @classmethod
    def create(cls, **values: Any) -> GenerationParams:
        """Build params, converting pydantic errors into a readable ``InvalidParamsError``."""
        try:
            return cls(**values)
        except ValidationError as exc:
            raise InvalidParamsError(_format_validation_error(exc)) from exc


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err["loc"])
        msg = err["msg"].removeprefix("Value error, ")
        parts.append(f"{field}: {msg}" if field else msg)
    return "Invalid generation parameters: " + "; ".join(parts)
