"""Exception hierarchy shared by the CLI, the job worker and the API.

Messages are meant to be shown to users: short, actionable, without tracebacks.
"""

from __future__ import annotations


class NeuroitburdError(Exception):
    """Base class for all expected, user-reportable errors."""


class InvalidImageError(NeuroitburdError):
    """Input image is missing, unreadable, unsupported or has invalid dimensions."""


class UnsupportedImageError(InvalidImageError):
    """Input is not an image at all, or not one of the accepted formats (JPEG, PNG, WebP)."""


class InvalidParamsError(NeuroitburdError):
    """Generation parameters are out of range or inconsistent."""


class DeviceUnavailableError(NeuroitburdError):
    """The requested compute device (e.g. CUDA) is not available."""


class ModelLoadError(NeuroitburdError):
    """The base model, VAE or ML dependencies could not be loaded."""


class LoraNotFoundError(NeuroitburdError):
    """The configured LoRA adapter file does not exist."""


class LoraLoadError(NeuroitburdError):
    """The LoRA adapter file exists but could not be applied to the model."""


class GpuOutOfMemoryError(NeuroitburdError):
    """CUDA ran out of memory during model placement or generation."""


class GenerationError(NeuroitburdError):
    """Unexpected failure inside the diffusion pipeline."""


class QueueFullError(NeuroitburdError):
    """Too many jobs are waiting; the client should retry later."""
