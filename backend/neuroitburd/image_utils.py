"""Image loading, validation, model-size fitting and saving (Pillow only)."""

from __future__ import annotations

import io
import logging
import math
import os
import tempfile
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from neuroitburd.errors import InvalidImageError, InvalidParamsError

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
SAVE_FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}
MIN_SOURCE_SIDE = 64
MAX_SOURCE_SIDE = 8192
MAX_ASPECT_RATIO = 4.0
DEFAULT_MAX_PIXELS = 1024 * 1024  # SDXL native resolution area
DEFAULT_MULTIPLE = 8  # VAE downsampling factor


def load_image(source: str | Path | bytes) -> Image.Image:
    """Open, validate and normalize an image to an EXIF-upright RGB ``Image``.

    ``source`` is a file path or the raw bytes of an uploaded file. The format is
    detected from the content, not from the file name.
    """
    if isinstance(source, bytes):
        stream: io.BytesIO | Path = io.BytesIO(source)
        label = "uploaded image"
    else:
        stream = Path(source)
        label = stream.name
        if not stream.is_file():
            raise InvalidImageError(f"Input file not found: {source}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(stream)
            _validate_header(img, label)
            img.load()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise InvalidImageError(
            f"Cannot read {label}: not a supported image (use JPEG, PNG or WebP)"
        ) from None
    except (OSError, SyntaxError, ValueError) as exc:
        logger.debug("Failed to decode %s: %s", label, exc)
        raise InvalidImageError(f"Cannot read {label}: file is corrupt or truncated") from None

    return _to_rgb(ImageOps.exif_transpose(img))


def _validate_header(img: Image.Image, label: str) -> None:
    """Check format and dimensions before decoding pixel data."""
    if img.format not in SUPPORTED_FORMATS:
        raise InvalidImageError(
            f"Unsupported image format {img.format or 'unknown'} for {label}; use JPEG, PNG or WebP"
        )
    w, h = img.size
    if min(w, h) < MIN_SOURCE_SIDE:
        raise InvalidImageError(
            f"Image {label} is too small ({w}x{h}); minimum side is {MIN_SOURCE_SIDE}px"
        )
    if max(w, h) > MAX_SOURCE_SIDE:
        raise InvalidImageError(
            f"Image {label} is too large ({w}x{h}); maximum side is {MAX_SOURCE_SIDE}px"
        )
    if max(w, h) / min(w, h) > MAX_ASPECT_RATIO:
        raise InvalidImageError(
            f"Image {label} aspect ratio is too extreme ({w}x{h}); "
            f"maximum is {MAX_ASPECT_RATIO:g}:1"
        )


def _to_rgb(img: Image.Image) -> Image.Image:
    """Convert any Pillow mode to RGB; transparency is flattened onto white."""
    if img.mode == "RGB":
        return img
    if img.mode in {"I", "I;16", "I;16B", "I;16L"}:
        # 16-bit grayscale: scale to 8 bits instead of clipping.
        img = img.convert("I").point(lambda v: v / 256).convert("L")
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in {"RGBA", "LA", "PA"}:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def fit_size(
    width: int,
    height: int,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    multiple: int = DEFAULT_MULTIPLE,
) -> tuple[int, int]:
    """Target size with the same aspect ratio, area close to ``max_pixels``, sides divisible
    by ``multiple``. Small images are scaled up, large ones down."""
    scale = math.sqrt(max_pixels / (width * height))
    target_w = max(multiple, int(width * scale) // multiple * multiple)
    target_h = max(multiple, int(height * scale) // multiple * multiple)
    return target_w, target_h


def fit_to_model(
    img: Image.Image,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    multiple: int = DEFAULT_MULTIPLE,
) -> Image.Image:
    """Resize ``img`` to the size returned by :func:`fit_size` (Lanczos resampling)."""
    size = fit_size(img.width, img.height, max_pixels, multiple)
    if size == img.size:
        return img
    return img.resize(size, Image.Resampling.LANCZOS)


def output_format(path: str | Path) -> str:
    """Return the Pillow format for an output path's extension, or raise ``InvalidParamsError``."""
    suffix = Path(path).suffix
    fmt = SAVE_FORMATS.get(suffix.lower())
    if fmt is None:
        raise InvalidParamsError(
            f"Unsupported output extension {suffix or '(none)'!r}; "
            f"use one of {', '.join(SAVE_FORMATS)}"
        )
    return fmt


def save_image(img: Image.Image, path: str | Path) -> Path:
    """Save atomically (temp file + rename) so readers never see a partial file.

    The format follows the extension: .png, .jpg/.jpeg or .webp.
    """
    path = Path(path)
    fmt = output_format(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            options = {"quality": 95} if fmt in {"JPEG", "WEBP"} else {}
            img.save(fh, format=fmt, **options)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path
