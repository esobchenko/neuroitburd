import io
from pathlib import Path

import pytest
from PIL import Image

from neuroitburd.errors import InvalidImageError, InvalidParamsError
from neuroitburd.image_utils import (
    DEFAULT_MAX_PIXELS,
    MAX_SOURCE_SIDE,
    fit_size,
    fit_to_model,
    load_image,
    save_image,
)

EXIF_ORIENTATION = 0x0112


def encode(img: Image.Image, fmt: str, **kwargs: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


# --- load_image -----------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_load_supported_formats_from_bytes(fmt: str) -> None:
    img = load_image(encode(Image.new("RGB", (120, 80), "red"), fmt))
    assert img.mode == "RGB"
    assert img.size == (120, 80)


def test_load_from_path(tmp_path: Path) -> None:
    path = tmp_path / "in.png"
    Image.new("RGB", (100, 100)).save(path)
    assert load_image(path).size == (100, 100)
    assert load_image(str(path)).size == (100, 100)


def test_format_detected_from_content_not_extension(tmp_path: Path) -> None:
    path = tmp_path / "actually_png.jpg"
    Image.new("RGB", (100, 100)).save(path, format="PNG")
    assert load_image(path).size == (100, 100)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InvalidImageError, match="not found"):
        load_image(tmp_path / "nope.jpg")


def test_unsupported_format() -> None:
    with pytest.raises(InvalidImageError, match="Unsupported image format BMP"):
        load_image(encode(Image.new("RGB", (100, 100)), "BMP"))


def test_not_an_image() -> None:
    with pytest.raises(InvalidImageError, match="not a supported image"):
        load_image(b"definitely not an image")


def test_truncated_image() -> None:
    data = encode(Image.effect_noise((256, 256), 64).convert("RGB"), "PNG")
    with pytest.raises(InvalidImageError, match="corrupt or truncated"):
        load_image(data[: len(data) // 2])


@pytest.mark.parametrize(
    ("size", "match"),
    [
        ((63, 200), "too small"),
        ((MAX_SOURCE_SIDE + 8, 4000), "too large"),
        ((1000, 200), "aspect ratio"),
    ],
)
def test_dimension_limits(size: tuple[int, int], match: str) -> None:
    with pytest.raises(InvalidImageError, match=match):
        load_image(encode(Image.new("L", size), "PNG"))


def test_exif_orientation_applied() -> None:
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = 6  # rotate 90 degrees clockwise on display
    data = encode(Image.new("RGB", (200, 100)), "JPEG", exif=exif)
    assert load_image(data).size == (100, 200)


def test_rgba_flattened_on_white() -> None:
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 0))  # fully transparent red
    out = load_image(encode(img, "PNG"))
    assert out.mode == "RGB"
    assert out.getpixel((50, 50)) == (255, 255, 255)


def test_palette_with_transparency() -> None:
    img = Image.new("P", (100, 100), 0)
    img.putpalette([0, 0, 0] * 256)
    data = encode(img, "PNG", transparency=0)
    assert load_image(data).getpixel((0, 0)) == (255, 255, 255)


def test_grayscale_and_16bit() -> None:
    assert load_image(encode(Image.new("L", (100, 100), 128), "PNG")).getpixel((0, 0)) == (
        128,
        128,
        128,
    )
    gray16 = Image.new("I;16", (100, 100), 65535)
    assert load_image(encode(gray16, "PNG")).getpixel((0, 0)) == (255, 255, 255)


# --- fit_size / fit_to_model ------------------------------------------------------


@pytest.mark.parametrize(
    "size",
    [(1024, 1024), (4000, 3000), (3000, 4000), (640, 480), (100, 100), (4096, 1024), (1919, 1081)],
)
def test_fit_size_properties(size: tuple[int, int]) -> None:
    w, h = size
    tw, th = fit_size(w, h)
    assert tw % 8 == 0 and th % 8 == 0
    assert tw * th <= DEFAULT_MAX_PIXELS
    assert tw * th >= DEFAULT_MAX_PIXELS * 0.97  # uses close to the full budget
    assert abs(tw / th - w / h) / (w / h) < 0.02  # aspect ratio preserved


def test_fit_size_exact_square() -> None:
    assert fit_size(1024, 1024) == (1024, 1024)
    assert fit_size(2048, 2048) == (1024, 1024)


def test_fit_size_custom_budget() -> None:
    assert fit_size(1000, 1000, max_pixels=512 * 512, multiple=64) == (512, 512)


def test_fit_to_model_upscales_small_image() -> None:
    out = fit_to_model(Image.new("RGB", (160, 120)))
    assert out.width > 160 and out.height > 120
    assert out.size == fit_size(160, 120)


def test_fit_to_model_noop_when_already_fitting() -> None:
    img = Image.new("RGB", (1024, 1024))
    assert fit_to_model(img) is img


# --- save_image -------------------------------------------------------------------


@pytest.mark.parametrize(("name", "fmt"), [("a.png", "PNG"), ("b.JPG", "JPEG"), ("c.webp", "WEBP")])
def test_save_formats_and_creates_dirs(tmp_path: Path, name: str, fmt: str) -> None:
    path = save_image(Image.new("RGB", (64, 64)), tmp_path / "nested" / name)
    with Image.open(path) as saved:
        assert saved.format == fmt
    assert [p.name for p in path.parent.iterdir()] == [name]  # no temp files left


def test_save_rejects_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(InvalidParamsError, match="extension"):
        save_image(Image.new("RGB", (64, 64)), tmp_path / "out.gif")
