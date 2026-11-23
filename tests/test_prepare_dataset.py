"""Tests for training/prepare_dataset.py (no torch needed)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from prepare_dataset import (
    MAX_AREA,
    caption_from_filename,
    crop_transparent,
    main,
    prepare_image,
    target_size,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Roitburd_Fish-on-Platter.jpg", "T, fish on platter"),
        ("Roitburd_King-with-Lyre_1988.jpg", "T, king with lyre"),
        ("Roitburd_Samurai_2010.jpg", "T, samurai"),
        ("Plain-Title.png", "T, plain title"),
        ("Roitburd_.png", "T"),
    ],
)
def test_caption_from_filename(name: str, expected: str) -> None:
    assert caption_from_filename(Path(name), "T") == expected


@pytest.mark.parametrize("size", [(3120, 1866), (654, 1592), (600, 477), (1024, 1024), (5000, 900)])
def test_target_size_rules(size: tuple[int, int]) -> None:
    w, h = target_size(*size)
    assert w % 64 == 0 and h % 64 == 0
    assert w * h <= MAX_AREA
    assert min(w, h) >= 512
    if min(size) >= 1024:
        assert abs(w / h - size[0] / size[1]) / (size[0] / size[1]) < 0.15


def test_crop_transparent_removes_padding() -> None:
    im = Image.new("RGBA", (300, 100), (0, 0, 0, 0))
    im.paste(Image.new("RGBA", (100, 100), (200, 10, 10, 255)), (100, 0))
    assert crop_transparent(im).size == (100, 100)


def test_crop_transparent_keeps_rgb() -> None:
    im = Image.new("RGB", (50, 40))
    assert crop_transparent(im) is im


def test_prepare_image_outputs_rgb_training_size() -> None:
    im = Image.new("RGBA", (3120, 1866), (0, 0, 0, 0))
    im.paste(Image.new("RGBA", (1400, 1866), (90, 60, 30, 255)), (860, 0))
    out = prepare_image(im)
    assert out.mode == "RGB"
    assert out.size == target_size(1400, 1866)
    assert out.getpixel((0, 0)) == (90, 60, 30)  # padding cropped, not resized into the image


def test_main_writes_pairs(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    Image.new("RGB", (800, 600), "red").save(src / "Roitburd_Red-Square.jpg")
    (src / "notes.txt").write_text("ignored")
    dst = tmp_path / "out"
    assert main(["--input", str(src), "--output", str(dst), "--trigger", "xyz"]) == 0
    assert (dst / "Roitburd_Red-Square.txt").read_text() == "xyz, red square\n"
    with Image.open(dst / "Roitburd_Red-Square.png") as out:
        assert out.size == target_size(800, 600)


def test_main_empty_folder(tmp_path: Path) -> None:
    assert main(["--input", str(tmp_path), "--output", str(tmp_path / "o")]) == 2
