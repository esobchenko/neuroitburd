"""Tests for training/prepare_dataset.py (no torch needed)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image
from prepare_dataset import (
    CROP_SIZE,
    MAX_AREA,
    MIN_DETAIL,
    cap_counts,
    caption_from_filename,
    choose_crops,
    crop_caption,
    crop_count,
    crop_transparent,
    crop_windows,
    detail_ratio,
    main,
    overlap,
    prepare_image,
    target_size,
    texture,
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


# ---- detail crops -----------------------------------------------------------------------


def _noise(size: tuple[int, int], seed: int = 0) -> Image.Image:
    """Sharp, textured RGB test image (random pixels)."""
    rnd = random.Random(seed)
    data = bytes(rnd.randrange(256) for _ in range(size[0] * size[1] * 3))
    return Image.frombytes("RGB", size, data)


def test_detail_ratio_separates_sharp_from_upscaled() -> None:
    sharp = _noise((256, 256))
    upscaled = _noise((64, 64)).resize((256, 256), Image.Resampling.BICUBIC)
    assert detail_ratio(sharp) > MIN_DETAIL > detail_ratio(upscaled)
    assert detail_ratio(Image.new("RGB", (64, 64), "gray")) == 0.0  # flat, no detail at all


def test_texture_prefers_detail() -> None:
    assert texture(_noise((64, 64))) > texture(Image.new("RGB", (64, 64), "gray")) == 0.0


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((1200, 1200), 0),  # 1.4 MP: below the 2 MP minimum
        ((1600, 1400), 1),  # 2.2 MP
        ((2200, 1500), 2),  # 3.3 MP
        ((3000, 2500), 3),  # 7.5 MP: capped at max_crops
        ((4000, 900), 0),  # too short for a 1024 crop
    ],
)
def test_crop_count(size: tuple[int, int], expected: int) -> None:
    assert crop_count(*size) == expected


def test_crop_windows_avoid_bottom_and_sides() -> None:
    w, h = 3000, 2000
    boxes = crop_windows(w, h)
    assert boxes
    for x0, y0, x1, y1 in boxes:
        assert (x1 - x0, y1 - y0) == (CROP_SIZE, CROP_SIZE)
        assert x0 >= round(0.06 * w) and x1 <= w - round(0.06 * w)
        assert y1 <= h - round(0.12 * h)


def test_crop_windows_none_when_no_room_above_signature_band() -> None:
    assert crop_windows(2000, 1100) == []  # 1024 + 12% bottom band > 1100


def test_overlap() -> None:
    assert overlap((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert overlap((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert overlap((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_choose_crops_picks_textured_non_overlapping_windows() -> None:
    image = Image.new("RGB", (3000, 2000), "gray")
    image.paste(_noise((1100, 1100)), (1700, 100))  # textured patch top right
    boxes = choose_crops(image, 2)
    assert len(boxes) == 2
    assert boxes[0][0] >= 1500  # the most textured window covers the patch
    assert overlap(boxes[0], boxes[1]) <= 0.2


def test_cap_counts_trims_largest_first() -> None:
    assert cap_counts({"a": 3, "b": 1, "c": 2}, 4) == {"a": 2, "b": 1, "c": 1}
    assert cap_counts({"a": 1}, 5) == {"a": 1}
    assert sum(cap_counts({"a": 3, "b": 3}, 0).values()) == 0


def test_main_crops_only(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    _noise((1700, 1400), 1).save(src / "Roitburd_Sharp-Big.png")  # 2.4 MP, sharp: 1 crop
    _noise((850, 700), 2).resize((1700, 1400)).save(src / "Roitburd_Soft-Big.png")  # upscaled
    _noise((900, 700), 3).save(src / "Roitburd_Small.png")  # too small
    crops = tmp_path / "crops"
    args = ["--input", str(src), "--crops-only", "--crops-output", str(crops), "--trigger", "T"]
    assert main(args) == 0
    assert sorted(p.name for p in crops.iterdir()) == [
        "Roitburd_Sharp-Big_crop0.png",
        "Roitburd_Sharp-Big_crop0.txt",
    ]
    assert (crops / "Roitburd_Sharp-Big_crop0.txt").read_text() == (
        "T, close-up detail of brushwork\n"
    )
    with Image.open(crops / "Roitburd_Sharp-Big_crop0.png") as im:
        assert im.size == (CROP_SIZE, CROP_SIZE)


def test_main_crops_only_requires_crops_output(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--input", str(tmp_path), "--crops-only"])


def test_crop_caption_tags_doodles() -> None:
    assert crop_caption(Path("Roitburd_Bird-Man-with-Doodles.png"), "T", "detail") == (
        "T, detail, with doodles"
    )
    assert crop_caption(Path("Roitburd_Rider-on-Zebra.png"), "T", "detail") == "T, detail"


def test_main_single_crop_pattern(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    _noise((2600, 2000), 4).save(src / "Roitburd_Gentleman-with-Swans.png")  # room for 2 crops
    _noise((2600, 2000), 5).save(src / "Roitburd_Big-with-Doodles.png")
    crops = tmp_path / "crops"
    args = ["--input", str(src), "--crops-only", "--crops-output", str(crops), "--trigger", "T"]
    assert main([*args, "--single-crop", "gentleman", "--max-crop-fraction", "5"]) == 0
    assert len(list(crops.glob("Roitburd_Gentleman-with-Swans_crop*.png"))) == 1
    assert len(list(crops.glob("Roitburd_Big-with-Doodles_crop*.png"))) > 1  # not limited
    assert (crops / "Roitburd_Big-with-Doodles_crop0.txt").read_text().endswith("with doodles\n")
