"""Prepare a folder of paintings for LoRA training.

For every image in the input folder this writes, to the output folder:
- ``<name>.png``: RGB, transparent padding cropped away, resized and center-cropped to a
  training size (about 1 megapixel, both sides multiples of 64, aspect ratio kept);
- ``<name>.txt``: the caption, ``"<trigger>, <title>"``. The title comes from the file
  name, e.g. ``Roitburd_Fish-on-Platter.jpg`` -> ``"fish on platter"``.

Usage:
    python training/prepare_dataset.py --input ~/Pictures/Roitburd --output data/roitburd
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_TRIGGER = "rtbrd style painting"
MAX_AREA = 1024 * 1024
MIN_SIDE = 512
STEP = 64


def caption_from_filename(path: Path, trigger: str) -> str:
    """Build a caption from a file name like ``Artist_Some-Title_1988.jpg``."""
    stem = path.stem
    if "_" in stem:
        stem = stem.split("_", 1)[1]  # drop the artist prefix
    stem = re.sub(r"_?\d{4}$", "", stem)  # drop a trailing year
    title = re.sub(r"[-_]+", " ", stem).strip().lower()
    return f"{trigger}, {title}" if title else trigger


def crop_transparent(image: Image.Image) -> Image.Image:
    """Crop away fully transparent borders (e.g. screenshots padded with alpha)."""
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        bbox = rgba.getchannel("A").point(lambda v: 255 if v > 0 else 0).getbbox()
        if bbox:
            return rgba.crop(bbox)
    return image


def target_size(width: int, height: int) -> tuple[int, int]:
    """Training size: aspect ratio kept, area at most ``MAX_AREA``, sides multiples of 64."""
    area = min(MAX_AREA, width * height)
    aspect = width / height
    w = math.sqrt(area * aspect)
    h = w / aspect
    tw = max(MIN_SIDE, round(w / STEP) * STEP)
    th = max(MIN_SIDE, round(h / STEP) * STEP)
    while tw * th > MAX_AREA:  # rounding up both sides can overshoot slightly
        if tw >= th:
            tw -= STEP
        else:
            th -= STEP
    return tw, th


def prepare_image(image: Image.Image) -> Image.Image:
    """Crop transparency, flatten to RGB and fit to the training size."""
    image = ImageOps.exif_transpose(image)
    image = crop_transparent(image)
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "black")
        background.paste(rgba, mask=rgba.getchannel("A"))
        image = background
    size = target_size(*image.size)
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trigger", default=DEFAULT_TRIGGER)
    args = parser.parse_args(argv)

    files = sorted(
        p for p in args.input.expanduser().iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        print(f"No images found in {args.input}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    for path in files:
        with Image.open(path) as src:
            src_size = src.size
            out = prepare_image(src)
        caption = caption_from_filename(path, args.trigger)
        out.save(args.output / f"{path.stem}.png")
        (args.output / f"{path.stem}.txt").write_text(caption + "\n", encoding="utf-8")
        print(f"{src_size[0]:>5}x{src_size[1]:<5} -> {out.width}x{out.height}  {caption}")
    print(f"{len(files)} images written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
