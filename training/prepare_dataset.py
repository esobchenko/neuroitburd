"""Prepare a folder of paintings for LoRA training.

For every image in the input folder this writes, to the output folder:
- ``<name>.png``: RGB, transparent padding cropped away, resized and center-cropped to a
  training size (about 1 megapixel, both sides multiples of 64, aspect ratio kept);
- ``<name>.txt``: the caption, ``"<trigger>, <title>"``. The title comes from the file
  name, e.g. ``Roitburd_Fish-on-Platter.jpg`` -> ``"fish on platter"``.

With ``--crops-output`` it also cuts detail crops (``<name>_crop<i>.png``) at native
resolution from the large, sharp sources, so the brushwork is learned at its real scale
instead of being shrunk to 1 megapixel:
- only sources of at least ``--crop-min-mp`` megapixels (after trimming padding), 1 crop per
  1.5 MP up to ``--max-crops``; all crops together at most ``--max-crop-fraction`` of the
  number of paintings;
- sources whose finest detail looks upscaled or blurry (``detail_ratio`` below
  ``--min-detail``) are skipped;
- the bottom edge and the side margins, where signatures sit, are avoided; within the rest,
  the most textured windows that do not overlap much are picked;
- the caption is ``"<trigger>, <crop caption>"``, without the title: a crop rarely shows
  the titled subject, and the tag lets prompts ask for the texture on purpose. Crops of the
  "...with Doodles" paintings get ``", with doodles"`` so that sub-style stays separable;
- paintings whose file name contains a ``--single-crop`` pattern (e.g. a large series that
  would otherwise dominate) get at most one crop.

Usage:
    python training/prepare_dataset.py --input ~/Pictures/Roitburd --output data/roitburd
    python training/prepare_dataset.py --input ~/Pictures/Roitburd --crops-only \
        --crops-output data/roitburd-detail
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_TRIGGER = "rtbrd style painting"
MAX_AREA = 1024 * 1024
MIN_SIDE = 512
STEP = 64
DEFAULT_CROP_CAPTION = "close-up detail of brushwork"
CROP_SIZE = 1024
CROP_MIN_MP = 2.0
MP_PER_CROP = 1.5
MAX_CROPS = 3
MAX_CROP_FRACTION = 0.5
MIN_DETAIL = 0.40  # calibrated on the Roitburd corpus: soft upscales score 0.2-0.36
SIDE_MARGIN = 0.06  # of the width, each side
TOP_MARGIN = 0.04  # of the height
BOTTOM_MARGIN = 0.12  # of the height: signatures
MAX_OVERLAP = 0.2  # intersection over union between crops of one painting


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


def clean_image(image: Image.Image) -> Image.Image:
    """Apply EXIF rotation, crop transparent padding and flatten to RGB (native size)."""
    image = ImageOps.exif_transpose(image)
    image = crop_transparent(image)
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "black")
        background.paste(rgba, mask=rgba.getchannel("A"))
        image = background
    return image


def prepare_image(image: Image.Image) -> Image.Image:
    """Crop transparency, flatten to RGB and fit to the training size."""
    image = clean_image(image)
    size = target_size(*image.size)
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def texture(image: Image.Image) -> float:
    """Mean fine-detail energy: average difference between the image and a slight blur."""
    gray = image.convert("L")
    blurred = gray.filter(ImageFilter.GaussianBlur(1))
    return float(ImageStat.Stat(ImageChops.difference(gray, blurred)).mean[0])


def detail_ratio(image: Image.Image) -> float:
    """Fine detail at native size relative to half size.

    A genuinely sharp image keeps a good share of its detail at full resolution (about
    0.45-0.95 on the paintings); one upscaled from a smaller original has almost none
    there, whatever it depicts (about 0.2-0.36).
    """
    half = image.resize(
        (max(1, image.width // 2), max(1, image.height // 2)), Image.Resampling.LANCZOS
    )
    coarse = texture(half)
    return texture(image) / coarse if coarse > 0 else 0.0


def crop_count(
    width: int,
    height: int,
    size: int = CROP_SIZE,
    max_crops: int = MAX_CROPS,
    min_mp: float = CROP_MIN_MP,
) -> int:
    """How many crops a source of this (trimmed) size gets: 1 per 1.5 MP, 0 below ``min_mp``."""
    if width * height < min_mp * 1e6 or width < size or height < size:
        return 0
    return max(1, min(max_crops, int(width * height / (MP_PER_CROP * 1e6))))


def crop_windows(
    width: int, height: int, size: int = CROP_SIZE, grid: int = 4
) -> list[tuple[int, int, int, int]]:
    """Candidate square windows away from the bottom edge and the side margins.

    The margins shrink when the image is too small for them, except the bottom one
    (signatures): if a window does not fit above it, there are no candidates.
    """
    left = min(round(SIDE_MARGIN * width), (width - size) // 2)
    top = min(round(TOP_MARGIN * height), max(0, height - size - round(BOTTOM_MARGIN * height)))
    x_max = width - left - size
    y_max = height - round(BOTTOM_MARGIN * height) - size
    if x_max < left or y_max < top:
        return []
    xs = sorted({left + round(i * (x_max - left) / (grid - 1)) for i in range(grid)})
    ys = sorted({top + round(i * (y_max - top) / (grid - 1)) for i in range(grid)})
    return [(x, y, x + size, y + size) for y in ys for x in xs]


def overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over union of two boxes."""
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area else 0.0


def choose_crops(
    image: Image.Image, n: int, size: int = CROP_SIZE
) -> list[tuple[int, int, int, int]]:
    """Pick up to ``n`` of the most textured candidate windows that barely overlap."""
    candidates = crop_windows(image.width, image.height, size)
    scored = sorted(candidates, key=lambda box: texture(image.crop(box)), reverse=True)
    chosen: list[tuple[int, int, int, int]] = []
    for box in scored:
        if len(chosen) == n:
            break
        if all(overlap(box, c) <= MAX_OVERLAP for c in chosen):
            chosen.append(box)
    return chosen


def cap_counts(counts: dict[str, int], limit: int) -> dict[str, int]:
    """Lower the largest per-image crop counts first until the total is at most ``limit``."""
    counts = dict(counts)
    while sum(counts.values()) > limit:
        name = max(counts, key=lambda k: (counts[k], k))
        if counts[name] == 0:
            break
        counts[name] -= 1
    return counts


def crop_caption(path: Path, trigger: str, caption: str) -> str:
    """Caption for a crop of ``path``; the doodle-overlay paintings get a separate tag."""
    tag = ", with doodles" if "doodles" in path.stem.lower() else ""
    return f"{trigger}, {caption}{tag}"


def write_crops(files: list[Path], output: Path, args: argparse.Namespace) -> int:
    """Cut detail crops from the large, sharp sources; return how many were written."""
    plans: dict[str, int] = {}
    images: dict[str, Path] = {}
    for path in files:
        with Image.open(path) as src:
            image = clean_image(src)
        n = crop_count(image.width, image.height, args.crop_size, args.max_crops, args.crop_min_mp)
        if any(pattern.lower() in path.stem.lower() for pattern in args.single_crop):
            n = min(n, 1)
        if n == 0:
            continue
        ratio = detail_ratio(image)
        if ratio < args.min_detail:
            print(f"skip (soft, detail {ratio:.2f})  {path.name}")
            continue
        plans[path.stem], images[path.stem] = n, path
    plans = cap_counts(plans, int(args.max_crop_fraction * len(files)))
    output.mkdir(parents=True, exist_ok=True)
    total = 0
    for stem, n in plans.items():
        with Image.open(images[stem]) as src:
            image = clean_image(src)
        boxes = choose_crops(image, n, args.crop_size)
        for i, box in enumerate(boxes):
            image.crop(box).save(output / f"{stem}_crop{i}.png")
            caption = crop_caption(images[stem], args.trigger, args.crop_caption)
            (output / f"{stem}_crop{i}.txt").write_text(caption + "\n", encoding="utf-8")
        total += len(boxes)
        print(f"{image.width:>5}x{image.height:<5} {len(boxes)} crop(s)  {stem}")
    print(f"{total} crops from {len(plans)} paintings written to {output}")
    return total


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="training set folder")
    parser.add_argument("--trigger", default=DEFAULT_TRIGGER)
    parser.add_argument("--crops-output", type=Path, help="also write detail crops here")
    parser.add_argument("--crops-only", action="store_true", help="skip the training set")
    parser.add_argument("--crop-caption", default=DEFAULT_CROP_CAPTION)
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument("--crop-min-mp", type=float, default=CROP_MIN_MP)
    parser.add_argument("--max-crops", type=int, default=MAX_CROPS)
    parser.add_argument("--max-crop-fraction", type=float, default=MAX_CROP_FRACTION)
    parser.add_argument("--min-detail", type=float, default=MIN_DETAIL)
    parser.add_argument(
        "--single-crop",
        action="append",
        default=[],
        metavar="PATTERN",
        help="at most one crop for file names containing PATTERN (repeatable)",
    )
    args = parser.parse_args(argv)
    if args.crops_only and not args.crops_output:
        parser.error("--crops-only needs --crops-output")
    if not args.crops_only and not args.output:
        parser.error("--output is required (or use --crops-only with --crops-output)")

    files = sorted(
        p for p in args.input.expanduser().iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        print(f"No images found in {args.input}", file=sys.stderr)
        return 2
    if not args.crops_only:
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
    if args.crops_output:
        write_crops(files, args.crops_output, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
