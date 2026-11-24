"""Fair LoRA evaluation: every model on the same photos x the same seeds, one grid per photo.

A single seed can mislead (seed changes the picture more than the LoRA does), so each model
is rendered on a fixed evaluation set: two reference photos, each at its own img2img
strength, times four seeds. Rows are models, columns are seeds.

Model specs: ``none`` (SDXL without a LoRA, default prompt), ``path.safetensors`` (weight
``--weight``) or ``path.safetensors@0.6``. LoRA rows are prompted with the trigger phrase.

Writes to ``<out-dir>/<tag>/``: ``<photo>_grid.jpg`` per photo, the full-size images in
``<photo>/``, and ``settings.json``.

Usage:
    HF_HUB_OFFLINE=1 python training/eval_lora.py --tag r8-vs-r64 \
        --models none models/loras/r8/r8.safetensors models/loras/r64/r64.safetensors@0.6
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TRIGGER = "rtbrd style painting"
DEFAULT_PHOTOS = ["examples/acapulco.jpg:0.7", "examples/picasso.jpg:0.5"]
DEFAULT_SEEDS = [42, 7, 1234, 2026]
DEFAULT_WEIGHT = 0.8
CELL = 400
PAD = 6
HEAD = 44
SIDE = 230


@dataclass(frozen=True)
class ModelSpec:
    """One grid row: a LoRA file at a weight, or the plain model (``path is None``)."""

    label: str
    path: Path | None
    weight: float


@dataclass(frozen=True)
class PhotoSpec:
    """One evaluation photo and the img2img strength it is rendered at."""

    path: Path
    strength: float


def parse_model(spec: str, default_weight: float = DEFAULT_WEIGHT) -> ModelSpec:
    """Parse ``none``, ``path`` or ``path@weight`` into a :class:`ModelSpec`."""
    if spec.lower() == "none":
        return ModelSpec("no LoRA", None, 0.0)
    path_str, sep, weight_str = spec.rpartition("@")
    if not sep:
        path_str, weight = spec, default_weight
    else:
        try:
            weight = float(weight_str)
        except ValueError as exc:
            raise ValueError(f"bad LoRA weight in {spec!r}") from exc
    if not 0.0 < weight <= 2.0:
        raise ValueError(f"LoRA weight must be in (0, 2], got {weight} in {spec!r}")
    path = Path(path_str)
    return ModelSpec(f"{path.stem} @ {weight:g}", path, weight)


def parse_photo(spec: str) -> PhotoSpec:
    """Parse ``path:strength`` (strength in (0, 1]) into a :class:`PhotoSpec`."""
    match = re.fullmatch(r"(.+):([0-9.]+)", spec)
    if not match:
        raise ValueError(f"photo must be given as path:strength, got {spec!r}")
    strength = float(match.group(2))
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"strength must be in (0, 1], got {strength} in {spec!r}")
    return PhotoSpec(Path(match.group(1)), strength)


def unique(models: list[ModelSpec]) -> list[ModelSpec]:
    """Drop repeated rows (same file and weight), keeping the first occurrence."""
    seen: set[tuple[Path | None, float]] = set()
    out = []
    for m in models:
        key = (m.path.resolve() if m.path else None, m.weight)
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def build_grid(
    title: str, col_labels: list[str], rows: list[tuple[str, list[Image.Image]]]
) -> Image.Image:
    """Lay out labelled rows of images (each fitted into a square cell) under column labels."""
    cols = len(col_labels)
    grid = Image.new("RGB", (SIDE + cols * (CELL + PAD), HEAD + len(rows) * (CELL + PAD)), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=17)
    draw.multiline_text((10, 4), title, "dimgray", small, spacing=2)
    for c, label in enumerate(col_labels):
        draw.text((SIDE + c * (CELL + PAD) + 8, 12), label, "black", font)
    for r, (label, images) in enumerate(rows):
        y = HEAD + r * (CELL + PAD)
        name, _, weight = label.partition(" @ ")
        draw.text((10, y + CELL // 2 - 26), name, "black", font)
        if weight:
            draw.text((10, y + CELL // 2 + 4), f"weight {weight}", "dimgray", small)
        for c, image in enumerate(images):
            thumb = image.copy()
            thumb.thumbnail((CELL, CELL), Image.Resampling.LANCZOS)
            x = SIDE + c * (CELL + PAD)
            grid.paste(thumb, (x + (CELL - thumb.width) // 2, y + (CELL - thumb.height) // 2))
    return grid


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Evaluate LoRAs on fixed photos x seeds.")
    p.add_argument("--models", nargs="+", required=True, help="none | path | path@weight")
    p.add_argument("--photos", nargs="+", default=DEFAULT_PHOTOS, help="path:strength")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--weight", type=float, default=DEFAULT_WEIGHT, help="default LoRA weight")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--tag", required=True, help="name of this evaluation (output folder)")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/eval"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = parse_args(argv)
    try:
        models = unique([parse_model(s, args.weight) for s in args.models])
        photos = [parse_photo(s) for s in args.photos]
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    missing = [str(m.path) for m in models if m.path and not m.path.is_file()]
    missing += [str(p.path) for p in photos if not p.path.is_file()]
    if missing:
        raise SystemExit(f"error: not found: {', '.join(missing)}")

    from neuroitburd.config import DEFAULT_PROMPT, Settings
    from neuroitburd.image_utils import load_image
    from neuroitburd.params import GenerationParams
    from neuroitburd.pipeline import SDXLStylizer

    out = args.out_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)
    sources = {p.path: load_image(p.path) for p in photos}
    stylizer = SDXLStylizer.load(replace(Settings.from_env(), lora_path=None))
    pipe = stylizer._pipe  # script-only: swap adapters on the loaded pipeline

    results: dict[tuple[Path, str], list[Image.Image]] = {}
    t0 = time.perf_counter()
    for m in models:
        if m.path:
            pipe.load_lora_weights(str(m.path.parent), weight_name=m.path.name, adapter_name="e")
            pipe.set_adapters(["e"], adapter_weights=[m.weight])
            prompt = f"{TRIGGER}, {DEFAULT_PROMPT}"
        else:
            prompt = DEFAULT_PROMPT
        for p in photos:
            (out / p.path.stem).mkdir(exist_ok=True)
            row = []
            for seed in args.seeds:
                params = GenerationParams(
                    strength=p.strength, steps=args.steps, seed=seed, prompt=prompt
                )
                image = stylizer.generate(sources[p.path], params)
                stem = m.path.stem if m.path else "none"
                image.save(out / p.path.stem / f"{stem}_w{m.weight:g}_s{seed}.png")
                row.append(image)
            results[(p.path, m.label)] = row
            print(f"{m.label:<28} {p.path.name:<14} done", flush=True)
        if m.path:
            pipe.delete_adapters("e")

    seed_labels = [f"seed {s}" for s in args.seeds]
    for p in photos:
        title = f"{p.path.name}\nstrength {p.strength:g} · {args.steps} steps"
        rows = [(m.label, results[(p.path, m.label)]) for m in models]
        path = out / f"{p.path.stem}_grid.jpg"
        build_grid(title, seed_labels, rows).save(path, quality=90)
        print(f"grid written to {path}")
    settings = {
        "models": [{"label": m.label, "path": str(m.path), "weight": m.weight} for m in models],
        "photos": [{"path": str(p.path), "strength": p.strength} for p in photos],
        "seeds": args.seeds,
        "steps": args.steps,
        "minutes": round((time.perf_counter() - t0) / 60, 1),
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
