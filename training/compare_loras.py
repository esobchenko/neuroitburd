"""Compare LoRA checkpoints on one photo: same seed, one model load, one labelled grid.

Row 1 is the photo and the app's current output without a LoRA. Each further row is one
checkpoint at each ``--weights`` value, prompted with the trigger phrase.

Usage:
    HF_HUB_OFFLINE=1 python training/compare_loras.py --image photo.jpg \
        --loras models/loras/roitburd/roitburd_step*.safetensors --output grid.jpg
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw

from neuroitburd.config import DEFAULT_PROMPT, Settings
from neuroitburd.image_utils import fit_to_model, load_image
from neuroitburd.params import GenerationParams
from neuroitburd.pipeline import SDXLStylizer

TRIGGER = "rtbrd style painting"
CELL = 512


def labelled(image: Image.Image, text: str) -> Image.Image:
    """Thumbnail of ``image`` with a caption bar."""
    thumb = image.copy()
    thumb.thumbnail((CELL, CELL))
    cell = Image.new("RGB", (CELL, CELL + 28), "white")
    cell.paste(thumb, ((CELL - thumb.width) // 2, 28))
    ImageDraw.Draw(cell).text((6, 7), text, fill="black")
    return cell


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    p = argparse.ArgumentParser(description="Compare LoRA checkpoints on one photo.")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--loras", type=Path, nargs="+", required=True)
    p.add_argument("--weights", type=float, nargs="+", default=[0.8, 1.0])
    p.add_argument("--strength", type=float, default=0.7)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    missing = [str(lora) for lora in args.loras if not lora.is_file()]
    if missing:
        p.error(f"LoRA file(s) not found: {', '.join(missing)}")

    photo = load_image(args.image)
    stylizer = SDXLStylizer.load(replace(Settings.from_env(), lora_path=None))
    pipe = stylizer._pipe  # script-only: swap adapters on the loaded pipeline

    def run(prompt: str) -> Image.Image:
        params = GenerationParams(
            strength=args.strength, steps=args.steps, seed=args.seed, prompt=prompt
        )
        t = time.perf_counter()
        out = stylizer.generate(photo, params)
        print(f"  {time.perf_counter() - t:.1f} s", flush=True)
        return out

    rows = [[labelled(fit_to_model(photo), "input"), labelled(run(DEFAULT_PROMPT), "no LoRA")]]
    prompt = f"{TRIGGER}, {DEFAULT_PROMPT}"
    for lora in args.loras:
        pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="cmp")
        row = []
        for weight in args.weights:
            print(f"{lora.name} @ {weight}", flush=True)
            pipe.set_adapters(["cmp"], adapter_weights=[weight])
            row.append(labelled(run(prompt), f"{lora.stem} @ {weight}"))
        pipe.delete_adapters("cmp")
        rows.append(row)

    cols = max(len(r) for r in rows)
    grid = Image.new("RGB", (cols * CELL, len(rows) * (CELL + 28)), "white")
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            grid.paste(cell, (x * CELL, y * (CELL + 28)))
    grid.save(args.output, quality=90)
    print(f"grid written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
