"""Optional real-model smoke test. Downloads SDXL on first run (about 7 GB).

Run manually:  .venv/bin/pytest -m gpu -s
Set NEUROITBURD_SMOKE_INPUT to use your own photo instead of a synthetic image.
"""

import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageStat

from neuroitburd.config import Settings
from neuroitburd.image_utils import fit_size, load_image, save_image
from neuroitburd.params import GenerationParams
from neuroitburd.pipeline import SDXLStylizer

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")


def synthetic_image() -> Image.Image:
    img = Image.linear_gradient("L").resize((1024, 768)).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse((300, 150, 700, 600), fill=(200, 120, 80))
    draw.rectangle((100, 500, 400, 760), fill=(40, 80, 160))
    return img


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sdxl_smoke(tmp_path: Path) -> None:
    settings = replace(Settings.from_env(), fake_pipeline=False)
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    stylizer = SDXLStylizer.load(settings)
    load_s = time.perf_counter() - started

    source_path = os.environ.get("NEUROITBURD_SMOKE_INPUT")
    image = load_image(source_path) if source_path else synthetic_image()
    params = GenerationParams.create(strength=0.65, steps=30, guidance=7.0, seed=42)

    torch.cuda.synchronize()
    started = time.perf_counter()
    out = stylizer.generate(image, params)
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - started

    path = save_image(out, tmp_path / "smoke.png")
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"\nSMOKE {stylizer.describe()}\n"
        f"load {load_s:.1f}s, generate {gen_s:.1f}s ({params.effective_steps} steps), "
        f"peak allocated {peak_gb:.2f} GiB, size {out.size}, saved {path}"
    )

    assert out.size == fit_size(*image.size)
    assert max(ImageStat.Stat(out).stddev) > 5, "output is flat (black/NaN image?)"
