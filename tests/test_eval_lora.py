"""Tests for training/eval_lora.py (no torch, no model)."""

from __future__ import annotations

from pathlib import Path

import pytest
from eval_lora import (
    CELL,
    HEAD,
    PAD,
    SIDE,
    ModelSpec,
    build_grid,
    main,
    parse_model,
    parse_photo,
    unique,
)
from PIL import Image


def test_parse_model_none() -> None:
    assert parse_model("none") == ModelSpec("no LoRA", None, 0.0)
    assert parse_model("NONE").path is None


def test_parse_model_default_weight() -> None:
    m = parse_model("models/loras/r8/r8.safetensors", 0.8)
    assert m == ModelSpec("r8 @ 0.8", Path("models/loras/r8/r8.safetensors"), 0.8)


def test_parse_model_explicit_weight() -> None:
    m = parse_model("a/r64.safetensors@0.6")
    assert (m.label, m.path, m.weight) == ("r64 @ 0.6", Path("a/r64.safetensors"), 0.6)


@pytest.mark.parametrize("spec", ["a.safetensors@x", "a.safetensors@0", "a.safetensors@2.5"])
def test_parse_model_rejects_bad_weight(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_model(spec)


def test_parse_photo() -> None:
    p = parse_photo("examples/acapulco.jpg:0.7")
    assert (p.path, p.strength) == (Path("examples/acapulco.jpg"), 0.7)


@pytest.mark.parametrize("spec", ["examples/acapulco.jpg", "x.jpg:0", "x.jpg:1.5"])
def test_parse_photo_rejects_bad_spec(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_photo(spec)


def test_unique_drops_repeats_keeps_order() -> None:
    a = parse_model("m/a.safetensors@0.8")
    b = parse_model("m/b.safetensors@0.8")
    a6 = parse_model("m/a.safetensors@0.6")
    none = parse_model("none")
    assert unique([none, a, b, a, a6, none]) == [none, a, b, a6]


def test_build_grid_size_and_mixed_orientation() -> None:
    land = Image.new("RGB", (2 * CELL, CELL), "red")
    port = Image.new("RGB", (CELL, 2 * CELL), "blue")
    grid = build_grid(
        "t", ["seed 1", "seed 2"], [("r8 @ 0.8", [land, port]), ("no LoRA", [port, land])]
    )
    assert grid.size == (SIDE + 2 * (CELL + PAD), HEAD + 2 * (CELL + PAD))
    # landscape fills the cell width, portrait fills the cell height
    assert grid.getpixel((SIDE + CELL // 2, HEAD + CELL // 2)) == (255, 0, 0)
    assert grid.getpixel((SIDE + CELL + PAD + CELL // 2, HEAD + 2)) == (0, 0, 255)


def test_main_reports_missing_files_before_loading(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not found: .*nope.safetensors"):
        main(["--tag", "t", "--out-dir", str(tmp_path), "--models", "nope.safetensors"])
    assert not (tmp_path / "t").exists()


def test_main_rejects_bad_spec(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="bad LoRA weight"):
        main(["--tag", "t", "--out-dir", str(tmp_path), "--models", "a.safetensors@x"])
