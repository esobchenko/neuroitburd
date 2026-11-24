"""Tests for the torch-free parts of training/train_lora.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from train_lora import TARGETS, load_pairs, lr_factor, parse_args, run_metadata, to_peft_keys

BASE = ["--data", "d", "--output", "o"]


def test_defaults_match_previous_behaviour() -> None:
    args = parse_args(BASE)
    assert args.data == [Path("d")]
    assert (args.max_timestep, args.noise_offset, args.targets) == (1000, 0.0, "attn")
    # the r8-2k recipe
    assert (args.rank, args.alpha, args.lr, args.steps, args.save_every) == (8, 8, 1e-4, 2000, 1000)
    assert (args.warmup, args.snr_gamma, args.caption_dropout) == (100, 5.0, 0.05)
    assert TARGETS["attn"] == ["to_q", "to_k", "to_v", "to_out.0"]


def test_new_options_and_multiple_folders() -> None:
    args = parse_args(
        ["--data", "a", "b", "--output", "o", "--max-timestep", "750", "--noise-offset", "0.05"]
        + ["--targets", "attn-ff-conv"]
    )
    assert args.data == [Path("a"), Path("b")]
    assert (args.max_timestep, args.noise_offset) == (750, 0.05)
    assert set(TARGETS["attn"]) < set(TARGETS["attn-ff"]) < set(TARGETS[args.targets])
    assert {"conv1", "conv2"} <= set(TARGETS["attn-ff-conv"])


@pytest.mark.parametrize(
    "extra",
    [
        ["--max-timestep", "0"],
        ["--max-timestep", "1001"],
        ["--noise-offset", "-0.1"],
        ["--noise-offset", "0.9"],
        ["--targets", "everything"],
    ],
)
def test_rejects_bad_values(extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(BASE + extra)


def _pair(folder: Path, name: str, caption: str | None = "cap") -> None:
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.png").write_bytes(b"")
    if caption is not None:
        (folder / f"{name}.txt").write_text(caption + "\n")


def test_load_pairs_combines_folders(tmp_path: Path) -> None:
    _pair(tmp_path / "a", "x", "one")
    _pair(tmp_path / "b", "y", "two")
    pairs = load_pairs([tmp_path / "a", tmp_path / "b"])
    assert [(p.name, c) for p, c in pairs] == [("x.png", "one"), ("y.png", "two")]


def test_load_pairs_errors(tmp_path: Path) -> None:
    _pair(tmp_path / "a", "x", None)
    with pytest.raises(SystemExit, match="missing caption"):
        load_pairs([tmp_path / "a"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no .png/.txt pairs"):
        load_pairs([tmp_path / "empty"])


def test_lr_factor_warmup_then_cosine() -> None:
    assert lr_factor(0, 100, 2000) == pytest.approx(0.01)
    assert lr_factor(99, 100, 2000) == pytest.approx(1.0)
    assert lr_factor(100, 100, 2000) == pytest.approx(1.0)
    assert lr_factor(1050, 100, 2000) == pytest.approx(0.5)
    assert lr_factor(2000, 100, 2000) == pytest.approx(0.0)


def test_run_metadata_is_json_safe() -> None:
    meta = run_metadata(parse_args(["--data", "a", "b", "--output", "o"]))
    assert json.loads(json.dumps(meta))["data"] == ["a", "b"]
    assert meta["output"] == "o"


def test_to_peft_keys_unifies_formats() -> None:
    mixed = {
        "unet.a.attn1.to_q.lora.down.weight": 1,
        "unet.a.attn1.to_q.lora.up.weight": 2,
        "unet.a.ff.net.2.lora_A.weight": 3,
        "unet.a.ff.net.2.lora_B.weight": 4,
    }
    assert to_peft_keys(mixed) == {
        "unet.a.attn1.to_q.lora_A.weight": 1,
        "unet.a.attn1.to_q.lora_B.weight": 2,
        "unet.a.ff.net.2.lora_A.weight": 3,
        "unet.a.ff.net.2.lora_B.weight": 4,
    }


def test_is_mixed() -> None:
    from fix_lora_keys import is_mixed

    assert is_mixed(["x.lora.down.weight", "y.lora_A.weight"])
    assert not is_mixed(["x.lora.down.weight", "x.lora.up.weight"])
    assert not is_mixed(["y.lora_A.weight", "y.lora_B.weight"])
