"""Rewrite LoRA files that mix old diffusers and PEFT key formats (see ``to_peft_keys``).

Each file is rewritten in place with PEFT keys; the original is kept as ``<file>.orig``.
Files already in a single format are left untouched.

Usage:
    python training/fix_lora_keys.py models/loras/r8-ff/*.safetensors
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from train_lora import to_peft_keys


def is_mixed(keys: list[str]) -> bool:
    """True when some keys use the old ``.lora.down/up`` names and others ``.lora_A/B``."""
    old = any(".lora.down." in k or ".lora.up." in k for k in keys)
    peft = any(".lora_A." in k or ".lora_B." in k for k in keys)
    return old and peft


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    p = argparse.ArgumentParser(description="Unify LoRA key formats in place.")
    p.add_argument("files", type=Path, nargs="+")
    args = p.parse_args(argv)
    from safetensors.torch import load_file, save_file

    for path in args.files:
        state = load_file(path)
        if not is_mixed(list(state)):
            print(f"ok (single format)  {path}")
            continue
        shutil.copy2(path, path.with_name(path.name + ".orig"))
        save_file(to_peft_keys(state), path, metadata={"format": "pt"})
        print(f"fixed ({len(state)} tensors)  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
