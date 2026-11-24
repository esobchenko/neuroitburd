"""Train a style LoRA for SDXL 1.0 (UNet only) on a prepared dataset.

The dataset folder holds ``<name>.png`` + ``<name>.txt`` pairs (see ``prepare_dataset.py``).
To fit a 12 GB GPU, the trainer works in stages:

1. encode every caption with both text encoders, then free them;
2. encode every image (and its mirror image) with the VAE, then free it;
3. train LoRA adapters on UNet layers (``--targets``: attention only by default). The UNet
   stays frozen in bf16, the adapters train in fp32, with gradient checkpointing and min-SNR
   loss weighting.

Options aimed at img2img use: ``--max-timestep`` trains only on the noise levels img2img
reaches (strength 0.7 starts at about t=700 of 1000, so higher noise levels, where layout
and subject matter form, are never used); ``--noise-offset`` lets the model learn very dark
and very bright images. Several ``--data`` folders can be combined (e.g. paintings + crops).

Checkpoints are written as ``<output>/<name>_step<N>.safetensors`` (diffusers LoRA format,
loadable with ``NEUROITBURD_LORA_PATH`` or ``--lora``), plus ``<name>.safetensors`` at the end.

Usage:
    HF_HUB_OFFLINE=1 python training/train_lora.py --data data/roitburd \
        --output models/loras/roitburd
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
NUM_TRAIN_TIMESTEPS = 1000
_ATTN = ["to_q", "to_k", "to_v", "to_out.0"]
_FF = ["ff.net.0.proj", "ff.net.2", "proj_in", "proj_out"]
_CONV = ["conv1", "conv2"]  # ResNet block convolutions (texture, local detail)
TARGETS = {"attn": _ATTN, "attn-ff": _ATTN + _FF, "attn-ff-conv": _ATTN + _FF + _CONV}


@dataclass
class Sample:
    """One training image, pre-encoded."""

    name: str
    latent_mean: list[Any]  # [plain, mirrored] latent distribution means (CPU tensors)
    latent_std: list[Any]
    prompt_embeds: Any
    pooled_embeds: Any
    height: int
    width: int


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train an SDXL style LoRA (UNet only).")
    p.add_argument("--data", type=Path, nargs="+", required=True, help="one or more folders")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="roitburd")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--vae", default=VAE_MODEL)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--snr-gamma", type=float, default=5.0)
    p.add_argument("--caption-dropout", type=float, default=0.05)
    p.add_argument("--no-flip", action="store_true", help="do not train on mirrored images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-timestep",
        type=int,
        default=NUM_TRAIN_TIMESTEPS,
        help="train on noise levels below this (img2img strength s uses about s*1000)",
    )
    p.add_argument("--noise-offset", type=float, default=0.0, help="e.g. 0.05; 0 = off")
    p.add_argument("--targets", choices=sorted(TARGETS), default="attn", help="layers to adapt")
    args = p.parse_args(argv)
    if not 1 <= args.max_timestep <= NUM_TRAIN_TIMESTEPS:
        p.error(f"--max-timestep must be between 1 and {NUM_TRAIN_TIMESTEPS}")
    if not 0.0 <= args.noise_offset <= 0.5:
        p.error("--noise-offset must be between 0 and 0.5")
    return args


def load_pairs(folders: list[Path]) -> list[tuple[Path, str]]:
    """Return (image path, caption) pairs from one or more dataset folders."""
    pairs = []
    for data in folders:
        found = []
        for img in sorted(data.glob("*.png")):
            txt = img.with_suffix(".txt")
            if not txt.exists():
                raise SystemExit(f"missing caption file for {img}")
            found.append((img, txt.read_text(encoding="utf-8").strip()))
        if not found:
            raise SystemExit(f"no .png/.txt pairs in {data}")
        pairs += found
    return pairs


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Learning-rate multiplier: linear warmup, then cosine decay to 0 at ``total``."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """JSON-safe copy of the arguments (paths as strings)."""

    def safe(v: Any) -> Any:
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, list):
            return [safe(x) for x in v]
        return v

    return {k: safe(v) for k, v in vars(args).items()}


def encode_prompts(args: argparse.Namespace, prompts: list[str], torch: Any) -> list[Any]:
    """Encode prompts with both SDXL text encoders: (prompt_embeds, pooled) per prompt."""
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    tok1 = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer_2")
    kw: dict[str, Any] = {"torch_dtype": torch.float16, "variant": "fp16"}
    te1 = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", **kw)
    te2 = CLIPTextModelWithProjection.from_pretrained(
        args.base_model, subfolder="text_encoder_2", **kw
    )
    te1.to("cuda").eval()
    te2.to("cuda").eval()
    out = []
    with torch.no_grad():
        for prompt in prompts:
            hidden = []
            pooled = None
            for tok, te in ((tok1, te1), (tok2, te2)):
                ids = tok(
                    prompt,
                    padding="max_length",
                    max_length=tok.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to("cuda")
                res = te(ids, output_hidden_states=True)
                hidden.append(res.hidden_states[-2])  # SDXL uses the penultimate layer
                if te is te2:
                    pooled = res.text_embeds
            embeds = torch.cat(hidden, dim=-1)
            out.append((embeds[0].to("cpu", torch.bfloat16), pooled[0].to("cpu", torch.bfloat16)))
    del te1, te2
    torch.cuda.empty_cache()
    return out


def encode_images(args: argparse.Namespace, paths: list[Path], torch: Any) -> list[Any]:
    """VAE-encode each image and its mirror; return (means, stds, height, width) per image."""
    import numpy as np
    from diffusers import AutoencoderKL
    from PIL import Image

    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float16).to("cuda").eval()
    scale = vae.config.scaling_factor
    out = []
    with torch.no_grad():
        for path in paths:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to("cuda", torch.float16)
            means, stds = [], []
            for variant in (x, torch.flip(x, dims=[3])):
                dist = vae.encode(variant).latent_dist
                means.append((dist.mean * scale)[0].float().cpu())
                stds.append((dist.std * scale)[0].float().cpu())
            out.append((means, stds, arr.shape[0], arr.shape[1]))
    del vae
    torch.cuda.empty_cache()
    return out


def to_peft_keys(state: dict[str, Any]) -> dict[str, Any]:
    """Rename old diffusers LoRA keys (``.lora.down/.lora.up``) to PEFT ones (``.lora_A/B``).

    A file must use one key format throughout: when it mixes both, diffusers loads it as
    PEFT and silently drops the old-format keys. ``convert_state_dict_to_diffusers`` writes
    attention layers in the old format but conv/feed-forward layers in the PEFT one, so a
    ``--targets attn-ff-conv`` LoRA saved that way lost all its attention adapters.
    """
    renamed = {}
    for key, value in state.items():
        key = key.replace(".lora.down.weight", ".lora_A.weight")
        key = key.replace(".lora.up.weight", ".lora_B.weight")
        renamed[key] = value
    return renamed


def save_lora(unet: Any, out_dir: Path, weight_name: str) -> Path:
    """Save the UNet LoRA adapters (PEFT key format, loadable by diffusers)."""
    from diffusers import StableDiffusionXLPipeline
    from peft.utils import get_peft_model_state_dict

    state = to_peft_keys(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(
        out_dir, unet_lora_layers=state, weight_name=weight_name
    )
    return out_dir / weight_name


def main(argv: list[str] | None = None) -> int:
    """Run the training."""
    args = parse_args(argv)
    import torch
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from diffusers.training_utils import cast_training_params, compute_snr
    from peft import LoraConfig

    if not torch.cuda.is_available():
        print("CUDA is required for training", file=sys.stderr)
        return 3
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.data)
    t0 = time.time()

    print(f"[1/3] encoding {len(pairs)} captions", flush=True)
    prompts = [c for _, c in pairs]
    embeds = encode_prompts(args, prompts, torch)
    empty_embeds = encode_prompts(args, [""], torch)[0]
    print(f"[2/3] encoding {len(pairs)} images", flush=True)
    latents = encode_images(args, [p for p, _ in pairs], torch)
    samples = [
        Sample(p.stem, m, s, e[0], e[1], h, w)
        for (p, _), e, (m, s, h, w) in zip(pairs, embeds, latents, strict=True)
    ]
    print(f"      encoded in {time.time() - t0:.0f} s", flush=True)

    print("[3/3] training", flush=True)
    scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model, subfolder="unet", torch_dtype=torch.bfloat16, variant="fp16"
    )
    unet.requires_grad_(False)
    unet.to("cuda")
    unet.enable_gradient_checkpointing()
    unet.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            init_lora_weights="gaussian",
            target_modules=TARGETS[args.targets],
        )
    )
    cast_training_params(unet, dtype=torch.float32)
    params = [p for p in unet.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(
        f"      LoRA rank {args.rank} on {args.targets}: {n_params / 1e6:.1f}M trainable"
        f" parameters; timesteps < {args.max_timestep}, noise offset {args.noise_offset:g}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, args.warmup, args.steps)
    )
    unet.train()

    order: list[int] = []
    losses: list[float] = []
    log: list[dict[str, float]] = []
    t_train = time.time()
    for step in range(1, args.steps + 1):
        if not order:
            order = list(range(len(samples)))
            random.shuffle(order)
        s = samples[order.pop()]
        flip = 0 if args.no_flip else random.randint(0, 1)
        mean = s.latent_mean[flip].to("cuda")
        latent = (mean + s.latent_std[flip].to("cuda") * torch.randn_like(mean)).unsqueeze(0)
        if random.random() < args.caption_dropout:
            prompt_embeds, pooled = empty_embeds
        else:
            prompt_embeds, pooled = s.prompt_embeds, s.pooled_embeds
        noise = torch.randn_like(latent)
        if args.noise_offset > 0:  # per-channel shift: teaches overall brightness changes
            noise += args.noise_offset * torch.randn(
                latent.shape[0], latent.shape[1], 1, 1, device="cuda"
            )
        t = torch.randint(0, args.max_timestep, (1,), device="cuda")
        noisy = scheduler.add_noise(latent, noise, t)
        time_ids = torch.tensor([[s.height, s.width, 0, 0, s.height, s.width]], device="cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(
                noisy.to(torch.bfloat16),
                t,
                encoder_hidden_states=prompt_embeds.unsqueeze(0).to("cuda"),
                added_cond_kwargs={
                    "text_embeds": pooled.unsqueeze(0).to("cuda"),
                    "time_ids": time_ids.to(torch.bfloat16),
                },
            ).sample
        loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
        if args.snr_gamma > 0:
            snr = compute_snr(scheduler, t)
            loss = loss * (torch.clamp(snr, max=args.snr_gamma) / snr).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        lr_sched.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())

        if step % 25 == 0 or step == args.steps:
            elapsed = time.time() - t_train
            avg = sum(losses[-25:]) / len(losses[-25:])
            vram = torch.cuda.max_memory_allocated() / 2**30
            eta = elapsed / step * (args.steps - step)
            print(
                f"step {step:>5}/{args.steps}  loss {avg:.4f}  lr {lr_sched.get_last_lr()[0]:.2e}"
                f"  {step / elapsed:.2f} it/s  peak {vram:.2f} GiB  eta {eta / 60:.0f} min",
                flush=True,
            )
            log.append({"step": step, "loss": avg, "elapsed_s": elapsed})
        if step % args.save_every == 0 or step == args.steps:
            path = save_lora(unet, args.output, f"{args.name}_step{step}.safetensors")
            print(f"      saved {path}", flush=True)

    final = save_lora(unet, args.output, f"{args.name}.safetensors")
    meta = run_metadata(args)
    meta.update(images=len(samples), minutes=(time.time() - t0) / 60, log=log)
    (args.output / f"{args.name}_train.json").write_text(json.dumps(meta, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min; final LoRA: {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
