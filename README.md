# Neuroitburd

SDXL image-to-image stylization inspired by the painting of Oleksandr Roitburd (MVP, work in progress).

Full documentation will be written in milestone M7. Quick dev setup:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Measured performance (M2 smoke test, 2026-11-21)

RTX 4070 SUPER (12 GB), fp16, whole pipeline on CUDA (`NEUROITBURD_OFFLOAD=none`),
SDXL base 1.0 + `madebyollin/sdxl-vae-fp16-fix`, no LoRA:

| Metric | Value |
|---|---|
| Model load (from local cache) | 10.7 s |
| Generation, 1176x880, 30 steps at strength 0.65 (19 effective steps) | 6.2 s |
| Peak VRAM (`torch.cuda.max_memory_allocated`) | 8.05 GiB |

Reproduce with `HF_HUB_OFFLINE=1 .venv/bin/pytest -m gpu -s`.
