# Neuroitburd

SDXL image-to-image stylization inspired by the painting of Oleksandr Roitburd (MVP, work in progress).

Full documentation will be written in milestone M7. Quick dev setup:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## CLI

```bash
python -m neuroitburd.cli \
  --input examples/input.jpg \
  --output outputs/result.png \
  --strength 0.65 --steps 30 --guidance 7.0 --seed 42
```

Optional: `--prompt`, `--negative-prompt`, `--lora models/roitburd-v1.safetensors`,
`--lora-weight 0.8`, `--device auto|cuda|cpu`, `--offload none|model`. Run with `--help` for
ranges. Without `--seed` a random seed is chosen and printed. Set `NEUROITBURD_FAKE_PIPELINE=1`
to try the CLI without a GPU (cheap posterize filter instead of SDXL).

Exit codes: 0 ok, 1 unexpected error, 2 bad arguments or input, 3 model/device/LoRA error,
4 generation or out-of-memory error.

## Web API

```bash
# Real model (loads in the background after startup; GET /api/health shows progress)
.venv/bin/uvicorn neuroitburd.api:app --host 127.0.0.1 --port 8000

# No GPU needed: cheap fake stylizer, for frontend work
NEUROITBURD_FAKE_PIPELINE=1 .venv/bin/uvicorn neuroitburd.api:app --port 8000
```

**Run a single process.** The model is loaded once, in a worker thread inside the server
process, and an in-process FIFO queue makes sure only one generation runs on the GPU at a
time. Do not use `--workers N` (each worker would load its own 8 GB model and they would
compete for the GPU) or `--reload` with the real model. Jobs live in memory and are lost on
restart; files in `outputs/` are kept.

| Method | Path | Result |
|---|---|---|
| GET | `/api/health` | `status` (`starting`/`ok`/`error`), model status and error, model id, device, LoRA, GPU name and memory, queue size |
| POST | `/api/generate` | multipart `image` (JPEG/PNG/WebP, max 20 MB) plus optional `strength`, `steps`, `guidance`, `seed`, `prompt`, `negative_prompt`; returns 202 with `job_id` |
| GET | `/api/jobs/{job_id}` | `status` (`queued`/`processing`/`completed`/`failed`), `progress` (0-1), `error`, `output_url`, `params` (including the seed used), timestamps |
| GET | `/api/images/{image_id}` | the generated PNG |

```bash
curl -F image=@photo.jpg -F strength=0.65 -F seed=42 http://127.0.0.1:8000/api/generate
curl http://127.0.0.1:8000/api/jobs/<job_id>
curl -o result.png http://127.0.0.1:8000/api/images/<job_id>
```

Errors are JSON `{"detail": "..."}`: 422 invalid parameters or image (too small, corrupt,
extreme aspect ratio), 413 file too large, 415 not a JPEG/PNG/WebP image, 404 unknown job or
image, 503 queue full (`NEUROITBURD_MAX_QUEUE_SIZE`, default 20) or model failed to load.

## Web interface

Needs Node 20.19+ (`sudo apt install nodejs npm` on Ubuntu 25.10 gives 20.19.4).

```bash
# Terminal 1: backend (add NEUROITBURD_FAKE_PIPELINE=1 to work on the UI without a GPU)
.venv/bin/uvicorn neuroitburd.api:app --host 127.0.0.1 --port 8000

# Terminal 2: frontend dev server on http://127.0.0.1:5173 (forwards /api to port 8000)
cd frontend && npm install && npm run dev
```

Drop or choose a JPEG/PNG/WebP image, adjust strength, steps and seed (guidance and prompts
are under "Advanced"), press Generate, then download the PNG. The header shows whether the
model is loading, ready or failed. Set `NEUROITBURD_BACKEND_URL` to point the dev server at
a backend on another address. `npm run build` type-checks and writes a static build to
`frontend/dist/` (`npm run preview` serves it with the same `/api` forwarding).

## Measured performance (M2 smoke test, 2026-11-21)

RTX 4070 SUPER (12 GB), fp16, whole pipeline on CUDA (`NEUROITBURD_OFFLOAD=none`),
SDXL base 1.0 + `madebyollin/sdxl-vae-fp16-fix`, no LoRA:

| Metric | Value |
|---|---|
| Model load (from local cache) | 10.7 s |
| Generation, 1176x880, 30 steps at strength 0.65 (19 effective steps) | 6.2 s |
| Peak VRAM (`torch.cuda.max_memory_allocated`) | 8.05 GiB |

Reproduce with `HF_HUB_OFFLINE=1 .venv/bin/pytest -m gpu -s`.
