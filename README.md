# Neuroitburd

Turn a photo into an expressive, postmodern painting with SDXL image-to-image, inspired by
the painting of Oleksandr Roitburd. This MVP has a CLI, a FastAPI backend with a single-GPU
job queue, and a small React web interface. It is the foundation for a future
Roitburd-specific LoRA; today it uses base SDXL plus a style prompt, so results are a
**style approximation, not a Roitburd-trained model**.

## 1. Overview and system requirements

```
photo ──> image_utils (validate, fit to ~1 MP) ──> pipeline (SDXL img2img, optional LoRA) ──> PNG
             ▲                                          ▲
           cli.py                         jobs.py (FIFO queue, one worker thread)
                                                         ▲
                                     api.py (FastAPI) <── frontend/ (React, Vite)
```

| | Requirement |
|---|---|
| OS | Linux (tested on Ubuntu 25.10) |
| GPU | NVIDIA with 10 GB+ free VRAM for the default mode (tested: RTX 4070 SUPER, 12 GB); less works with `NEUROITBURD_OFFLOAD=model`. CPU-only works but takes minutes per image. |
| Driver | NVIDIA driver 580+ for the default PyPI torch wheel (CUDA 13.2) |
| Python | 3.11+ (tested on 3.13.7) |
| Node | 20.19+ for the web interface (tested on 20.19.4) |
| Disk | ~7 GB for the SDXL weights (Hugging Face cache), several GB for the Python ML stack |

The model is downloaded from Hugging Face on first use (`stabilityai/stable-diffusion-xl-base-1.0`
fp16 weights plus the `madebyollin/sdxl-vae-fp16-fix` VAE, about 7 GB, into
`~/.cache/huggingface`, or `$HF_HOME`). Nothing is downloaded during installation or by the
automated tests.

## 2. Python environment and CUDA/PyTorch

```bash
git clone <repo-url> neuroitburd && cd neuroitburd
python3 -m venv .venv

# Full install: backend + CUDA PyTorch + diffusers + dev tools (several GB download)
.venv/bin/pip install -r requirements.txt

# Check that PyTorch sees the GPU (should print: True 2.x.x+cuXXX)
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

- The default PyPI torch wheel ships its own CUDA runtime; the system CUDA toolkit is not used.
  With a driver older than 580, install torch from a matching index first, for example
  `.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128`, then run
  the `requirements.txt` install.
- Lightweight install without torch, enough for the tests, the fake pipeline and UI work:
  `.venv/bin/pip install -e ".[dev]"`.
- transformers may print a warning that torchvision is not installed. It is harmless; SDXL
  img2img does not need torchvision.

Optionally pre-download the weights so the first run starts quickly, then work offline:

```bash
.venv/bin/python -c "from neuroitburd.config import Settings; from neuroitburd.pipeline import SDXLStylizer; SDXLStylizer.load(Settings())"
export HF_HUB_OFFLINE=1   # later runs use only the local cache
```

## 3. Backend and frontend installation

Backend: done by the step above. Frontend (Node 20.19+; on Ubuntu 25.10
`sudo apt install nodejs npm` gives Node 20.19.4):

```bash
cd frontend
npm install
npm run build   # optional: type-check and production build into frontend/dist/
```

### Configuration

Everything is configured with environment variables (no config file is read):

| Variable | Default | Meaning |
|---|---|---|
| `NEUROITBURD_MODEL_ID` | `stabilityai/stable-diffusion-xl-base-1.0` | Base model (Hugging Face id or local folder) |
| `NEUROITBURD_VAE_ID` | `madebyollin/sdxl-vae-fp16-fix` | VAE override; empty uses the model's own (may give black images in fp16) |
| `NEUROITBURD_DEVICE` | `auto` | `auto`, `cuda` or `cpu`. `cuda` fails with an error if CUDA is missing; it never falls back to CPU silently |
| `NEUROITBURD_OFFLOAD` | `none` | `none` (whole pipeline on the GPU, fastest) or `model` (CPU offload, less VRAM, slower) |
| `NEUROITBURD_OUTPUT_DIR` | `outputs` | Results (`{job_id}.png`) and uploads (`uploads/`) |
| `NEUROITBURD_LORA_PATH` | empty | Optional SDXL LoRA `.safetensors` file (see section 6) |
| `NEUROITBURD_LORA_WEIGHT` | `0.8` | LoRA strength, 0 to 2 |
| `NEUROITBURD_DEFAULT_PROMPT` | expressive postmodern oil painting… | Prompt used when a request has none |
| `NEUROITBURD_DEFAULT_NEGATIVE_PROMPT` | photo, photorealistic, blurry… | Negative prompt used when a request has none |
| `NEUROITBURD_MAX_UPLOAD_MB` | `20` | Upload size limit for the API |
| `NEUROITBURD_MAX_QUEUE_SIZE` | `20` | Jobs that may wait; more get HTTP 503 |
| `NEUROITBURD_FAKE_PIPELINE` | `0` | `1` replaces SDXL with a cheap filter (no GPU, no torch); for development |
| `NEUROITBURD_LOG_LEVEL` | `INFO` | Python logging level |

## 4. CLI usage

```bash
.venv/bin/python -m neuroitburd.cli \
  --input examples/input.jpg \
  --output outputs/result.png \
  --strength 0.65 --steps 30 --guidance 7.0 --seed 42
```

Put your own photo at `examples/input.jpg` (the folder is empty in the repo) or pass any path.
The installed `neuroitburd` command is the same as `python -m neuroitburd.cli`.

| Option | Default | Meaning |
|---|---|---|
| `--input`, `--output` | required | Source image (JPEG/PNG/WebP); result path (`.png`, `.jpg`, `.jpeg` or `.webp`). Missing folders are created |
| `--strength` | 0.65 | 0 to 1: how far to move away from the photo |
| `--steps` | 30 | 1 to 100. img2img runs `steps × strength` denoising steps, which must be at least 1 |
| `--guidance` | 7.0 | 0 to 20: how strictly to follow the prompt |
| `--seed` | random | 0 to 2³²−1. Without it a random seed is chosen and printed, so any result can be reproduced |
| `--prompt`, `--negative-prompt` | configured defaults | Override the style prompts (max 500 characters) |
| `--lora`, `--lora-weight` | env settings | LoRA file and strength (see section 6) |
| `--device`, `--offload` | env settings | Same as `NEUROITBURD_DEVICE` / `NEUROITBURD_OFFLOAD` |

Every parameter, the resized target size and the seed are logged. All inputs are checked
before the model loads, so a typo fails immediately. Exit codes: 0 ok, 1 unexpected error,
2 bad arguments or input, 3 model/device/LoRA error, 4 generation or out-of-memory error.
Add `NEUROITBURD_FAKE_PIPELINE=1` to try the CLI without a GPU.

Images are resized to about 1 megapixel (SDXL's native size) with the aspect ratio kept and
both sides divisible by 8; for example a 4000x3000 photo becomes 1176x880.

## 5. Running the web application locally

Two terminals, from the repo root:

```bash
# Terminal 1: backend on http://127.0.0.1:8000 (the model loads in the background, 10-60 s)
.venv/bin/uvicorn neuroitburd.api:app --host 127.0.0.1 --port 8000

# Terminal 2: web interface on http://127.0.0.1:5173 (forwards /api to the backend)
cd frontend && npm run dev
```

Open http://127.0.0.1:5173, drop or choose an image, adjust strength, steps and seed
(guidance and prompts are under "Advanced"), press Generate, and download the PNG. The header
shows whether the model is loading, ready (with the GPU name) or failed. To work on the UI
without a GPU, start the backend with `NEUROITBURD_FAKE_PIPELINE=1`. Set
`NEUROITBURD_BACKEND_URL` to point the dev server at a backend elsewhere.

**Run exactly one backend process.** The model is loaded once, in a worker thread, and an
in-process FIFO queue runs one generation at a time on the GPU. Never use `--workers N` (each
worker would load its own 8 GB copy and they would fight over the GPU) or `--reload` with the
real model. Keep the backend on `127.0.0.1`: there is no authentication.

### REST API

| Method | Path | Result |
|---|---|---|
| GET | `/api/health` | `status` (`starting`/`ok`/`error`), model status and error, model id, device, LoRA, GPU name and memory (once the model is loaded), queue size |
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
image, 503 queue full, model failed to load, or server shutting down. A failed job reports
its error in `/api/jobs/{job_id}` and does not block later jobs.

## 6. Loading a future LoRA adapter

The pipeline already supports one LoRA adapter, applied when the model loads:

```bash
cp ~/Downloads/roitburd-v1.safetensors models/   # models/ is git-ignored

# Web app: set before starting the backend (restart to change it)
NEUROITBURD_LORA_PATH=models/roitburd-v1.safetensors NEUROITBURD_LORA_WEIGHT=0.8 \
  .venv/bin/uvicorn neuroitburd.api:app --host 127.0.0.1 --port 8000

# CLI
.venv/bin/python -m neuroitburd.cli --input photo.jpg --output outputs/r.png \
  --lora models/roitburd-v1.safetensors --lora-weight 0.8
```

- The file must be a LoRA trained for **SDXL 1.0 base**, in `.safetensors` format (the
  formats diffusers' `load_lora_weights` reads, including kohya/A1111 exports). A LoRA for
  another base model (SD 1.5, SD 3, Flux) either fails with a "could not apply LoRA" error
  or loads with no visible effect; in that case diffusers logs a warning about unmatched
  keys, so check the backend log.
- `peft` is required; it is part of `requirements.txt`.
- If the LoRA was trained with a trigger word, put it in the prompt: set
  `NEUROITBURD_DEFAULT_PROMPT="roitburd style, an expressive postmodern oil painting…"`, or
  type it in the prompt field / `--prompt`.
- Typical weights are 0.6 to 1.0; above 1 tends to over-bake. `/api/health` shows the loaded
  LoRA name and weight.
- A missing file is reported before the 7 GB model load: CLI exit code 3, or health status
  `error` with the message in the web app.

## 7. Known limitations

- **Style approximation only.** Base SDXL plus a prompt is not Roitburd's style. Flat,
  graphic images change little at strength 0.65; for photos try 0.65 to 0.8.
- **GPU memory.** The default mode peaks at about 8 GiB (see below), which fits a 12 GB card
  with ~1.3 GB used by the desktop. If generation fails with "GPU ran out of memory", close
  other GPU applications or set `NEUROITBURD_OFFLOAD=model` (slower, much less VRAM). An
  out-of-memory error fails only that job; the server keeps running.
- **Output size** is about 1 megapixel, not the original resolution. Inputs must be 64 to
  8192 px per side, aspect ratio at most 4:1, and (in the API) at most 20 MB.
- **Jobs are in memory.** Restarting the backend forgets jobs (files in `outputs/` remain),
  and there is no automatic cleanup of `outputs/` or `outputs/uploads/`. Jobs cannot be
  cancelled.
- **Single process, single GPU, local use.** No authentication, no multi-GPU, no horizontal
  scaling. Uploads without a `Content-Length` header (chunked) are buffered to a temporary
  file before the size limit is applied.
- **CPU mode** (`NEUROITBURD_DEVICE=cpu`) runs in fp32 and takes several minutes per image.
- **Node 20** reached end of upstream support in April 2026. The frontend should also work
  on Node 22 or 24 (not tested yet).

## Measured performance (2026-11-21)

RTX 4070 SUPER (12 GB), fp16, whole pipeline on CUDA (`NEUROITBURD_OFFLOAD=none`),
SDXL base 1.0 + `madebyollin/sdxl-vae-fp16-fix`, no LoRA:

| Metric | Value |
|---|---|
| Model load (from local cache, cold) | 10.7 s |
| Generation, 1176x880, 30 steps at strength 0.65 (19 effective steps) | 5.6 to 6.2 s |
| Peak VRAM (`torch.cuda.max_memory_allocated`) | 8.05 GiB |

Reproduce with `HF_HUB_OFFLINE=1 .venv/bin/pytest -m gpu -s`.

## Development

```bash
.venv/bin/pip install -e ".[dev]"   # no torch needed
.venv/bin/pytest                    # GPU tests are skipped by default; no downloads
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy
cd frontend && npx tsc --noEmit && npm run build
```

`AGENTS.md` describes the working agreement and module boundaries, and
`IMPLEMENTATION_PLAN.md` the milestones and design decisions.
