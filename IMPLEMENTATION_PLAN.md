# Neuroitburd MVP — Incremental Implementation Plan

## Context

`NEUROITBURD_MVP_PROMPT.md` specifies a local SDXL img2img app (CLI + FastAPI + React) that stylizes images toward Oleksandr Roitburd's painting style, with a slot for a future LoRA. The repo currently contains **only that spec** — no code, no datasets, no git history — so there are no existing conventions to preserve.

Environment findings (checked read-only):
- GPU: RTX 4070 SUPER, 12 282 MiB, driver 580, CUDA toolkit 12.4.
- Python: **3.13.7** is the only interpreter (spec says 3.11+; recent torch/diffusers ship 3.13 wheels — verify in M0).
- **Node/npm not installed** → blocks the frontend milestone until installed.
- SDXL weights are **not** in the HF cache (~7 GB fp16 download needed for the GPU smoke test; requires your approval).
- Directory is **not a git repo**; ~450 GB free disk.

Goal: small, testable milestones; each leaves the repo in a working, tested state.

---

## Architecture (summary)

```
cli.py ─┐                       ┌─> pipeline.py (SDXL img2img, LoRA, device, OOM mapping)
        ├─> GenerationParams ───┤
api.py ─┘   (config.py)         └─> image_utils.py (load/validate/resize)
  │
  └─> jobs.py (FIFO queue.Queue + 1 worker thread; owns the pipeline; in-memory job table)
```

- **One inference interface**: `Stylizer` protocol with `generate(image, params, on_progress) -> PIL.Image`. Real `SDXLStylizer` and a `FakeStylizer` (PIL filter) implement it. CLI, jobs, and tests depend only on the protocol.
- **Lazy heavy imports**: `torch`/`diffusers` imported only inside `pipeline.py` functions, so API/jobs/tests run without torch installed.
- **Model loaded once, in the worker thread** at server start; `/api/health` reports `model_status: loading|ready|error` so the API boots instantly and all CUDA use stays on one thread.
- **Progress**: `callback_on_step_end` updates `job.progress = step/total` for the UI.
- **Packaging**: root `pyproject.toml` with `package-dir = {"" = "backend"}` so `pip install -e .` makes `python -m neuroitburd.cli` work from repo root. Extras: `[ml]` (torch, diffusers, transformers, accelerate, peft, safetensors), `[dev]` (pytest, httpx, ruff, mypy). `requirements.txt` mirrors pinned versions + torch CUDA index URL.
- **OpenCV**: listed in the spec but nothing in the MVP needs it (Pillow covers loading/resizing/EXIF). Proposal: defer until ControlNet (Canny). Flag if you want it now.

### GPU memory strategy
- fp16 weights (`variant="fp16"`, `torch_dtype=float16`), whole pipeline `.to("cuda")` by default — SDXL fp16 at ≤1024² peaks ~9–11 GB, which fits if the desktop isn't holding much VRAM.
- `NEUROITBURD_OFFLOAD=none|model` — `model` uses `enable_model_cpu_offload()` (never combined with `.to(cuda)`). Sequential offload not exposed (too slow).
- `vae.enable_tiling()` only for large outputs; VAE in fp16 via `madebyollin/sdxl-vae-fp16-fix` (configurable `NEUROITBURD_VAE_ID`) to avoid the known fp16 NaN/black-image issue without the VRAM cost of upcasting.
- Input resized to ≤ ~1024² total pixels, aspect preserved, sides multiple of 8.
- On `torch.cuda.OutOfMemoryError`: free memory (`gc.collect()`, `torch.cuda.empty_cache()`), fail that job with a clear "reduce size/steps or enable offload" message; worker keeps running.

---

## Milestones

### M0 — Scaffold & tooling
*Depends on: nothing.*
- `git init`, `.gitignore` (outputs/, models/*.safetensors, node_modules, .venv, caches).
- `pyproject.toml`, `requirements.txt`, empty package `backend/neuroitburd/__init__.py`, `tests/`, `models/.gitkeep`, `examples/`, `outputs/.gitkeep`.
- `config.py`: `Settings` dataclass from env vars (`MODEL_ID`, `DEVICE`, `OUTPUT_DIR`, `LORA_PATH`, `LORA_WEIGHT`, `OFFLOAD`, `VAE_ID`, `DEFAULT_PROMPT`, `DEFAULT_NEGATIVE_PROMPT`, `MAX_UPLOAD_MB`, `FAKE_PIPELINE`), relative paths resolved against CWD, no machine-specific paths. Logging setup helper.
- Verify torch+cu12x and diffusers install on Python 3.13 (install only — no weight download).

**Acceptance:** `pip install -e ".[dev]"` succeeds; `pytest` (config tests), `ruff check`, `mypy backend` pass; `python -c "import torch; print(torch.cuda.is_available())"` prints `True` after `[ml]` install.

### M1 — Image utilities & parameter validation
*Depends on: M0.*
- `image_utils.py`: `load_image(path|bytes)` (EXIF transpose, RGB conversion incl. RGBA/palette/grayscale, format whitelist JPEG/PNG/WebP, max source dims e.g. 8192), `fit_to_model(img, max_pixels=1024*1024, multiple=8)`, `save_image`.
- `GenerationParams` (pydantic model, shared by CLI/API): strength (0,1], steps 1–100, guidance 0–20, seed optional int ≥0, prompt/negative ≤ ~500 chars; cross-check `int(steps*strength) >= 1` (diffusers otherwise errors).
- Custom exceptions: `InvalidImageError`, `InvalidParamsError`.
- *As implemented:* exceptions live in `errors.py` (base `NeuroitburdError`, shared by every layer); `GenerationParams` lives in `params.py`. `lora_weight` (0–2) is validated in `Settings`, since it is a load-time setting, not a per-request one. `load_image` also rejects sides < 64 px and aspect ratios > 4:1 (SDXL degrades badly outside that); `save_image` writes atomically.

**Acceptance:** unit tests for aspect-ratio preservation, multiple-of-8 dims, upscaling small images, EXIF rotation, RGBA→RGB, corrupt/unsupported file rejection, all validation bounds.

### M2 — Inference module
*Depends on: M1.*
- `pipeline.py`: `resolve_device(setting)` (`auto`→cuda if available else cpu with warning; explicit `cuda` without CUDA → `DeviceUnavailableError`, no silent fallback); `SDXLStylizer.load(settings)` (`StableDiffusionXLImg2ImgPipeline.from_pretrained`, fp16 on CUDA / fp32 on CPU, VAE swap, offload mode, optional LoRA via `load_lora_weights` + `set_adapters([...], [weight])`, missing LoRA file → `LoraNotFoundError`); `generate()` with `torch.Generator` seeded, progress callback, OOM → `GpuOutOfMemoryError`; `ModelLoadError` wrapping HF/IO failures.
- `FakeStylizer`: deterministic PIL transform (posterize + color shift + small sleep per "step"), selected via `NEUROITBURD_FAKE_PIPELINE=1` — used by tests and for frontend dev without GPU.
- `tests/test_gpu_smoke.py` marked `@pytest.mark.gpu`, skipped unless `-m gpu` and CUDA present.
- *As implemented:* `Stylizer` also has `describe()` (model/device/LoRA facts for logs and `/api/health`); `create_stylizer(settings)` picks fake vs. SDXL; `generate()` fits the image to model size itself, so callers pass any loaded image. Seeds are resolved by callers via `GenerationParams.with_resolved_seed()` so the used seed can be logged and stored. The generator is CPU-side for seed reproducibility across devices and offload modes. Extras: fallback when a model repo has no fp16 variant; the LoRA path is checked before the 7 GB load; a warning is logged on an all-black output. Unit tests are split into `test_pipeline_fake.py` (no torch) and `test_pipeline_sdxl.py` (CPU torch plus a fake `diffusers` module, skipped without torch).

**Acceptance:** unit tests with monkeypatched diffusers classes verify device resolution errors, dtype choice, LoRA missing-file error, LoRA adapter call with weight, OOM mapping, seed passed to generator. **Manual:** `pytest -m gpu` produces an image; record peak VRAM (`torch.cuda.max_memory_allocated`) and time per image in README. *(Done 2026-11-21: 8.05 GiB peak, 6.2 s per image; weights were already cached, so no download was needed.)*

### M3 — CLI
*Depends on: M2.*
- `cli.py` (argparse): flags exactly as spec plus `--device`, `--offload`. Logs all params + resolved size + seed (random seed generated and logged when omitted, for reproducibility). Exit codes: 2 bad args/input, 3 model/device/LoRA error, 4 generation/OOM. Creates output dir.

**Acceptance:** tests run `main([...])` with `FakeStylizer` injected → output file exists, correct size; missing input → exit 2 with message; manual GPU run of the spec's example command.

### M4 — Job queue & worker
*Depends on: M2 (protocol only; tests use fake).*
- `jobs.py`: `Job` dataclass (id uuid4, status, progress, params, input/output paths, error, timestamps); `JobManager(stylizer_factory, settings)` with `queue.Queue`, `threading.Lock`-protected dict, one daemon worker thread that loads the model then loops; `submit()`, `get()`, `start()`, `stop()` (sentinel). Per-job try/except → `failed` with user-safe message; internals logged only. Uploaded input saved to `outputs/uploads/{id}.png`, result to `outputs/{id}.png`.
- If model load fails: all queued/new jobs fail fast with the load error; health shows `error`.

**Acceptance:** tests: queued→processing→completed transitions; jobs execute strictly one at a time in FIFO order (fake records concurrency counter, asserts max 1); a raising job is `failed` and the next job completes; progress increases; load failure path.

### M5 — FastAPI backend
*Depends on: M1, M4.*
- `api.py`: app factory `create_app(settings, stylizer_factory=None)`; lifespan starts/stops `JobManager`.
  - `GET /api/health` → `{status, model_status, model_id, device, lora, gpu: {name, total_mb, free_mb} | null, queue_size}` (GPU info via lazy torch import, null if unavailable).
  - `POST /api/generate` (multipart: `image` + form fields) → 202 `{job_id, status}`; validates size (stream with limit, default 20 MB), type by actual decode not header, dims; 422 for bad params, 413 too large, 415 unsupported.
  - `GET /api/jobs/{job_id}` → `{job_id, status, progress, error, output_url, params, created_at, ...}`; 404 unknown.
  - `GET /api/images/{image_id}` → `FileResponse`; `image_id` must parse as UUID (prevents path traversal); 404 otherwise.
  - Consistent error body `{"detail": "..."}`; no tracebacks.
- Run: `uvicorn neuroitburd.api:app --host 127.0.0.1 --port 8000` — **single process, no `--workers`, no `--reload` with the real model** (documented). Optional CORS for the Vite dev origin only (prefer Vite proxy).

**Acceptance:** `TestClient` tests with fake stylizer for every endpoint's happy path + each error status; end-to-end poll until `completed` then fetch image bytes; `curl` manual check with `FAKE_PIPELINE=1`.

### M6 — Frontend
*Depends on: M5 API contract; **requires Node 20+ installed** (e.g. via nvm).*
- Vite + React + TS + Tailwind (v4 Vite plugin). Vite proxy `/api` → `127.0.0.1:8000`.
- Files: `src/api.ts` (typed fetch + polling), `src/App.tsx`, `src/components/{Dropzone,Controls,ImagePanel,StatusBar}.tsx`. No state library, no UI kit.
- Features: drag-drop + click upload with client-side type/size check, original preview (object URL), basic controls (strength, steps, seed + randomize), collapsible advanced (guidance, prompt, negative prompt), Generate (disabled while running), status + progress bar from polling (1 s), result preview, Download link. Small note: "Style approximation — not a Roitburd-trained model."
- Dark neutral layout, side-by-side on desktop, stacked on mobile.

**Acceptance:** `npm run build` and `tsc --noEmit` pass; manual walkthrough against backend in fake mode (upload → progress → result → download; error display for invalid file and failed job).

### M7 — Documentation & end-to-end validation
*Depends on: all.*
- `README.md`: overview, requirements, venv + CUDA torch install commands, backend/frontend setup, CLI usage, running web app (two terminals), LoRA usage (`NEUROITBURD_LORA_PATH`, `--lora`, expected SDXL-compatible format, future trigger token), VRAM notes/OOM tips, known limitations, single-process constraint.
- Full real-GPU run through the web UI (with approval).

**Acceptance:** fresh-clone following README reproduces CLI + web run; all tests/lint/typecheck green.

---

## Dependency graph

```
M0 → M1 → M2 → M3
            ↘
             M4 → M5 → M6 → M7
```
M3 and M4 can proceed in parallel after M2. M6 can start with a mocked API once M5's contract is fixed.

---

## Risks & architectural concerns

| Risk | Mitigation |
|---|---|
| fp16 SDXL VAE yields NaN / black images | fp16-fix VAE by default (configurable); test smoke output not all-black |
| VRAM tight at 1024² with desktop using VRAM | measure in M2; `OFFLOAD=model` fallback; cap max pixels; OOM caught per job |
| Python 3.13 wheel compatibility (torch/diffusers/peft/xformers) | verify in M0; skip xformers (SDPA is built into torch 2.x); fall back to 3.11/3.12 venv via `uv` if needed |
| Node not installed | install Node 20 LTS before M6 |
| ~7 GB weight download | only with explicit approval; respects `HF_HOME` |
| Multiple uvicorn workers / `--reload` → model loaded twice, OOM | documented; app logs a warning if it detects >1 worker env |
| In-memory job table lost on restart; outputs grow unbounded | accepted for MVP; documented; optional cleanup later |
| Future LoRA trained on a different base / format → load failure | clear `LoraLoadError`; document SDXL-1.0 LoRA requirement; requires `peft` |
| CUDA calls from multiple threads | model loaded and used only in the worker thread |
| `steps*strength < 1` diffusers error | validated in `GenerationParams` |
| Style claims / artwork copyright for future training data | UI + README disclaimer; dataset licensing flagged for the training phase |
| Concurrency limits: no backpressure | cap queue size (e.g. 20) → 503 when full |

---

## Verification (overall)

1. `pip install -e ".[dev]"` then `pytest` (no GPU, no downloads), `ruff check .`, `mypy backend`.
2. `NEUROITBURD_FAKE_PIPELINE=1 uvicorn neuroitburd.api:app` + `cd frontend && npm run dev` → full UI flow without GPU.
3. With approval: `pip install -e ".[ml]"`, `pytest -m gpu`, the spec's CLI example, then the web UI against the real model; record VRAM peak and latency.

## Execution approach
Implement one milestone at a time; after each, run its tests/lint and report briefly before continuing. Pause for approval before (a) the model weight download / GPU runs and (b) if Python 3.13 compatibility forces a different interpreter.
