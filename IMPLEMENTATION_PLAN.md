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
- *As implemented:* `main(argv, stylizer_factory=None) -> int`; `--input` and `--output` are required (as in the spec example). Flags not given fall back to `GenerationParams` defaults and `NEUROITBURD_*` env settings; CLI flags override env. Everything cheap (params, env settings, output extension, input image, output directory) is validated before the model loads. Extra exit codes: 1 for unexpected errors (short message on stderr, traceback in the log only) and 130 on Ctrl-C. A single-line step counter goes to stderr only when it is a terminal. The output-extension check moved into `image_utils.output_format()` so the CLI and `save_image` share it.

**Acceptance:** tests run `main([...])` with `FakeStylizer` injected → output file exists, correct size; missing input → exit 2 with message; manual GPU run of the spec's example command.

### M4 — Job queue & worker
*Depends on: M2 (protocol only; tests use fake).*
- `jobs.py`: `Job` dataclass (id uuid4, status, progress, params, input/output paths, error, timestamps); `JobManager(stylizer_factory, settings)` with `queue.Queue`, `threading.Lock`-protected dict, one daemon worker thread that loads the model then loops; `submit()`, `get()`, `start()`, `stop()` (sentinel). Per-job try/except → `failed` with user-safe message; internals logged only. Uploaded input saved to `outputs/uploads/{id}.png`, result to `outputs/{id}.png`.
- If model load fails: all queued/new jobs fail fast with the load error; health shows `error`.
- *As implemented:* one `threading.Condition` is both the lock and the "state changed" signal, which gives `wait(job_id, timeout)` and `wait_for_model(timeout)` (used by tests now and by M5 for clean shutdown/health). `get()` returns copies, never the live record. The queue size cap (`max_queue_size`) is a count of queued jobs checked in `submit()`, which raises the new `QueueFullError` (M5 maps it to 503); the queue itself is unbounded so the stop sentinel never blocks. After a load failure, `submit()` raises `ModelLoadError` instead of creating a job that fails immediately. Statuses: jobs `queued/processing/completed/failed`, model `not_started/loading/ready/error`. The seed is resolved at submit so the job record shows it. The original upload is saved (not a pre-fitted copy) because `fit_size` is not idempotent: refitting a fitted image can add 8 px to a side. `stop()` lets the running job finish and fails the queued ones with a "server shut down" message.

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

- *As implemented:* `api.app = create_app()` at module level for uvicorn. Health `status` is `starting`/`ok`/`error` and also returns `model_error`; `gpu` is read by `pipeline.gpu_info()` only when torch is already imported (so it is `null` in fake mode and while the model loads, and health never triggers a slow torch import). Upload limit: a `Content-Length` check in middleware rejects big bodies before they are spooled to disk, and the file read is capped as well (a chunked upload without `Content-Length` is still spooled; acceptable for a local app). New `UnsupportedImageError` (subclass of `InvalidImageError`) separates 415 (not JPEG/PNG/WebP) from 422 (corrupt, too small, extreme aspect ratio). Empty form fields mean "use the default". 503 covers a full queue, a failed model load and shutdown. A warning is logged when `WEB_CONCURRENCY` > 1. No CORS: the Vite dev server will proxy `/api` (M6).

**Acceptance:** `TestClient` tests with fake stylizer for every endpoint's happy path + each error status; end-to-end poll until `completed` then fetch image bytes; `curl` manual check with `FAKE_PIPELINE=1`.

### M6 — Frontend
*Depends on: M5 API contract; **requires Node 20+ installed** (e.g. via nvm).*
- Vite + React + TS + Tailwind (v4 Vite plugin). Vite proxy `/api` → `127.0.0.1:8000`.
- Files: `src/api.ts` (typed fetch + polling), `src/App.tsx`, `src/components/{Dropzone,Controls,ImagePanel,StatusBar}.tsx`. No state library, no UI kit.
- Features: drag-drop + click upload with client-side type/size check, original preview (object URL), basic controls (strength, steps, seed + randomize), collapsible advanced (guidance, prompt, negative prompt), Generate (disabled while running), status + progress bar from polling (1 s), result preview, Download link. Small note: "Style approximation — not a Roitburd-trained model."
- Dark neutral layout, side-by-side on desktop, stacked on mobile.

- *As implemented:* Node 20.19.4 from the Ubuntu 25.10 archive (apt), not nvm. Packages are pinned to majors that support Node 20.19 (Vite 7, `@vitejs/plugin-react` 5, React 19, Tailwind 4, TypeScript 5.9) so a future Vite that needs Node 22 is not pulled in. `npm run build` runs `tsc --noEmit` first. The proxy target can be changed with `NEUROITBURD_BACKEND_URL` and also applies to `npm run preview`. Extras beyond the plan: a header badge from `/api/health` (loading/ready with GPU name/error/backend offline; polled while the model loads), the model-load error shown as a banner, a "Reuse seed" link after a result, and an inline SVG favicon. The manual walkthrough was scripted with headless Chromium (`puppeteer-core`, kept outside the repo) because no browser extension was available: 19 checks covering upload by drop, client and server (415) file errors, progress, result, download, the slider value reaching the API, mobile layout, and a failed job followed by a successful one (using a test backend whose stylizer fails on seed 13).

**Acceptance:** `npm run build` and `tsc --noEmit` pass; manual walkthrough against backend in fake mode (upload → progress → result → download; error display for invalid file and failed job).

### M7 — Documentation & end-to-end validation
*Depends on: all.*
- `README.md`: overview, requirements, venv + CUDA torch install commands, backend/frontend setup, CLI usage, running web app (two terminals), LoRA usage (`NEUROITBURD_LORA_PATH`, `--lora`, expected SDXL-compatible format, future trigger token), VRAM notes/OOM tips, known limitations, single-process constraint.
- Full real-GPU run through the web UI (with approval).

- *As implemented:* the fresh clone was checked with the lightweight install (no torch) and fake mode; the full CUDA install was not repeated in the clone (it is the one this repo uses for the GPU runs). The check found that mypy failed without torch installed; fixed. The real-GPU web run used headless Chromium against the real backend (6.0 s per image on the server, total VRAM in use up to 11.5 of 11.6 GiB). Open item: without the ML stack, Starlette's TestClient prints a deprecation warning recommending `httpx2` (needs approval to add to the dev extras).

**Acceptance:** fresh-clone following README reproduces CLI + web run; all tests/lint/typecheck green.

### M8 — Experimental: Roitburd style LoRA (added after the MVP, at the user's request)
*Depends on: M2 (LoRA loading). Lifts the spec's "no LoRA training" non-goal for this milestone only.*
- Dataset: the user's own folder of Roitburd paintings (30 images), kept outside the repo; `data/` and `*.safetensors` are gitignored. The paintings are copyrighted: the dataset and the resulting LoRA are for personal, non-commercial experiments and are not committed or published.
- `training/prepare_dataset.py`: crop transparent padding, flatten to RGB, fit to ~1 MP with sides in multiples of 64 (aspect kept, per-image size, no fixed bucket list), caption `"rtbrd style painting, <title from file name>"`.
- `training/train_lora.py`: own minimal trainer on the installed diffusers + peft (no new dependencies, no kohya / diffusers example scripts). Text embeddings and VAE latents (plus mirrored copies) are pre-computed and the encoders freed, then UNet-only LoRA (rank 32 on attention `to_q/k/v/out`), UNet frozen in bf16, adapters in fp32, gradient checkpointing, AdamW 1e-4 with warmup + cosine, min-SNR γ=5, 5% caption dropout, batch 1, 2000 steps, checkpoint every 500. Output in diffusers LoRA format, loaded by the existing `--lora` / `NEUROITBURD_LORA_PATH`.
- Evaluation: the existing CLI on a few real photos, same seed, each checkpoint × a couple of LoRA weights, compared with no LoRA. Prompt must include the trigger `rtbrd style painting`.

**Acceptance:** prep unit-tested (no torch); training completes on the 12 GB card; at least one checkpoint visibly moves img2img output towards the paintings' look compared to no LoRA; README documents how to train and use it.

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
