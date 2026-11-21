# AGENTS.md

Working agreement for coding agents (Claude Code, Codex, etc.) and humans on Neuroitburd.

- What to build: `NEUROITBURD_MVP_PROMPT.md` (the spec)
- How and in what order: `IMPLEMENTATION_PLAN.md` (milestones M0 to M7)
- Where we are: the Progress section below

If the spec and the plan conflict, follow the plan and point out the conflict. If the plan looks wrong, propose a change. Do not silently work around it.

---

## 1. Progress

Update this list at the end of every milestone. Statuses: `TODO`, `IN PROGRESS`, `DONE`, `BLOCKED`.

| Milestone | Status | Notes |
|---|---|---|
| M0 Scaffold and tooling | DONE | Packaging, `Settings`, logging, tooling in place. Full stack installed on Python 3.13: torch 2.15.0+cu132 (CUDA available, fp16 ok), diffusers 0.43.0, transformers 5.23.0, peft 0.22.0. SDXL img2img pipeline class imports. |
| M1 Image utilities and validation | DONE | `image_utils.py` (load, validate, fit, atomic save), `params.py` (`GenerationParams`), `errors.py`. 67 tests pass. `lora_weight` validated in `Settings`, not per request. |
| M2 Inference module | TODO | GPU smoke test needs approval (about 7 GB download) |
| M3 CLI | TODO | |
| M4 Job queue and worker | TODO | |
| M5 FastAPI backend | TODO | |
| M6 Frontend | BLOCKED | Node 20+ is not installed |
| M7 Documentation and end-to-end validation | TODO | |

## 2. Rules

1. **One milestone at a time.** Work on the first milestone that is not `DONE`. Start the next one only after the current one meets its acceptance criteria.
2. **Stop after each milestone.** Report (section 4) and wait for the user before continuing.
3. **Ask before heavy or irreversible actions:**
   - downloading model weights, or any download over about 500 MB (this includes the CUDA torch wheels)
   - running real GPU inference (`pytest -m gpu`, the real CLI, the web app with the real model)
   - installing system software (Node, CUDA, apt packages)
   - changing the Python interpreter or adding a dependency the plan does not list
   - `git commit`, `git push`, or rewriting history
4. **Stay in scope.** The spec's non-goals still apply: no databases, Redis, Celery, Docker, auth, cloud, ControlNet, or LoRA training.
5. **Prefer simple code.** Use plain functions and small modules, with type hints everywhere. Add an abstraction only when a second caller needs it.

## 3. Milestone loop

1. **Read** the milestone in `IMPLEMENTATION_PLAN.md`, including its dependencies and acceptance criteria.
2. **Mark** it `IN PROGRESS` in section 1.
3. **Build** the smallest change that satisfies the acceptance criteria.
4. **Test** as you go. Write the tests together with the code, not afterwards.
5. **Verify** with the full check suite (section 5) and fix everything it reports.
6. **Update** section 1: set the status to `DONE` (or `BLOCKED` with the reason) and add short notes. If you diverged from the plan (different library, changed API shape, dropped item), edit the milestone in `IMPLEMENTATION_PLAN.md` and give the reason. Update `README.md` if user-facing behaviour changed.
7. **Report** to the user, then stop.

A milestone is `DONE` only when all of these hold:

- every acceptance criterion is met, or the user agreed to defer it
- `pytest`, `ruff check .`, `ruff format --check .` and `mypy` pass (plus `npm run build` and `tsc --noEmit` for frontend work)
- section 1 is updated

## 4. Report format

```
Milestone: M<n> <name> -> DONE | BLOCKED
Implemented: <short bullets>
Files: <created / modified>
Checks: pytest <N passed>, ruff ok, mypy ok (paste any failures verbatim)
Deviations from plan: <none | what and why>
Next: M<n+1> <one line>; approval needed for: <items from rule 3, or none>
```

## 5. Commands

Run everything from the repo root with the project virtualenv.

```bash
# Setup
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # core + dev tools, no torch
.venv/bin/pip install -r requirements.txt  # full CUDA stack (ask first)

# Checks, run for every milestone
.venv/bin/pytest                           # GPU tests excluded by default
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy

# Manual runs
NEUROITBURD_FAKE_PIPELINE=1 .venv/bin/uvicorn neuroitburd.api:app --port 8000   # no GPU needed
.venv/bin/pytest -m gpu                    # real SDXL (ask first)
cd frontend && npm install && npm run dev  # needs Node 20+
```

## 6. Code conventions

Layout: the Python package is in `backend/neuroitburd/`, tests are in `tests/`, and the frontend is in `frontend/`.

Module boundaries:

| Module | Responsibility | Must not |
|---|---|---|
| `config.py` | `Settings` from `NEUROITBURD_*` env vars | import torch |
| `errors.py` | shared exception hierarchy (`NeuroitburdError` and subclasses) | import anything heavy |
| `params.py` | `GenerationParams`, the only parameter validation | import torch |
| `image_utils.py` | load, validate, resize, save images | know about the API or jobs |
| `pipeline.py` | `Stylizer` protocol, `SDXLStylizer`, `FakeStylizer`, device, LoRA, OOM handling | import FastAPI |
| `jobs.py` | FIFO queue, single worker thread, job records | import FastAPI |
| `api.py` | HTTP layer only | contain inference logic |
| `cli.py` | argparse wrapper over `pipeline` and `image_utils` | duplicate validation |

General rules:

- **Imports:** import `torch` and `diffusers` lazily, inside functions in `pipeline.py` only. Tests and fake mode must run without torch installed.
- **Validation:** `GenerationParams` (pydantic) is the only place generation parameters are validated. The CLI and the API both use it.
- **Errors:** use the custom exception classes defined in the plan. Messages shown to users must be actionable and must not include stack traces or internal paths. Log full details with `logger.exception`.
- **Logging:** use `logging.getLogger(__name__)`. Only the CLI may use `print`, for its user-facing output.
- **Paths:** never hardcode machine-specific paths. Resolve from settings or the working directory.
- **Style:** ruff with line length 100, mypy strict, and a short docstring on every public function.

## 7. Testing rules

- Automated tests never download weights, use the network, or require a GPU. Use `FakeStylizer` or monkeypatched diffusers objects instead.
- Mark real-model tests with `@pytest.mark.gpu`. They are skipped by default.
- Write test files only under `tmp_path`, never into `outputs/`.
- Concurrency tests synchronize with events or barriers, not with sleep-based timing.

## 8. GPU and runtime constraints

- The GPU is an RTX 4070 SUPER with 12 GB of VRAM. By default, run in fp16 with the whole pipeline on CUDA. Switch to `NEUROITBURD_OFFLOAD=model` when memory is tight. Never combine `.to("cuda")` with CPU offload.
- Load the model once, in the worker thread. The API runs as a single process: no `--workers N`, and no `--reload` with the real model.
- If `NEUROITBURD_DEVICE=cuda` is set and CUDA is unavailable, raise an error. Never silently fall back to CPU.

## 9. Environment notes (2026-11-21)

- The system interpreter is Python 3.13.7. Check in M0 that torch and diffusers wheels install on it.
- Node and npm are not installed, so M6 is blocked.
- The SDXL weights are not in the Hugging Face cache.
- NVIDIA driver 580 is installed. The default PyPI torch wheel (2.15.0+cu132) works with it, so no extra index is needed. The system CUDA toolkit (12.4) is not used by torch.
- The desktop holds about 1.3 GB of VRAM at idle, which leaves roughly 10.9 GB for inference.
- torchvision is not installed. transformers prints a harmless fallback warning about it, and SDXL img2img does not need it.
