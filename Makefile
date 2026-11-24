# Neuroitburd: shortcuts for the web UI, LoRA training and one-off stylizing.
# Run `make` (or `make help`) for the list. Override any variable on the command line,
# e.g. `make start LORA_WEIGHT=1.0` or `make train NAME=r16 RANK=16 STEPS=3000`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PY       := .venv/bin/python
VITE     := frontend/node_modules/.bin/vite
RUN_DIR  := .run
TRIGGER  := rtbrd style painting
export HF_HUB_OFFLINE ?= 1

# ---- web UI ---------------------------------------------------------------------------
BACKEND_PORT  ?= 8000
FRONTEND_PORT ?= 5173
# LORA=none runs without a LoRA; FAKE=1 uses the fake model (no GPU)
LORA          ?= models/loras/r8-2k/r8-2k.safetensors
LORA_WEIGHT   ?= 0.8
FAKE          ?= 0

# ---- training -------------------------------------------------------------------------
NAME            ?= roitburd
DATA            ?= data/roitburd
OUT             ?= models/loras/$(NAME)
STEPS           ?= 2000
SAVE_EVERY      ?= 1000
RANK            ?= 8
ALPHA           ?= $(RANK)
LR              ?= 1e-4
WARMUP          ?= 100
SNR_GAMMA       ?= 5.0
CAPTION_DROPOUT ?= 0.05
SEED            ?= 0
NO_FLIP         ?= 0
MAX_T           ?= 1000
NOISE_OFFSET    ?= 0
TARGETS         ?= attn
# extra dataset folders trained together with DATA, e.g. EXTRA_DATA=$(CROPS)
EXTRA_DATA      ?=
CROPS           ?= data/roitburd-detail
# file-name patterns that get at most one crop (a big series would dominate the crops)
CROPS_SINGLE    ?= Gentleman
FORCE           ?= 0
# paintings folder for `make prepare`
SRC             ?=

# ---- compare / stylize ----------------------------------------------------------------
IMG       ?= examples/acapulco.jpg
WEIGHTS   ?= 0.8 1.0
STRENGTH  ?= 0.65
GEN_STEPS ?= 30
GEN_SEED  ?= 42

# ---- eval: fixed photos x seeds (see training/eval_lora.py) ---------------------------
# models: none | path | path@weight; the reference row is r8-2k, the current best
EVAL_MODELS ?= none models/loras/r8-2k/r8-2k.safetensors $(OUT)/$(NAME).safetensors
EVAL_SEEDS  ?= 42 7 1234 2026
EVAL_TAG    ?= $(NAME)

LORA_ON := $(filter-out none,$(strip $(LORA)))

# Refuse GPU work while the web UI (and its loaded model) holds the GPU.
define require_gpu_free
	@if [ -f $(RUN_DIR)/backend.pid ] && kill -0 $$(cat $(RUN_DIR)/backend.pid) 2>/dev/null; then \
	  echo "The web UI is running and holds GPU memory; run 'make stop' first"; exit 1; fi
endef

.PHONY: help start stop restart status logs train prepare crops eval compare stylize check

help: ## Show this help
	@echo "Usage: make <target> [VAR=value ...]"
	@echo
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-9s %s\n", $$1, $$2}'
	@echo
	@echo "UI:       LORA=$(LORA_ON) LORA_WEIGHT=$(LORA_WEIGHT) FAKE=$(FAKE)"
	@echo "          BACKEND_PORT=$(BACKEND_PORT) FRONTEND_PORT=$(FRONTEND_PORT)"
	@echo "Training: NAME=$(NAME) DATA=$(DATA) OUT=$(OUT)"
	@echo "          STEPS=$(STEPS) SAVE_EVERY=$(SAVE_EVERY) RANK=$(RANK) ALPHA=$(ALPHA) LR=$(LR)"
	@echo "          WARMUP=$(WARMUP) SNR_GAMMA=$(SNR_GAMMA) CAPTION_DROPOUT=$(CAPTION_DROPOUT)"
	@echo "          SEED=$(SEED) NO_FLIP=$(NO_FLIP) FORCE=$(FORCE)"
	@echo "          MAX_T=$(MAX_T) NOISE_OFFSET=$(NOISE_OFFSET) TARGETS=$(TARGETS) (attn|attn-ff|attn-ff-conv)"
	@echo "          EXTRA_DATA=$(EXTRA_DATA) CROPS=$(CROPS)"
	@echo "Stylize:  IMG=$(IMG) STRENGTH=$(STRENGTH) GEN_STEPS=$(GEN_STEPS) GEN_SEED=$(GEN_SEED)"
	@echo "Compare:  WEIGHTS='$(WEIGHTS)'"
	@echo "Eval:     EVAL_TAG=$(EVAL_TAG) EVAL_SEEDS='$(EVAL_SEEDS)'"
	@echo "          EVAL_MODELS='$(EVAL_MODELS)'"

start: ## Start backend + frontend in the background (LORA=, LORA_WEIGHT=, FAKE=1)
	@mkdir -p $(RUN_DIR)
	@for s in backend frontend; do \
	  if [ -f $(RUN_DIR)/$$s.pid ] && kill -0 $$(cat $(RUN_DIR)/$$s.pid) 2>/dev/null; then \
	    echo "$$s already running (pid $$(cat $(RUN_DIR)/$$s.pid)); use 'make restart'"; exit 1; fi; \
	done
	@if [ -n "$(LORA_ON)" ] && [ "$(FAKE)" != 1 ] && [ ! -f "$(LORA_ON)" ]; then \
	  echo "LoRA not found: $(LORA_ON) (train one with 'make train' or pass LORA=none)"; exit 1; fi
	@[ -x $(VITE) ] || (cd frontend && npm ci)
	@prompt="$$($(PY) -c 'from neuroitburd.config import DEFAULT_PROMPT; print(DEFAULT_PROMPT)')"; \
	if [ -n "$(LORA_ON)" ] && [ "$(FAKE)" != 1 ]; then \
	  export NEUROITBURD_LORA_PATH="$(LORA_ON)" NEUROITBURD_LORA_WEIGHT="$(LORA_WEIGHT)" \
	         NEUROITBURD_DEFAULT_PROMPT="$(TRIGGER), $$prompt"; \
	fi; \
	NEUROITBURD_FAKE_PIPELINE=$(FAKE) setsid nohup .venv/bin/uvicorn neuroitburd.api:app \
	  --host 127.0.0.1 --port $(BACKEND_PORT) > $(RUN_DIR)/backend.log 2>&1 < /dev/null & \
	echo $$! > $(RUN_DIR)/backend.pid
	@NEUROITBURD_BACKEND_URL=http://127.0.0.1:$(BACKEND_PORT) setsid nohup $(VITE) frontend \
	  --host 127.0.0.1 --port $(FRONTEND_PORT) --strictPort \
	  > $(RUN_DIR)/frontend.log 2>&1 < /dev/null & echo $$! > $(RUN_DIR)/frontend.pid
	@echo -n "waiting for the backend"; \
	for i in $$(seq 60); do \
	  curl -sf http://127.0.0.1:$(BACKEND_PORT)/api/health >/dev/null && break; \
	  kill -0 $$(cat $(RUN_DIR)/backend.pid) 2>/dev/null || { echo; echo "backend exited:"; \
	    tail -20 $(RUN_DIR)/backend.log; $(MAKE) -s stop; exit 1; }; \
	  echo -n "."; sleep 1; \
	done; echo
	@echo "UI:      http://127.0.0.1:$(FRONTEND_PORT)"
	@echo "Model:   $(if $(filter 1,$(FAKE)),fake (no GPU),SDXL$(if $(LORA_ON), + $(LORA_ON) @ $(LORA_WEIGHT),, no LoRA))"
	@echo "Model loading continues in the background; 'make status' shows when it is ready."
	@echo "Logs:    make logs     Stop: make stop"

stop: ## Stop the backend and frontend
	@for s in frontend backend; do \
	  f=$(RUN_DIR)/$$s.pid; \
	  if [ -f $$f ] && kill -0 $$(cat $$f) 2>/dev/null; then \
	    kill -TERM -- -$$(cat $$f) 2>/dev/null || kill -TERM $$(cat $$f); \
	    for i in $$(seq 20); do kill -0 $$(cat $$f) 2>/dev/null || break; sleep 0.5; done; \
	    kill -0 $$(cat $$f) 2>/dev/null && kill -KILL -- -$$(cat $$f) 2>/dev/null; \
	    echo "stopped $$s"; \
	  else echo "$$s not running"; fi; \
	  rm -f $$f; \
	done

restart: stop start ## Stop, then start again (same variables as start)

status: ## Show whether the UI is running and the model state
	@for s in backend frontend; do \
	  f=$(RUN_DIR)/$$s.pid; \
	  if [ -f $$f ] && kill -0 $$(cat $$f) 2>/dev/null; then echo "$$s: running (pid $$(cat $$f))"; \
	  else echo "$$s: stopped"; fi; \
	done
	@curl -sf http://127.0.0.1:$(BACKEND_PORT)/api/health && echo || true

logs: ## Follow the backend and frontend logs (Ctrl-C to quit)
	@tail -n 30 -F $(RUN_DIR)/backend.log $(RUN_DIR)/frontend.log

prepare: ## Rebuild the training set from a paintings folder (SRC=..., DATA=...)
	@[ -n "$(strip $(SRC))" ] || { echo "set SRC=<folder with the paintings>"; exit 1; }
	@if [ -d "$(DATA)" ] && [ "$(FORCE)" != 1 ]; then \
	  echo "$(DATA) exists; pass FORCE=1 to rebuild it"; exit 1; fi
	rm -rf "$(DATA)"
	$(PY) training/prepare_dataset.py --input "$(strip $(SRC))" --output "$(DATA)"

crops: ## Cut native-resolution detail crops from SRC into CROPS (FORCE=1 to rebuild)
	@[ -n "$(strip $(SRC))" ] || { echo "set SRC=<folder with the paintings>"; exit 1; }
	@if [ -d "$(CROPS)" ] && [ "$(FORCE)" != 1 ]; then \
	  echo "$(CROPS) exists; pass FORCE=1 to rebuild it"; exit 1; fi
	rm -rf "$(CROPS)"
	$(PY) training/prepare_dataset.py --input "$(strip $(SRC))" --crops-only --crops-output "$(CROPS)" \
	  $(foreach p,$(CROPS_SINGLE),--single-crop $(p))

train: ## Train a LoRA (NAME=, STEPS=, RANK=, LR=, ...; see 'make help')
	@if [ -e "$(OUT)/$(NAME).safetensors" ] && [ "$(FORCE)" != 1 ]; then \
	  echo "$(OUT)/$(NAME).safetensors exists; choose another NAME= or pass FORCE=1"; exit 1; fi
	$(require_gpu_free)
	@mkdir -p "$(OUT)"
	$(PY) training/train_lora.py --data "$(DATA)" $(EXTRA_DATA) --output "$(OUT)" --name "$(NAME)" \
	  --steps $(STEPS) --save-every $(SAVE_EVERY) --rank $(RANK) --alpha $(ALPHA) --lr $(LR) \
	  --warmup $(WARMUP) --snr-gamma $(SNR_GAMMA) --caption-dropout $(CAPTION_DROPOUT) \
	  --seed $(SEED) --max-timestep $(MAX_T) --noise-offset $(NOISE_OFFSET) --targets $(TARGETS) \
	  $(if $(filter 1,$(NO_FLIP)),--no-flip) 2>&1 | tee "$(OUT)/train.log"
	@echo "Next: make eval NAME=$(NAME)   or   make start LORA=$(OUT)/$(NAME).safetensors"

eval: ## Fair test: EVAL_MODELS x 2 photos x 4 seeds -> outputs/eval/<EVAL_TAG>/
	$(require_gpu_free)
	$(PY) training/eval_lora.py --tag "$(EVAL_TAG)" --models $(EVAL_MODELS) --seeds $(EVAL_SEEDS) \
	  --weight $(LORA_WEIGHT) --steps $(GEN_STEPS)

compare: ## Grid of NAME's checkpoints on IMG vs no LoRA -> outputs/<NAME>_grid.jpg
	$(require_gpu_free)
	$(PY) training/compare_loras.py --image "$(IMG)" --weights $(WEIGHTS) \
	  --strength $(STRENGTH) --steps $(GEN_STEPS) --seed $(GEN_SEED) \
	  --loras $$(ls "$(OUT)"/$(NAME)_step*.safetensors | sort -V) \
	  --output outputs/$(NAME)_grid.jpg

stylize: ## Stylize one image with LORA -> outputs/<image>_<NAME>.png (IMG=, STRENGTH=)
	$(require_gpu_free)
	@prompt="$$($(PY) -c 'from neuroitburd.config import DEFAULT_PROMPT; print(DEFAULT_PROMPT)')"; \
	out=outputs/$$(basename "$(IMG)" | sed 's/\.[^.]*$$//')_$(NAME)_s$(STRENGTH).png; \
	$(PY) -m neuroitburd.cli --input "$(IMG)" --output "$$out" --strength $(STRENGTH) \
	  --steps $(GEN_STEPS) --seed $(GEN_SEED) \
	  $(if $(LORA_ON),--lora "$(LORA_ON)" --lora-weight $(LORA_WEIGHT) --prompt "$(TRIGGER), $$prompt")

check: ## Run tests, lint, format check and type check
	.venv/bin/pytest -q
	.venv/bin/ruff check . && .venv/bin/ruff format --check .
	.venv/bin/mypy
