# Fireworks AI fine-tuning case study — Southworks
#
# Mode 1  make results            results from the published data and metrics (offline, free)
# Mode 2  make <process>          each process runs on its own; `make pipeline` runs them in order
#
# Every process is a Python script driven by a config file in configs/. Options:
#   FORCE=1    delete and recreate a Fireworks resource that already exists (fixed names)
#   DRY_RUN=1  print the firectl commands without running them
#   LIMIT=n    benchmark only the first n cases
#   PY=python  Python command, if it is not called python3 on this machine
# Credentials: .env (see .env.example). The benchmarks call the GPT-5 and GPT-6 Sol deployments that
# already exist in your Microsoft Foundry resource; their names go in .env.

PY    ?= python3
FT_DEPLOY  := configs/deployment/qwen3-4b-ft-adapters.toml
FLAGS := $(if $(FORCE),--force) $(if $(DRY_RUN),--dry-run)
BENCH := $(if $(LIMIT),--limit $(LIMIT)) $(if $(DRY_RUN),--dry-run)

.PHONY: help results pipeline \
        upload-datasets upload-dataset-amazon-reviews upload-dataset-banking77 \
        train-models train-model-amazon-reviews train-model-banking77 \
        deploy-models deploy-base-model deploy-ft-models deploy-ft-model-amazon-reviews deploy-ft-model-banking77 \
        exec-benchmark exec-benchmark-amazon-reviews exec-benchmark-banking77 \
        undeploy

help:                                    ## list the processes
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-40s %s\n", $$1, $$2}'

# ---------------------------------------------------------------- Mode 1
results:                                 ## Mode 1: validate data, recompute metrics, show results
	$(PY) scripts/results/show_results.py

# ---------------------------------------------------------------- Mode 2
pipeline:                                ## Mode 2: every process in order, then undeploy (also on failure)
	$(MAKE) upload-datasets train-models deploy-models exec-benchmark; rc=$$?; $(MAKE) undeploy; exit $$rc

upload-datasets: upload-dataset-amazon-reviews upload-dataset-banking77  ## upload all datasets to Fireworks

upload-dataset-amazon-reviews:           ## Amazon reviews training set
	$(PY) scripts/fireworks/upload_dataset.py --name amazon-reviews-train --path data/dataset/fireworks_v2_training.jsonl $(FLAGS)

upload-dataset-banking77:                ## Banking77 training and evaluation sets
	$(PY) scripts/fireworks/upload_dataset.py --name banking77-train --path data/banking77/train.jsonl $(FLAGS)
	$(PY) scripts/fireworks/upload_dataset.py --name banking77-eval --path data/banking77/validation.jsonl $(FLAGS)

train-models: train-model-amazon-reviews train-model-banking77  ## fine-tune all models

train-model-amazon-reviews:              ## fine-tune Qwen3-4B on Amazon reviews (~10 min)
	$(PY) scripts/fireworks/train_model.py --config configs/training/amazon-reviews.toml $(FLAGS)

train-model-banking77:                   ## fine-tune Qwen3-4B on Banking77 (~2-2.5 h)
	$(PY) scripts/fireworks/train_model.py --config configs/training/banking77.toml $(FLAGS)

deploy-models: deploy-base-model deploy-ft-models  ## deploy the baseline and the fine-tuned models

deploy-base-model:                       ## Qwen3-4B base (benchmark baseline)
	$(PY) scripts/fireworks/deploy_model.py --config configs/deployment/qwen3-4b-base.toml $(FLAGS)

deploy-ft-models: deploy-ft-model-amazon-reviews deploy-ft-model-banking77  ## both fine-tuned models, sharing one GPU (multi-LoRA)

deploy-ft-model-amazon-reviews:          ## fine-tuned Amazon reviews model (creates or reuses the shared deployment)
	$(PY) scripts/fireworks/deploy_model.py --config $(FT_DEPLOY) --adapter ft-qwen3-4b-amazon-reviews $(FLAGS)

deploy-ft-model-banking77:               ## fine-tuned Banking77 model (creates or reuses the shared deployment)
	$(PY) scripts/fireworks/deploy_model.py --config $(FT_DEPLOY) --adapter ft-qwen3-4b-banking77 $(FLAGS)

exec-benchmark: exec-benchmark-amazon-reviews exec-benchmark-banking77  ## run all benchmarks

exec-benchmark-amazon-reviews:           ## base vs fine-tuned vs GPT-5 + blind judge (103 cases)
	$(PY) scripts/benchmark/run_benchmark.py --config configs/benchmark/amazon-reviews.toml $(BENCH)

exec-benchmark-banking77:                ## base vs fine-tuned vs GPT-5 (3,080 cases)
	$(PY) scripts/benchmark/run_benchmark.py --config configs/benchmark/banking77.toml $(BENCH)

undeploy:                                ## delete every Fireworks deployment in configs/deployment (stops GPU billing)
	$(PY) scripts/fireworks/undeploy_model.py --all $(if $(DRY_RUN),--dry-run)
