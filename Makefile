# Fireworks AI fine-tuning case study — Southworks
#
# Mode 1  make results            results from the published data and metrics (offline, free)
# Mode 2  make <process>          each process runs on its own; `make pipeline` runs them in order
#
# Every process is a Python script driven by a config file in configs/. Options:
#   FORCE=1    delete and recreate a Fireworks resource that already exists (fixed names)
#   DRY_RUN=1  print the firectl / az commands without running them
#   LIMIT=n    benchmark only the first n cases
# Credentials and accounts: .env (see .env.example).

PY    := python3
FLAGS := $(if $(FORCE),--force) $(if $(DRY_RUN),--dry-run)
BENCH := $(if $(LIMIT),--limit $(LIMIT)) $(if $(DRY_RUN),--dry-run)

.PHONY: help results pipeline \
        deploy-foundry-models \
        upload-datasets upload-dataset-amazon-reviews upload-dataset-banking77 \
        train-models train-model-amazon-reviews train-model-banking77 \
        deploy-models deploy-base-model deploy-ft-models deploy-published-models \
        exec-benchmark exec-benchmark-amazon-reviews exec-benchmark-banking77 exec-benchmark-published \
        undeploy

help:                                    ## list the processes
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-32s %s\n", $$1, $$2}'

# ---------------------------------------------------------------- Mode 1
results:                                 ## Mode 1: validate data, recompute metrics, show results
	$(PY) scripts/show_results.py

# ---------------------------------------------------------------- Mode 2
pipeline:                                ## Mode 2: every process in order, then undeploy (also on failure)
	$(MAKE) deploy-foundry-models upload-datasets train-models deploy-models exec-benchmark; rc=$$?; $(MAKE) undeploy; exit $$rc

deploy-foundry-models:                   ## GPT-5 and GPT-6 Sol on Microsoft Foundry (Azure CLI)
	$(PY) scripts/deploy_foundry.py --config configs/foundry/models.toml $(FLAGS)

upload-datasets: upload-dataset-amazon-reviews upload-dataset-banking77  ## upload all datasets to Fireworks

upload-dataset-amazon-reviews:           ## Amazon reviews training set
	$(PY) scripts/upload_dataset.py --name amazon-reviews-train --path data/dataset/fireworks_v2_training.jsonl $(FLAGS)

upload-dataset-banking77:                ## Banking77 training and evaluation sets
	$(PY) scripts/upload_dataset.py --name banking77-train --path data/banking77/train.jsonl $(FLAGS)
	$(PY) scripts/upload_dataset.py --name banking77-eval --path data/banking77/validation.jsonl $(FLAGS)

train-models: train-model-amazon-reviews train-model-banking77  ## fine-tune all models

train-model-amazon-reviews:              ## fine-tune Qwen3-4B on Amazon reviews (~10 min)
	$(PY) scripts/train_model.py --config configs/training/amazon-reviews.toml $(FLAGS)

train-model-banking77:                   ## fine-tune Qwen3-4B on Banking77 (~2-2.5 h)
	$(PY) scripts/train_model.py --config configs/training/banking77.toml $(FLAGS)

deploy-models: deploy-base-model deploy-ft-models  ## deploy the baseline and the fine-tuned models

deploy-base-model:                       ## Qwen3-4B base (benchmark baseline)
	$(PY) scripts/deploy_model.py --config configs/deployment/qwen3-4b-base.toml $(FLAGS)

deploy-ft-models:                        ## the fine-tuned models, both on one GPU (multi-LoRA)
	$(PY) scripts/deploy_model.py --config configs/deployment/qwen3-4b-ft-adapters.toml $(FLAGS)

deploy-published-models:                 ## the fine-tuned models behind the published results
	$(PY) scripts/deploy_model.py --config configs/deployment/qwen3-4b-ft-published.toml $(FLAGS)

exec-benchmark: exec-benchmark-amazon-reviews exec-benchmark-banking77  ## run all benchmarks

exec-benchmark-amazon-reviews:           ## base vs fine-tuned vs GPT-5 + blind judge (103 cases)
	$(PY) scripts/run_benchmark.py --config configs/benchmark/amazon-reviews.toml $(BENCH)

exec-benchmark-banking77:                ## base vs fine-tuned vs GPT-5 (3,080 cases)
	$(PY) scripts/run_benchmark.py --config configs/benchmark/banking77.toml $(BENCH)

exec-benchmark-published:                ## same benchmarks with the published fine-tuned models
	$(PY) scripts/run_benchmark.py --config configs/benchmark/amazon-reviews-published.toml $(BENCH)
	$(PY) scripts/run_benchmark.py --config configs/benchmark/banking77-published.toml $(BENCH)

undeploy:                                ## delete every Fireworks deployment in configs/deployment (stops GPU billing)
	$(PY) scripts/undeploy_model.py --all $(if $(DRY_RUN),--dry-run)
