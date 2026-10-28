# Small fine-tuned models vs. general-purpose LLMs on Fireworks AI

**Southworks × Fireworks AI — reproducible case study**

This repository shows, end to end, how we specialize a small open model for a narrow business task on
[Fireworks AI](https://fireworks.ai) and how we prove whether it is better than a general-purpose LLM:
data preparation, fine-tuning with `firectl`, deployment, benchmarking, and an honest reading of the results.

It contains two cases:

| Case | Task | Fine-tuned model | Compared with | Headline |
|---|---|---|---|---|
| **Amazon MRO purchasing copilot** (main) | Recommend `BUY` / `CONDITIONAL_BUY` / `DO_NOT_BUY` / `INSUFFICIENT_EVIDENCE` for an industrial product, with a grounded rationale | Qwen3-4B + LoRA on Fireworks (`ft-qwen3-4b-amazon-v2`) | GPT-4.1 mini, GPT-5, Qwen3-4B base | Best decision accuracy (54.4% vs 43.7% and 37.9%), ~20x lower median latency than GPT-5 and half its tokens; GPT-5 writes better explanations |
| **Banking77** (supporting) | Classify a customer query into 1 of 77 intents | Qwen3.5-0.8B + LoRA (local) · Qwen3-4B + LoRA (Fireworks path) | GPT-5.6 Luna, GPT-4o agent | 0.8B model: 86.9% vs 85.2% for GPT-5.6 Luna with ~35x fewer tokens |

Start with the **[executive summary](docs/EXECUTIVE_SUMMARY.md)**.

## Amazon case: main result (103 frozen, product-disjoint test cases)

| Model | Decision accuracy | Macro-F1 | Judge score (same session) | Output tokens | Median latency |
|---|---:|---:|---:|---:|---:|
| **Qwen3-4B fine-tuned v2** | **54.37%** | **0.4607** | 61.00 vs GPT-4.1 mini · 60.40 vs GPT-5 | 15,663 | **1.16 s** |
| GPT-4.1 mini | 43.69% | 0.3749 | 53.64 | 20,506 | 3.41 s |
| GPT-5 | 37.86% | 0.3330 | **63.70** | 246,273 | 25.00 s |
| Qwen3-4B base | 27.18% | 0.2333 | 42.75 · 43.12 | 11,042 | 1.35 s |

- Fine-tuning moved Qwen3-4B from 27% to 54% decision accuracy with **731 training examples, a 9-minute
  job and an estimated USD 0.86 training cost** on Fireworks.
- GPT-5 over-recommends `BUY` (52 times; the reference has 4). The fine-tuned model reproduces the
  expected decision mix far better.
- GPT-5 produces the best-grounded explanations according to a blind GPT-6 Sol judge, so the
  recommended pattern is **fine-tuned model for the decision, large model as reviewer/writer, human
  approval for purchases**. No model is fit for autonomous purchasing: every one shows critical errors in
  more than half of the cases according to the judge.

Full tables: [evaluation/qwen3-4b-amazon-v2/RESULTS.md](evaluation/qwen3-4b-amazon-v2/RESULTS.md).
The claim is deliberately scoped to this task, data, prompt and rubric.

## Run it

Two modes:

| Mode | What it does | Needs | Cost |
|---|---|---|---|
| **1 · results** | Shows the results from the published data and metrics: validates data and leakage, recomputes every metric from the frozen answers, prints the tables, side-by-side answers and the Banking77 case | Python 3.10+ | Free, ~1 min |
| **2 · processes** | Runs the real process on Fireworks and Foundry, one process at a time or all together | `firectl`, Azure CLI, credentials in `.env` | Training (~USD 1 per job) + GPU time + Foundry tokens |

```bash
make results                 # Mode 1
cp .env.example .env         # Mode 2 credentials
make help                    # list of processes
make pipeline                # Mode 2, every process in order, then undeploy
```

Mode 2 processes. Each one is a Python script driven by a config file, so it can be run, adapted or
reused on its own:

| Process | Make target | Script | Config |
|---|---|---|---|
| Deploy GPT-5 and GPT-6 Sol on Foundry (reused if they exist) | `deploy-foundry-models` | `deploy_foundry.py` | `configs/foundry/models.toml` |
| Upload datasets to Fireworks | `upload-datasets` (`upload-dataset-amazon-reviews`, `upload-dataset-banking77`) | `upload_dataset.py --name --path` | — |
| Fine-tune | `train-models` (`train-model-amazon-reviews`, `train-model-banking77`) | `train_model.py --config` | `configs/training/*.toml` |
| Deploy on Fireworks | `deploy-models` (`deploy-base-model`, `deploy-ft-models`) | `deploy_model.py --config` | `configs/deployment/*.toml` |
| Benchmark | `exec-benchmark` (`exec-benchmark-amazon-reviews`, `exec-benchmark-banking77`) | `run_benchmark.py --config` | `configs/benchmark/*.toml` |
| Delete Fireworks deployments | `undeploy` | `undeploy_model.py` | `configs/deployment/*.toml` |

Options for any target: `DRY_RUN=1` prints the exact `firectl` / `az` commands without running them;
`FORCE=1` deletes and recreates a Fireworks resource that already exists (names are fixed in the configs);
`LIMIT=10` benchmarks a sample. To re-run the benchmark on the fine-tuned models behind the published
results without training: `make deploy-base-model deploy-published-models exec-benchmark-published`.
Fresh results go to `evaluation/reruns/` (git-ignored); the published evidence is never overwritten.
Step-by-step `firectl` details: [docs/FIREWORKS_WORKFLOW.md](docs/FIREWORKS_WORKFLOW.md).

## Repository map

```text
README.md                       this page
Makefile                        Mode 1 (results) and the Mode 2 processes
.env.example                    credentials for Mode 2
configs/
  training/                     fine-tuning recipes (Amazon reviews, Banking77)
  deployment/                   Fireworks deployments (base model; fine-tuned adapters on one GPU; published adapters)
  foundry/                      Foundry deployments (GPT-5 baseline, GPT-6 Sol judge)
  benchmark/                    which models, test set, prompts and judge for each benchmark
records/fireworks/              firectl exports of the real jobs, models and datasets (signed URLs redacted)
data/
  dataset/                      Amazon SFT ledger; fireworks_v2_training.jsonl = exact file used by job o9pa8ud2
  source/                       deterministic 5,000-product selection plan (raw reviews are downloaded, not shipped)
  banking77/                    Banking77 train/validation/test in chat format
evaluation/
  qwen3-4b-amazon-v2/           frozen 103-case benchmark: test set, 4 models' answers, 2 judge sessions, metrics, RESULTS.md
  banking77/                    frozen local results + instructions for the Fireworks path
scripts/                        standard-library Python: one script per process + validation, inference, judge, metrics, reports
docs/                           executive summary, method, results, cost, limitations, reproducibility
```

## Documentation

- [Executive summary](docs/EXECUTIVE_SUMMARY.md) — the business reading in two pages
- [Fireworks workflow](docs/FIREWORKS_WORKFLOW.md) — `firectl` from dataset upload to deployment teardown
- [Fine-tuning record](docs/FINETUNING.md) — the actual jobs, hyperparameters, duration and cost
- [Dataset](docs/DATASET.md) · [Dataset generation](docs/DATASET_GENERATION.md) · [Review rules](docs/DATASET_REVIEW_RULES.md)
- [Benchmark methodology](docs/BENCHMARK_METHODOLOGY.md) · [Results](docs/RESULTS.md)
- [Cost and deployment](docs/COST_AND_DEPLOYMENT.md) · [Limitations](docs/LIMITATIONS.md) · [Reproducibility](docs/REPRODUCIBILITY.md)
- [Banking77 case](evaluation/banking77/README.md)

## Data sources and licensing

Amazon data is derived from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab);
the 533,879 raw reviews are not redistributed. Banking77 is from
[PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) (CC BY 4.0). See [NOTICE.md](NOTICE.md).
A repository-wide license has not been asserted yet.
