# Reproducibility

## Mode 1 · results (offline)

```bash
make results            # = python3 scripts/results/show_results.py
```

1. `scripts/results/validate_dataset.py` and checksum verification. Checks, among others: 3-message structure and
   one canonical `Result` per row; the SHA-256 of the 731-row file used by job `o9pa8ud2`; that each of its
   rows traces to an approved training row of the ledger; zero product overlap between training, evaluation
   and the 103 benchmark cases; 103 valid answers per model and per judge session; consistency of the
   Fireworks records; no unredacted signed URLs; Banking77 split sizes and labels.
2. Recomputes decision and overlap metrics from the raw answers and byte-compares them with the published JSON.
3. Confirms `RESULTS.md` is what `scripts/benchmark/build_report.py` generates from those files.
4. Prints the results, three cases side by side, and the frozen Banking77 results.

## Mode 2 · processes

Each process is one script with explicit inputs; the Makefile only names them.

| Process | Command (what `make` runs) | Inputs |
|---|---|---|
| Upload a dataset | `python3 scripts/fireworks/upload_dataset.py --name <id> --path <jsonl>` | `FIREWORKS_API_KEY` or `firectl signin` |
| Train | `python3 scripts/fireworks/train_model.py --config configs/training/<x>.toml` | dataset uploaded |
| Deploy | `python3 scripts/fireworks/deploy_model.py --config configs/deployment/<x>.toml` | model trained |
| Benchmark | `python3 scripts/benchmark/run_benchmark.py --config configs/benchmark/<x>.toml [--limit N]` | Fireworks deployments up; Foundry resource with GPT-5 and GPT-6 Sol deployments (`FOUNDRY_BASE_URL`, `FOUNDRY_API_KEY`, `FOUNDRY_GPT5_DEPLOYMENT`, `FOUNDRY_GPT6_SOL_DEPLOYMENT`) |
| Undeploy | `python3 scripts/fireworks/undeploy_model.py --all` or `--config <x>` | — |

Common options: `--dry-run` prints the `firectl` commands without running them; `--force` deletes an
existing Fireworks resource with the same name and recreates it. Through make: `DRY_RUN=1`, `FORCE=1`, `LIMIT=n`.

### Names

Names are fixed in the configs and used as Fireworks IDs: dataset IDs, job ID = output model name, and
deployment ID = deployment `name`. A second run with the same name stops with an error; `FORCE=1` replaces the
resource (a model that is still deployed must be undeployed first). Deleting in Fireworks cannot be undone.
The training configs use names different from the published models, so the evidence models cannot be
overwritten by a re-run.

### Deployments

One deployment per model:

| Config | Serves | Make target |
|---|---|---|
| `qwen3-4b-base` | Qwen3-4B without fine-tuning (baseline for both benchmarks) | `deploy-base-model` |
| `qwen3-4b-amazon-reviews` | `ft-qwen3-4b-amazon-reviews` | `deploy-ft-model-amazon-reviews` |
| `qwen3-4b-banking77` | `ft-qwen3-4b-banking77` | `deploy-ft-model-banking77` |

All scale to zero after 10 idle minutes (`scale_to_zero_window`).

### Benchmark

`scripts/benchmark/run_benchmark.py` builds the test file (`limit` cases), then for each `[[models]]` entry generates the
answers (Fireworks or Foundry), resuming finished rows and retrying failed ones up to 3 passes; it stops if
any case is still missing. For Amazon it then runs the blind GPT-6 Sol judge (base vs fine-tuned vs GPT-5)
and writes `RESULTS.md` in the same format as the published one; for Banking77 it writes `metrics.json`.
Use the same `LIMIT` for every model of a run (the script refuses to mix sizes in one output folder).

Hosted model behaviour changes over time; keep model IDs, dates and parameters with every run.

## Rebuild the source data and generate examples

```bash
python3 scripts/dataset/build_source_subset.py          # downloads the upstream category files into data/amazon-reviews/source/raw/
export FOUNDRY_BASE_URL=https://YOUR-RESOURCE.services.ai.azure.com/openai/v1
export FOUNDRY_API_KEY=...
python3 scripts/dataset/generate_sft_dataset.py --assistant-model gpt-6-sol
```

## Checksums

After intentionally changing a published file: `python3 scripts/results/generate_checksums.py` and commit the
updated `CHECKSUMS.sha256`.
