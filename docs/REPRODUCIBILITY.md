# Reproducibility

## Mode 1 · results (offline)

```bash
make results            # = python3 scripts/show_results.py
```

1. `scripts/validate_dataset.py` and checksum verification. Checks, among others: 3-message structure and
   one canonical `Result` per row; the SHA-256 of the 731-row file used by job `o9pa8ud2`; that each of its
   rows traces to an approved training row of the ledger; zero product overlap between training, evaluation
   and the 103 benchmark cases; 103 valid answers per model and per judge session; consistency of the
   Fireworks records; no unredacted signed URLs; Banking77 split sizes and labels.
2. Recomputes decision and overlap metrics from the raw answers and byte-compares them with the published JSON.
3. Confirms `RESULTS.md` is what `scripts/build_report.py` generates from those files.
4. Prints the results, three cases side by side, and the frozen Banking77 results.

## Mode 2 · processes

Each process is one script with explicit inputs; the Makefile only names them.

| Process | Command (what `make` runs) | Inputs |
|---|---|---|
| Foundry models | `python3 scripts/deploy_foundry.py --config configs/foundry/models.toml` | `AZ_RESOURCE_GROUP`, `AZ_FOUNDRY_ACCOUNT`; `az login` |
| Upload a dataset | `python3 scripts/upload_dataset.py --name <id> --path <jsonl>` | `FIREWORKS_API_KEY` or `firectl signin` |
| Train | `python3 scripts/train_model.py --config configs/training/<x>.toml` | dataset uploaded |
| Deploy | `python3 scripts/deploy_model.py --config configs/deployment/<x>.toml` | model trained |
| Benchmark | `python3 scripts/run_benchmark.py --config configs/benchmark/<x>.toml [--limit N]` | deployments up; `FOUNDRY_BASE_URL`, `FOUNDRY_API_KEY` |
| Undeploy | `python3 scripts/undeploy_model.py --all` or `--config <x>` | — |

Common options: `--dry-run` prints the `firectl` / `az` commands without running them; `--force` deletes an
existing Fireworks resource with the same name and recreates it. Through make: `DRY_RUN=1`, `FORCE=1`, `LIMIT=n`.

### Names

Names are fixed in the configs and used as Fireworks IDs: dataset IDs, job ID = output model name, and
deployment ID = deployment `name`. A second run with the same name stops with an error; `FORCE=1` replaces the
resource (a model that is still deployed must be undeployed first). Deleting in Fireworks cannot be undone.
The training configs use names different from the published models, so the evidence models cannot be
overwritten by a re-run. Foundry deployments are reused when they exist (pay-per-token, nothing to reset).

### Deployments

| Config | Strategy | Serves |
|---|---|---|
| `qwen3-4b-base` | `dedicated` | Qwen3-4B without fine-tuning (baseline) |
| `qwen3-4b-ft-adapters` | `addons` | `ft-qwen3-4b-amazon-reviews` and `ft-qwen3-4b-banking77` on one GPU (multi-LoRA) |
| `qwen3-4b-ft-published` | `addons` | the published `ft-qwen3-4b-amazon-v2` and `ft-qwen3-4b-banking-validation` |

All scale to zero after 10 idle minutes (`scale_to_zero_window`). Multi-LoRA (`addons`) is created through
firectl (`deployment create --enable-addons` + `model load-lora`); the Fireworks web UI does not offer it.
A fine-tuned model can also be served alone with `strategy = "dedicated"` and `model = "<fine-tuned model>"`.

### Benchmark

`run_benchmark.py` builds the test file (`limit` cases), then for each `[[models]]` entry generates the
answers (Fireworks or Foundry), resuming finished rows and retrying failed ones up to 3 passes; it stops if
any case is still missing. For Amazon it then runs the blind GPT-6 Sol judge (base vs fine-tuned vs GPT-5)
and writes `RESULTS.md` in the same format as the published one; for Banking77 it writes `metrics.json`.
Use the same `LIMIT` for every model of a run (the script refuses to mix sizes in one output folder).

Hosted model behaviour changes over time; keep model IDs, dates and parameters with every run.

## Rebuild the source data and generate examples

```bash
python3 scripts/build_source_subset.py          # downloads the upstream category files into data/source/raw/
export FOUNDRY_BASE_URL=https://YOUR-RESOURCE.services.ai.azure.com/openai/v1
export FOUNDRY_API_KEY=...
python3 scripts/generate_sft_dataset.py --assistant-model gpt-6-sol
```

## Checksums

After intentionally changing a published file: `python3 scripts/generate_checksums.py` and commit the
updated `CHECKSUMS.sha256`.
