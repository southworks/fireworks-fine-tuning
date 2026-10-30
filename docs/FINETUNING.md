# Fine-tuning record

All values below come from `firectl ... -o json` exports of the real resources, stored in
`records/fireworks/` (time-limited signed URLs redacted). Nothing is inferred.

## Amazon purchasing copilot — `ft-qwen3-4b-amazon-v2`

| Field | Value | Source |
|---|---|---|
| Fireworks account | `southworks` | job record |
| Job | `o9pa8ud2` ("FT Qwen3 4B with Amazon reviews v2") | `sftj_amazon_v2_o9pa8ud2.json` |
| Output model | `accounts/southworks/models/ft-qwen3-4b-amazon-v2` (kind `HF_PEFT_ADDON`) | `model_ft-qwen3-4b-amazon-v2.json` (`fine_tuning_job` → `o9pa8ud2`) |
| Base model | `accounts/fireworks/models/qwen3-4b` (Qwen/Qwen3-4B, 4.41B parameters, BF16, tunable) | `model_base_qwen3-4b.json` |
| Method | LoRA, rank 8, on `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | model record |
| Training dataset | `amazon-reviews-v2-train`: 731 examples, ~1.83M tokens, chat format | `dataset_amazon-reviews-v2-train.json` |
| Local copy of that dataset | `data/dataset/fireworks_v2_training.jsonl` (SHA-256 `0640754b…8b1e11`) | checked by `scripts/validate_dataset.py` |
| Evaluation during training | none (no evaluation dataset, no auto carve-out) | job record |
| Epochs | 1 | job record |
| Learning rate / schedule | 1e-4, constant | job record |
| Batch size | 8 samples | job record |
| Max context length | 32,768 tokens | job record |
| Created → completed | 2026-10-20 19:09:16 → 19:18:16 UTC (9 min 0 s) | job record |
| Estimated cost | USD 0.86 | job record (`estimated_cost`) |

Training data label mix (731): 300 `DO_NOT_BUY`, 191 `INSUFFICIENT_EVIDENCE`, 157 `CONDITIONAL_BUY`, 83 `BUY`.

### Prompt used in training vs. in the benchmark

The training rows carry a shorter system prompt (763 characters). The benchmark sends a longer system
prompt (1,400 characters) that adds the exact output template. All four models in the benchmark received
the same long prompt. The user and assistant contents are identical to the canonical ledger; only the
system message differs.

### How the benchmark served it

- Base: `accounts/southworks/deployments/w7u134w9` (dedicated deployment of Qwen3-4B).
- Fine-tuned: `accounts/southworks/models/ft-qwen3-4b-amazon-v2#accounts/southworks/deployments/wjybzim4`
  (dedicated deployment of the fine-tuned model, created from the Fireworks console).
- Requests: `temperature=0`, `top_p=1`, `max_tokens=500`, `reasoning_effort="none"` (disables Qwen3 thinking).

## Banking77 on Fireworks

| Job | Output model | Dataset | Eval dataset | Epochs | LR | Rank | Batch | Duration | Est. cost |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `jwbjh5ed` | `ft-qwen3-4b-banking` | `banking77` (9,002) | — | 3 | 1e-4 | 8 | 8 | 2 h 07 m | USD 0.77 |
| `n3gtjgd9` | `ft-qwen3-4b-banking-validation` | `banking77` (9,002) | `banking77-validation` | 3 | 1e-4 | 8 | 8 | 2 h 25 m | USD 0.77 |

Both use Qwen3-4B and the files in `data/banking77/`, the same recipe as `configs/training/banking77.toml`.
The Banking77 results reported in this repository come from a separate local experiment; see
`evaluation/banking77/README.md`.

## Reproducing a job

`configs/training/amazon-reviews.toml` and `configs/training/banking77.toml` hold exactly these recipes
under new, fixed names (`ft-qwen3-4b-amazon-reviews`, `ft-qwen3-4b-banking77`), so a re-run can never
overwrite the published models. `make upload-datasets train-models` runs them
(`make train-models DRY_RUN=1` prints the `firectl sftj create` commands first); when a job finishes, its
job and model records are saved next to these ones in `records/fireworks/`.
A new run will not produce bit-identical weights; evaluate it with the same benchmark before comparing.

## Training curves

The loss curve of every job is in the Fireworks console (job page) and downloadable through the
`metrics_file_signed_url` returned by `firectl sftj get <job> -o json` (a time-limited link, redacted in
the stored records).
