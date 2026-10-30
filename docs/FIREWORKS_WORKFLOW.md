# Fireworks workflow with firectl

This is the path we follow on [Fireworks AI](https://fireworks.ai) for a fine-tuning engagement. Each step
is a script that prints the exact `firectl` command it runs (`DRY_RUN=1` prints without running), so this
page and the scripts stay in sync. Run a new account or CLI version first with `DRY_RUN=1` and a small `LIMIT`.

Reference: [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl) ·
[supervised fine-tuning](https://docs.fireworks.ai/fine-tuning/supervised-fine-tuning) ·
[`sftj create`](https://docs.fireworks.ai/tools-sdks/firectl/commands/supervised-fine-tuning-job-create) ·
[`deployment create`](https://docs.fireworks.ai/tools-sdks/firectl/commands/deployment-create) ·
[`model load-lora`](https://docs.fireworks.ai/tools-sdks/firectl/commands/model-load-lora) ·
[deploying LoRAs](https://docs.fireworks.ai/fine-tuning/deploying-loras) ·
[multi-LoRA](https://docs.fireworks.ai/fine-tuning/multi-lora)

## 0. Install and authenticate

```bash
# install firectl for your OS: https://docs.fireworks.ai/tools-sdks/firectl/firectl
firectl signin                 # or FIREWORKS_API_KEY in .env (the scripts pass it without printing it)
firectl whoami
firectl model get -a fireworks qwen3-4b      # "tunable": true
```

## 1. Upload the dataset — `make upload-datasets`

JSONL, one `{"messages": [...]}` per line (`system`, `user`, `assistant`), 3 to 3M lines. The script checks
the file locally first.

```bash
firectl dataset create amazon-reviews-train data/dataset/fireworks_v2_training.jsonl
firectl dataset create banking77-train data/banking77/train.jsonl
firectl dataset create banking77-eval  data/banking77/validation.jsonl
```

## 2. Fine-tune — `make train-models` (config: `configs/training/*.toml`)

```bash
firectl sftj create --base-model accounts/fireworks/models/qwen3-4b \
  --dataset amazon-reviews-train --output-model ft-qwen3-4b-amazon-reviews --job-id ft-qwen3-4b-amazon-reviews \
  --epochs 1 --learning-rate 0.0001 --learning-rate-scheduler constant --lora-rank 8 \
  --batch-size-samples 8 --max-context-length 32768
firectl sftj get ft-qwen3-4b-amazon-reviews -o json      # polled until JOB_STATE_COMPLETED
firectl model get ft-qwen3-4b-amazon-reviews -o json     # saved to evaluation/reruns/training/
```

| Config key | firectl flag |
|---|---|
| `name` | `--output-model` and `--job-id` |
| `base_model`, `dataset`, `eval_dataset` | `--base-model`, `--dataset`, `--evaluation-dataset` |
| `epochs`, `learning_rate`, `lr_scheduler`, `warmup_steps` | `--epochs`, `--learning-rate`, `--learning-rate-scheduler`, `--learning-rate-warmup-steps` |
| `lora_rank`, `batch_size_samples`, `max_context_length` | `--lora-rank`, `--batch-size-samples`, `--max-context-length` |

Other useful `sftj create` options: `--warm-start-from <model>` (continue from an adapter),
`--source-job <job>` (copy a job's configuration), `--wandb ...` (training curves). The job page in the
console shows the loss curve and "Render Samples" (tokens and loss mask).

## 3. Deploy — `make deploy-models`, or per case `deploy-base-model`, `deploy-ft-model-<case>` (config: `configs/deployment/*.toml`)

Fine-tuned LoRA models are served only on **on-demand (dedicated) deployments**, billed per GPU-second.

```bash
# baseline: Qwen3-4B alone                                       (strategy = "dedicated")
firectl deployment create accounts/fireworks/models/qwen3-4b --deployment-id qwen3-4b-base \
  --deployment-shape default --min-replica-count 0 --max-replica-count 1 --scale-to-zero-window 10m --wait

# fine-tuned models: adapters on one GPU                         (strategy = "addons", multi-LoRA)
# make deploy-ft-model-amazon-reviews creates the deployment and loads its adapter;
# make deploy-ft-model-banking77 reuses the deployment and loads the second one.
firectl deployment create accounts/fireworks/models/qwen3-4b --deployment-id qwen3-4b-ft-adapters \
  --deployment-shape default --min-replica-count 0 --max-replica-count 1 --scale-to-zero-window 10m --wait --enable-addons
firectl model load-lora accounts/southworks/models/ft-qwen3-4b-amazon-reviews --deployment qwen3-4b-ft-adapters --wait
firectl model load-lora accounts/southworks/models/ft-qwen3-4b-banking77     --deployment qwen3-4b-ft-adapters --wait
```

- Multi-LoRA needs a BF16 shape; `default` lets Fireworks pick a validated one. If creation fails, list
  shapes with `firectl deployment-shape-version list --base-model accounts/fireworks/models/qwen3-4b` and set
  `shape` in the config. Multi-LoRA is available through firectl and the SDK, not the web UI.
- Deploying a fine-tuned model from the web UI creates a dedicated deployment of that model alone (no add-ons):
  the equivalent here is `strategy = "dedicated"` with `model = "<fine-tuned model>"`.
- With `min_replicas = 0` the deployment scales to zero after `scale_to_zero_window`; the first request after
  that returns HTTP 503 while it scales up (the benchmark client retries).

## 4. Query

OpenAI-compatible endpoint `https://api.fireworks.ai/inference/v1/chat/completions`:

| Target | `model` field |
|---|---|
| Base deployment | `accounts/southworks/deployments/qwen3-4b-base` |
| Adapter on a deployment | `accounts/southworks/models/ft-qwen3-4b-amazon-reviews#accounts/southworks/deployments/qwen3-4b-ft-adapters` |

For Qwen3 send `reasoning_effort: "none"` to disable thinking tokens. `make exec-benchmark` does this.

## 5. Tear down — `make undeploy`

```bash
firectl deployment delete qwen3-4b-base
firectl deployment delete qwen3-4b-ft-adapters
```
