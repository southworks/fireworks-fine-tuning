# Fine-tuning record

## Published identity

- Base model: Qwen3-4B.
- Fine-tuned model label: `ft-qwen3-4b-amazon-v2`.
- Platform: Fireworks AI.
- Training data: `data/dataset/training.jsonl` (797 approved conversations).
- Task: grounded industrial purchasing recommendation.

## Reproducibility boundary

The project snapshot contains the training corpus and evaluated model identity, but it does not contain a machine-readable export of the Fireworks training job configuration or billing record. Consequently, this repository reproduces dataset construction and benchmark evaluation, but cannot independently reconstruct every training hyperparameter or prove the historical training charge.

Before using this as a formal audit package, add the immutable Fireworks job export or screenshots recording:

- base-model revision;
- training job/model ID;
- epochs or steps;
- effective batch size;
- learning rate and scheduler;
- maximum sequence length;
- checkpoint selection;
- wall-clock duration;
- billed training cost.

These fields are intentionally marked as unavailable rather than inferred.
