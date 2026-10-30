# Banking77 data (chat format)

Derived from [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) (CC BY 4.0).

| File | Rows | Use |
|---|---:|---|
| `train.jsonl` | 9,002 | Fireworks SFT training dataset |
| `validation.jsonl` | 1,001 | Fireworks evaluation dataset (`--evaluation-dataset`) |
| `test.jsonl` | 3,080 | Benchmark (`scripts/benchmark/banking77.py prepare-test`) |

Each row is `{"messages": [system, user, assistant]}` where the assistant content is the intent label.
The validation split is a stratified 10% of the original training split (seed 42). The test split is the
original test split.
