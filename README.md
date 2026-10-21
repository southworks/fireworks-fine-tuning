# Qwen3-4B fine-tuning on Amazon Reviews 2023

This repository is the reproducible evidence package for a domain-specialization experiment: can a small fine-tuned model outperform a larger general-purpose model on a narrow procurement task?

On this 103-case, product-disjoint benchmark, fine-tuned Qwen3-4B outperformed both base Qwen3-4B and GPT-4.1 mini on categorical decision accuracy, Macro-F1, Weighted-F1, reference similarity, and the overall score from a blinded LLM judge.

## Main result

| Model | Accuracy | Macro-F1 | Weighted-F1 | Judge score | Preferred |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B fine-tuned v2 | **54.37%** | **0.4607** | **0.5423** | **61.00** | **66/103 (64.08%)** |
| GPT-4.1 mini | 43.69% | 0.3749 | 0.3959 | 53.64 | 30/103 (29.13%) |
| Qwen3-4B base | 27.18% | 0.2333 | 0.1689 | 42.75 | 7/103 (6.80%) |

The claim is deliberately scoped: the fine-tuned model is better **on this task, dataset, prompt, and rubric**. It is not evidence that a 4B model is generally better than GPT-4.1 mini.

## What is included

- Canonical SFT ledger and approved train/evaluation exports.
- Deterministic product-selection plan and source manifest.
- Dataset generation and validation code.
- The exact 103 benchmark cases and all three model outputs.
- Deterministic metrics, per-case GPT-6 Sol judgments, and aggregated judge results.
- Methodology, deployment economics, limitations, and reproduction instructions.

## Quick validation

Requires Python 3.11+ and no third-party Python packages.

```bash
python3 scripts/validate_dataset.py
python3 scripts/benchmark.py summarize \
  --predictions \
    evaluation/qwen3-4b-amazon-v2/predictions_qwen3_4b_base.jsonl \
    evaluation/qwen3-4b-amazon-v2/predictions_qwen3_4b_finetuned.jsonl \
    evaluation/qwen3-4b-amazon-v2/predictions_gpt41mini.jsonl \
  --output /tmp/metrics.json
```

Running inference requires `FIREWORKS_API_KEY`. Dataset generation also requires the Foundry-compatible variables documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Documentation

- [Dataset](docs/DATASET.md)
- [Dataset generation](docs/DATASET_GENERATION.md)
- [Fine-tuning record](docs/FINETUNING.md)
- [Benchmark methodology](docs/BENCHMARK_METHODOLOGY.md)
- [Results](docs/RESULTS.md)
- [Cost and deployment](docs/COST_AND_DEPLOYMENT.md)
- [Limitations](docs/LIMITATIONS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)

## Safety conclusion

The result supports a human-reviewed purchasing copilot, not autonomous procurement. The automatic judge identified critical errors in 59.22% of fine-tuned outputs and at least one unsupported claim in 75.73%.

## Data source and licensing

The source is [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), published by McAuley Lab. This repository does not redistribute the 533,879 raw source reviews. Review the upstream terms and the notices in [NOTICE.md](NOTICE.md) before commercial use.
