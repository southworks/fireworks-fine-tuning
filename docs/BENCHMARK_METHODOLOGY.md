# Benchmark methodology

## Frozen evaluation set

The benchmark contains 103 product-level procurement cases converted from approved SFT evaluation rows. The file hash is stored in `evaluation/qwen3-4b-amazon-v2/test_evaluation_103.manifest.json`.

The canonical evaluation export contains 118 rows. The benchmark is a frozen 103-row snapshot created before the later canonical export reached 118 rows. Results therefore apply to the frozen 103 cases, not automatically to the additional 15 rows.

Reference-label distribution:

| Label | Cases |
|---|---:|
| BUY | 4 |
| CONDITIONAL_BUY | 24 |
| DO_NOT_BUY | 51 |
| INSUFFICIENT_EVIDENCE | 24 |

The small `BUY` support makes class-specific estimates unstable.

## Compared systems

- Qwen3-4B base deployment.
- Qwen3-4B Amazon v2 fine-tuned deployment.
- GPT-4.1 mini.

All received the same messages and explicit output contract. The recorded runs produced 103/103 valid responses and 103/103 explicit `Result` fields for every system.

## Deterministic metrics

Accuracy, Macro-F1, Weighted-F1, per-class metrics, and confusion matrices are calculated from the extracted categorical decision. Token-F1 and ROUGE-L F1 compare the generated answer to the approved reference. Reference-similarity metrics measure overlap, not factual correctness.

## Blinded LLM judge

GPT-6 Sol evaluated anonymized answers as A/B/C. Alias order changes between cases. It scored grounding, evidence coverage, rule compliance, severity calibration, catalog/user separation, actionability, unsupported claims, critical errors, overall quality, and relative preference.

The judge is not ground truth. Its primary value is a consistent comparative rubric. Categorical accuracy and F1 against approved labels remain the main decision metrics.

## Reproduction

`scripts/benchmark.py` prepares cases, runs models through the Fireworks-compatible chat-completions endpoint, and recomputes reference-overlap metrics. The repository publishes the complete per-case judge results so its aggregation is auditable. Exact model deployments should be supplied through `configs/models.example.json`; never commit API keys.
