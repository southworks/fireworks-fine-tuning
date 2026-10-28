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

| System | Served by | Generation settings |
|---|---|---|
| Qwen3-4B base | Fireworks dedicated deployment | temperature 0, top_p 1, max 500 tokens, `reasoning_effort=none` |
| Qwen3-4B Amazon v2 fine-tuned (LoRA) | Fireworks, LoRA add-on on a dedicated deployment | same as base |
| GPT-4.1 mini | Microsoft Foundry, Responses API | max 500 output tokens |
| GPT-5 | Microsoft Foundry, Responses API | no temperature (rejected by the deployment), max 8,000 output tokens (includes reasoning) |

All received the same messages and explicit output contract. The recorded runs produced 103/103 valid responses and 103/103 explicit `Result` fields for every system.

## Deterministic metrics

Accuracy, Macro-F1, Weighted-F1, per-class metrics, and confusion matrices are calculated from the extracted categorical decision. Token-F1 and ROUGE-L F1 compare the generated answer to the approved reference. Reference-similarity metrics measure overlap, not factual correctness.

## Blinded LLM judge

GPT-6 Sol evaluated anonymized answers as A/B/C. Alias order is shuffled per case with a deterministic seed. The judge prompt and JSON schema are in `scripts/foundry_models.py` (`JUDGE_INSTRUCTIONS`, `score_schema`).

Two sessions were run, each with three candidates: **A** = base, fine-tuned, GPT-4.1 mini (`judgments_gpt6sol.jsonl`); **B** = base, fine-tuned, GPT-5 (`judgments_gpt6sol_with_gpt5.jsonl`). Scores are relative to the other candidates in the same session and must not be compared across sessions. It scored grounding, evidence coverage, rule compliance, severity calibration, catalog/user separation, actionability, unsupported claims, critical errors, overall quality, and relative preference.

The judge is not ground truth. Its primary value is a consistent comparative rubric. Categorical accuracy and F1 against approved labels remain the main decision metrics.

## Scripts

| Step | Script |
|---|---|
| Fireworks inference, reference-overlap metrics | `scripts/benchmark.py run` / `summarize` |
| Foundry inference (GPT-4.1 mini, GPT-5), judge, judge aggregation | `scripts/foundry_models.py generate` / `judge` / `summarize` |
| Label extraction, accuracy, F1, confusion matrices | `scripts/decision_metrics.py` |
| Result tables | `scripts/build_report.py` |

The repository publishes every per-case answer and judgment so all aggregates are auditable. Models and keys are supplied through `.env` (see `.env.example`); never commit API keys.
