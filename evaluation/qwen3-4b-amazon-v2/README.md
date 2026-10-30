# Qwen3-4B Amazon v2 benchmark (frozen)

Frozen evidence for the 103-case comparison. Read the tables in [RESULTS.md](RESULTS.md).

| File | Content |
|---|---|
| `test_evaluation_103.jsonl` (+ `.manifest.json` with SHA-256) | Prompt and approved reference answer for every case |
| `predictions_qwen3_4b_finetuned.jsonl` | Fine-tuned Qwen3-4B v2 (Fireworks) |
| `predictions_qwen3_4b_base.jsonl` | Qwen3-4B base (Fireworks) |
| `predictions_gpt41mini.jsonl` | GPT-4.1 mini (Microsoft Foundry) |
| `predictions_gpt5.jsonl` | GPT-5 (Microsoft Foundry) |
| `judgments_gpt6sol.jsonl` / `judge_summary_gpt6sol.json` | Judge session A: base vs fine-tuned vs GPT-4.1 mini |
| `judgments_gpt6sol_with_gpt5.jsonl` / `judge_summary_gpt6sol_with_gpt5.json` | Judge session B: base vs fine-tuned vs GPT-5 |
| `decision_metrics_all_models.json` | Accuracy, F1, per-class metrics, confusion matrices (`scripts/benchmark/decision_metrics.py`) |
| `metrics_all_models.json` | Token-F1, ROUGE-L and other reference-overlap metrics (`scripts/benchmark/overlap_metrics.py`) |
| `RESULTS.md` | Generated tables (`scripts/benchmark/build_report.py`) |

Every prediction row keeps the full answer, token usage, latency and error field. All four systems returned
103 valid responses; both judge sessions returned 103 valid judgments. `make results` recomputes
the derived files and fails if they differ from what is published.
