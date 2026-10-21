# Qwen3-4B Amazon v2 benchmark

This directory is the frozen evidence package for the 103-case comparison.

- `test_evaluation_103.jsonl`: prompt and approved reference for every case.
- `predictions_*.jsonl`: complete responses, usage, latency, and errors for each model.
- `metrics_three_models.json`: reference-overlap metrics.
- `decision_metrics.json`: label metrics and confusion matrices.
- `judgments_gpt6sol.jsonl`: blinded per-case scoring and ranking.
- `judge_summary_gpt6sol.json`: aggregate judge metrics.

All three systems returned 103 valid responses, and the judge returned 103 valid judgments. See `docs/BENCHMARK_METHODOLOGY.md` and `docs/RESULTS.md`.
