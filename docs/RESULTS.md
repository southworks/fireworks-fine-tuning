# Results

Detailed, generated tables (per-class F1, all judge dimensions, tokens and latency):
[evaluation/qwen3-4b-amazon-v2/RESULTS.md](../evaluation/qwen3-4b-amazon-v2/RESULTS.md).
Banking77: [evaluation/banking77/README.md](../evaluation/banking77/README.md).

## Amazon purchasing copilot (103 cases)

### Decision quality — primary metric

| Model | Accuracy | Macro-F1 | Weighted-F1 | `BUY` predicted (reference: 4) |
|---|---:|---:|---:|---:|
| **Qwen3-4B FT v2** | **54.37%** | **0.4607** | **0.5423** | 7 |
| GPT-4.1 mini | 43.69% | 0.3749 | 0.3959 | 12 |
| GPT-5 | 37.86% | 0.3330 | 0.4529 | 52 |
| Qwen3-4B base | 27.18% | 0.2333 | 0.1689 | 6 |

Fine-tuning added 27.2 points of accuracy to the base model and put it 10.7 points above GPT-4.1 mini and
16.5 points above GPT-5. GPT-5's main failure is over-recommending `BUY`; GPT-4.1 mini never answered
`INSUFFICIENT_EVIDENCE`; the base model answers `CONDITIONAL_BUY` 85 times out of 103.

### Explanation quality — blind GPT-6 Sol judge

Two separate judge sessions, three anonymized answers per case. Compare within a session only.

| Session | Model | Score /100 | Grounding | Evidence coverage | Pass rate | Critical errors | Preferred |
|---|---|---:|---:|---:|---:|---:|---:|
| A | **Qwen3-4B FT v2** | **61.00** | **75.17** | 68.91 | **34.95%** | **59.22%** | **64.08%** |
| A | GPT-4.1 mini | 53.64 | 65.32 | **73.91** | 10.68% | 69.90% | 29.13% |
| A | Qwen3-4B base | 42.75 | 62.40 | 43.35 | 1.94% | 81.55% | 6.80% |
| B | **GPT-5** | **63.70** | **84.07** | **86.04** | **35.92%** | 62.14% | **58.25%** |
| B | Qwen3-4B FT v2 | 60.40 | 74.65 | 68.44 | 33.01% | **58.25%** | 33.98% |
| B | Qwen3-4B base | 43.12 | 62.51 | 44.08 | 2.91% | 79.61% | 7.77% |

The fine-tuned model beats GPT-4.1 mini on both decisions and explanations. GPT-5 writes better-grounded
explanations than the fine-tuned model but reaches the right decision less often.

### Efficiency

| Model | Total tokens (103 requests) | Output tokens | Median latency | p95 latency |
|---|---:|---:|---:|---:|
| Qwen3-4B FT v2 | 247,281 | 15,663 | 1.16 s | 2.59 s |
| GPT-4.1 mini | 244,289 | 20,506 | 3.41 s | 5.89 s |
| GPT-5 | 469,953 | 246,273 (220,288 reasoning) | 25.00 s | 40.67 s |

Qwen models ran on Fireworks dedicated deployments and GPT models on Microsoft Foundry, so latency
compares the two serving setups as used, not the architectures.

### Operational reading

- Use the fine-tuned model to take the decision and draft the answer.
- Use rules and, selectively, a large model to review grounding on high-value or low-confidence cases.
- Keep human approval: critical-error rates near 60% rule out autonomous purchasing for every model.

## Banking77 (3,080 cases, frozen local results)

| Model | Accuracy | Macro-F1 | Tokens per request |
|---|---:|---:|---:|
| **Qwen3.5-0.8B + LoRA** | **86.88%** | **86.89%** | ~12.6 |
| Qwen3-0.6B + LoRA | 85.06% | 85.09% | ~12.6 |
| GPT-5.6 Luna (direct) | 85.19% | 84.37% | 444 |
| GPT-5.6 Luna (agent) | 84.71% | 83.99% | 4,903 |
| GPT-4o (agent) | 78.15% | 77.39% | 917 |
