# Results

## Categorical and reference metrics

| Model | Accuracy | Macro-F1 | Weighted-F1 | Token-F1 | ROUGE-L F1 |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B FT v2 | **54.37%** | **0.4607** | **0.5423** | **0.5200** | **0.3551** |
| GPT-4.1 mini | 43.69% | 0.3749 | 0.3959 | 0.3767 | 0.2367 |
| Qwen3-4B base | 27.18% | 0.2333 | 0.1689 | 0.3279 | 0.2221 |

Fine-tuning improved accuracy over base Qwen3-4B by 27.19 percentage points and exceeded GPT-4.1 mini by 10.68 points on this benchmark.

## Blinded GPT-6 Sol evaluation

| Model | Overall | Grounding | Coverage | Rule compliance | Actionability | Preferred |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B FT v2 | **61.00** | **75.17** | 68.91 | **62.72** | **62.92** | **64.08%** |
| GPT-4.1 mini | 53.64 | 65.32 | **73.91** | 50.92 | 58.85 | 29.13% |
| Qwen3-4B base | 42.75 | 62.40 | 43.35 | 42.22 | 39.79 | 6.80% |

## Operational interpretation

The fine-tuned 4B model is the strongest of the three for this narrow task. It is also not ready for autonomous purchasing: its judge pass rate is 34.95%, critical-error rate 59.22%, and unsupported-claim rate 75.73%. The appropriate target is a structured, human-reviewed copilot with validation and escalation controls.

Machine-readable metrics and all 103 outputs are under `evaluation/qwen3-4b-amazon-v2/`.
