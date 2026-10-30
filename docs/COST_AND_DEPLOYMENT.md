# Cost and deployment

Measured values come from the recorded jobs and benchmark runs. Prices are list-price inputs (October 2026);
replace them with current Fireworks and Microsoft Foundry prices to apply the same calculation to another workload.

## Fixed cost: fine-tuning (measured)

| Job | Data | Duration | Fireworks estimated cost |
|---|---|---:|---:|
| Amazon v2 (`o9pa8ud2`) | 731 examples, ~1.83M tokens, 1 epoch | 9 min | USD 0.86 |
| Banking77 (`n3gtjgd9`) | 9,002 examples, ~0.70M tokens, 3 epochs | 2 h 25 m | USD 0.77 |

The training run is not the main fixed cost. Data work is: building and auditing the 1,822-example Amazon
ledger took 21 generation batches, with three model calls (request author, answer, audit) per group of four products (see
`data/amazon-reviews/dataset/manifest.json` for timings), plus human review. Budget a project around data and evaluation,
not GPU time.

## Variable cost: tokens per request (measured on the 103-case benchmark)

| Model | Mean input tokens | Mean output tokens |
|---|---:|---:|
| Qwen3-4B FT v2 | 2,249 | 152 |
| GPT-4.1 mini | 2,173 | 199 |
| GPT-5 | 2,172 | 2,391 (2,139 reasoning) |

## API cost per request (assumed prices)

`cost = input_tokens × input_price + output_tokens × output_price`

| Model | Assumed price per 1M tokens (input / output) | Cost per request | Cost per 1,000 requests |
|---|---|---:|---:|
| GPT-5 | USD 1.25 / 10.00 | USD 0.0266 | USD 26.6 |
| GPT-4.1 mini | USD 0.40 / 1.60 | USD 0.0012 | USD 1.2 |

GPT-5's cost is dominated by hidden reasoning tokens; the fine-tuned model produces none.

## Serving the fine-tuned model on Fireworks

Fine-tuned LoRA models run on **on-demand (dedicated) deployments** billed per GPU-second; serverless is
not available for them. The pipeline gives each model its own deployment: the baseline and each fine-tuned
model. Options:

- Scale to zero (`--min-replica-count 0`): no GPU charge once the deployment has been idle for the
  scale-to-zero window (10 minutes in this repository's configs; Fireworks default 1 hour), at the price
  of a cold start (HTTP 503 until a replica is up).
- Fixed replicas for steady traffic and predictable latency.

## Break-even against an API model

For steady traffic, a dedicated GPU costs less than API calls once

```text
requests per hour  >  GPU price per hour ÷ API cost per request
```

provided one GPU can serve that rate; a 4B model on a single H200 or B200 is sized for well above these
volumes.

| GPU price assumption | vs GPT-5 (USD 0.0266/request) | vs GPT-4.1 mini (USD 0.0012/request) |
|---|---:|---:|
| USD 7 / h | ~263 requests/h | ~5,900 requests/h |
| USD 8 / h (H200 list price, Oct 2026) | ~300 requests/h | ~6,700 requests/h |
| USD 13 / h (B200 list price, Oct 2026) | ~489 requests/h | ~10,900 requests/h |

Reading: against a reasoning model like GPT-5, a dedicated small model pays off at modest steady volume.
Against a cheap non-reasoning model like GPT-4.1 mini, the argument is mainly quality (+10.7 points of
accuracy) and latency, not cost, unless volume is high.

For bursty or low traffic, scale-to-zero changes the calculation to *active GPU hours* rather than 730 h
per month; e.g. 8 active hours per business day ≈ 176 h/month.

## Inputs that drive the cost of a workload

- Requests per month, peak requests per second, mean input/output tokens.
- Sustained throughput and p95 latency of the fine-tuned model on the chosen deployment shape.
- Current prices of the GPU and of the API model it replaces.
- Cold-start time when scale-to-zero is used.
- Data preparation and review effort (usually the largest fixed cost).
