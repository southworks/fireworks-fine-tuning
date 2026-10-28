# Cost and deployment

Prices change frequently. Every price below is an **input assumption** to be refreshed from the current
Fireworks and Microsoft Foundry price lists (or a client quote) before it is shown externally. Measured
values are marked as such.

## Fixed cost: fine-tuning (measured)

| Job | Data | Duration | Fireworks estimated cost |
|---|---|---:|---:|
| Amazon v2 (`o9pa8ud2`) | 731 examples, ~1.83M tokens, 1 epoch | 9 min | USD 0.86 |
| Banking77 (`n3gtjgd9`) | 9,002 examples, ~0.70M tokens, 3 epochs | 2 h 25 m | USD 0.77 |

The training run is not the main fixed cost. Data work is: building and auditing the 1,822-example Amazon
ledger took 21 generation batches, with three model calls (request author, answer, audit) per group of four products (see
`data/dataset/manifest.json` for timings), plus human review. Budget a project around data and evaluation,
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

LoRA adapters run on **on-demand (dedicated) deployments** billed per GPU-second; serverless is not
available for fine-tuned adapters. Options:

- One GPU with the base model and several adapters loaded as add-ons (multi-LoRA). This is how the demo
  deploys: base, Amazon adapter and Banking77 adapter share one deployment.
- Scale to zero (`--min-replica-count 0`): no GPU charge while idle after one hour, at the price of a
  cold start (HTTP 503 until a replica is up).
- Fixed replicas for steady traffic and predictable latency.

## Break-even against an API model

For steady traffic, a dedicated GPU costs less than API calls once

```text
requests per hour  >  GPU price per hour ÷ API cost per request
```

provided one GPU can serve that rate (Qwen3-4B on one modern GPU should serve far more than these rates,
but our throughput under load has **not** been measured).

| GPU price assumption | vs GPT-5 (USD 0.0266/request) | vs GPT-4.1 mini (USD 0.0012/request) |
|---|---:|---:|
| USD 7 / h | ~263 requests/h | ~5,900 requests/h |
| USD 8 / h (H200, partnership deck, Oct 2026) | ~300 requests/h | ~6,700 requests/h |
| USD 13 / h (B200, partnership deck, Oct 2026) | ~489 requests/h | ~10,900 requests/h |

Reading: against a reasoning model like GPT-5, a dedicated small model pays off at modest steady volume.
Against a cheap non-reasoning model like GPT-4.1 mini, the argument is mainly quality (+10.7 points of
accuracy) and latency, not cost, unless volume is high or the GPU is shared across several adapters.

For bursty or low traffic, scale-to-zero changes the calculation to *active GPU hours* rather than 730 h
per month; e.g. 8 active hours per business day ≈ 176 h/month.

## What to measure in a client pilot

- Requests per month, peak requests per second, mean input/output tokens.
- Sustained throughput and p95 latency of the fine-tuned model on the chosen deployment shape.
- Current price quotes for the GPU and for the API model being replaced, with date.
- Cold-start time if scale-to-zero is used.
- Data preparation and review effort (usually the largest fixed cost).
