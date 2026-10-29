# Scope and interpretation

How to read the results, and what they do and do not claim.

## The claim

On this task, data, prompt and rubric, the fine-tuned Qwen3-4B takes better purchase decisions than
GPT-4.1 mini and GPT-5, at lower cost and latency, while GPT-5 writes better-grounded explanations.
The same pattern holds on Banking77 intent classification. Other domains, languages or prompts call for
their own evaluation, which this repository is designed to run.

## Benchmark design

- **Amazon:** 103 product-level cases from the Industrial & Scientific category, product-disjoint from
  training. Four reference cases are `BUY`, so per-class figures for `BUY` carry wide uncertainty; accuracy
  and macro-F1 over all classes are the headline metrics.
- **References** are answers written by GPT-6 Sol and approved by an independent audit against
  [DATASET_REVIEW_RULES.md](DATASET_REVIEW_RULES.md). They define the expected decision for each case.
- **Single run per configuration.** On 103 cases, differences of a few points are within normal
  run-to-run and sampling variation and should be read as ties.
- **Banking77:** the standard 3,080-case test split, with exact labels.

## The LLM judge

- GPT-6 Sol scores three anonymized, shuffled answers per case against a fixed rubric
  ([BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md)). It complements, and does not replace, the
  label-based metrics.
- The two judge sessions (vs GPT-4.1 mini and vs GPT-5) are separate runs; compare scores within a session.
  The fine-tuned model scored 61.00 and 60.40 in them, which shows the session-to-session variation.
- The rubric is strict by design: it marks critical errors in 58–82% of answers and unsupported claims in
  69–99% across all models. This is why the recommended use is a human-reviewed copilot.

## Setup details that matter

- The fine-tuned model was trained with a shorter system prompt than the one used at evaluation time
  (see [FINETUNING.md](FINETUNING.md)); every model received the same evaluation prompt.
- GPT-5 runs with its default temperature (the deployment does not accept the parameter) and up to 8,000
  output tokens including reasoning; the other models run at temperature 0 with 500 output tokens.
- Latency is end-to-end at low concurrency on two providers (Fireworks for Qwen, Microsoft Foundry for GPT).
  It describes the user-perceived response time of each setup, not peak throughput.
- Banking77 frozen results come from a local classification-head model trained for 2 epochs; the Fireworks
  path in this repository trains the same data as a generative classifier.

## Data

Catalog and review evidence reflects what sellers and customers published: it can be incomplete,
outdated or mix product variants, which is exactly what the copilot is trained to reason about. Amazon
Reviews 2023 and Banking77 terms apply to the derived data (see the README).
