# Limitations

## Evidence

- The Amazon benchmark has 103 cases from one category (Industrial & Scientific) and one task. Only four
  reference cases are `BUY`, so `BUY` precision/recall are unstable.
- References are synthetic answers written by GPT-6 Sol and approved by an automatic audit, not real
  procurement outcomes. A model that imitates the reference style is favoured by overlap metrics.
- The 103-case benchmark predates the current 118-row evaluation export; results apply to the frozen 103.
- One training run per configuration; no seeds or confidence intervals. A ±5-point difference on 103 cases
  is within plausible run-to-run and sampling noise.

## Judge

- GPT-6 Sol is an automatic judge with its own biases (it is also the model that wrote the references).
- The two judge sessions (vs GPT-4.1 mini, vs GPT-5) are separate runs; the fine-tuned model scored 61.00
  in one and 60.40 in the other, which shows the session-to-session variation. Do not compare across sessions.
- The judge reports high error rates for every model: critical errors in 58–82% of answers and unsupported
  claims in 69–99%. The rubric is strict by design; human calibration of the judge is still pending.

## Setup differences

- The fine-tuned model was trained with a shorter system prompt than the one used at evaluation time
  (see FINETUNING.md). All models received the same evaluation prompt.
- GPT-5 ran without `temperature` (the deployment rejects it) and with up to 8,000 output tokens, most of
  them reasoning; the other models ran at temperature 0 with 500 output tokens.
- Latency was measured end to end on two different providers and deployments, at low concurrency. It is
  not a throughput benchmark. Throughput of Qwen3-4B on one GPU under load has not been measured.

## Banking77

- The frozen results come from a local classification-head model, not from Fireworks. The Fireworks path is
  generative and has not been benchmarked yet.
- A third training epoch collapsed accuracy in the local runs; the cause was not identified.

## Data

- Catalog and review evidence can be stale, incomplete, manipulated or mix product variants.
- Amazon data terms apply; see NOTICE.md before any commercial use of the derived data.

## Generalization

Results should not be generalized to other domains, languages, prompts or deployment settings without a
new evaluation. The claim is: on this task, data, prompt and rubric, the fine-tuned model takes better
decisions at lower cost and latency.
