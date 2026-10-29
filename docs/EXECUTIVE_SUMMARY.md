# Executive summary

## The question

Many enterprise AI workloads are narrow and repetitive: classify a request, take a structured decision,
extract a field. Teams usually start with a large general-purpose model behind an API. We wanted to know
whether a **small open model, fine-tuned on a few hundred to a few thousand examples and served on
Fireworks AI**, can match or beat that approach on quality, latency and cost — and where it cannot.

## What we did

| | Amazon MRO purchasing copilot | Banking77 intent classification |
|---|---|---|
| Task | Read a procurement request plus catalog data and customer reviews; decide `BUY`, `CONDITIONAL_BUY`, `DO_NOT_BUY` or `INSUFFICIENT_EVIDENCE` and justify it | Map a banking customer query to 1 of 77 intents |
| Training data | 731 synthetic, audited examples built from Amazon Reviews 2023 (Industrial & Scientific) | 9,002 labelled public examples |
| Small model | Qwen3-4B + LoRA, trained on Fireworks | Qwen3.5-0.8B + LoRA (local); Qwen3-4B on Fireworks as reproducible path |
| Training effort | 1 epoch, 9 minutes, est. USD 0.86 | 2 epochs, 4.1 h on a laptop |
| Compared with | GPT-4.1 mini, GPT-5, the untuned base | GPT-5.6 Luna (direct and as an agent), GPT-4o agent |
| Test set | 103 frozen cases, products never seen in training | 3,080 standard test cases |

## What we found

**1. Fine-tuning gives the best decisions on the narrow task.**
On the purchasing task the fine-tuned 4B model picked the reference decision in 54.4% of cases, versus
43.7% for GPT-4.1 mini, 37.9% for GPT-5 and 27.2% for the same model before fine-tuning. On Banking77 a
0.8B model reached 86.9% accuracy versus 85.2% for GPT-5.6 Luna.

**2. It is much cheaper and faster to run.**
On the purchasing task GPT-5 used ~470k tokens for 103 answers (most of them hidden reasoning) against
~247k for the fine-tuned model, and answered in 25 s median versus 1.2 s. On Banking77 the frontier model
processed ~35x more tokens per request, and ~389x when wrapped in an agent.

**3. Large models still write better explanations.**
A blind GPT-6 Sol judge preferred GPT-5's rationales (score 63.7 vs 60.4) for grounding and evidence
coverage, even though GPT-5's decisions were less often correct (it recommended `BUY` 52 times when the
reference had 4). Against GPT-4.1 mini, the fine-tuned model won on both decision and explanation (61.0 vs 53.6).

**4. Nobody is ready for autonomous purchasing.**
The judge flags critical errors in 58–62% of answers for both the fine-tuned model and GPT-5. The right
target is a human-reviewed copilot.

## Recommended architecture

```text
request ──> fine-tuned small model (decision + draft)  ──> rule / confidence checks
                                                             │ pass          │ fail or high value
                                                             v               v
                                                       human approval   large-model review + human
```

The small model handles the volume cheaply and consistently; the large model is used selectively as a
reviewer or writer; a human approves anything that commits money.

## When this pattern fits

- The task is narrow and stable, with a fixed output contract (labels, schema).
- Volume is high enough that per-request cost and latency matter.
- Labelled or auditable examples can be produced (a few hundred to a few thousand).
- Quality can be measured against a reference set before go-live.

It does not fit open-ended tasks, rapidly changing policies, or cases where explanation quality matters
more than decision accuracy.

## Economics in one line

Training is a small fixed cost (under USD 1 per job here). The variable cost is either API tokens
(large model) or GPU time (dedicated Fireworks deployment). A dedicated GPU pays off once traffic exceeds
roughly *GPU price per hour ÷ API cost per request*; with the GPT-5 token mix measured here that is about
300 requests/hour for a USD 8/h GPU. Details and caveats: [COST_AND_DEPLOYMENT.md](COST_AND_DEPLOYMENT.md).

## Reading the numbers

103 test cases from one product category (4 `BUY` references), references written by a large model and
audited, an automatic judge as a second lens, and latency measured on each provider as used. See
[SCOPE.md](SCOPE.md).

## Proposed next step with a client

A 4–6 week pilot on one narrow workflow: build a reference set with the client's experts, generate and
audit training data, fine-tune and deploy on Fireworks, run the same benchmark against the model they use
today, and decide go/no-go on measured accuracy, latency and cost.
