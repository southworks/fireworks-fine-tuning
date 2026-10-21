# Limitations

- The benchmark has 103 cases from one Amazon category and one procurement task.
- Only four reference cases are labeled `BUY`.
- The 103-case benchmark predates the current 118-row evaluation export.
- Approved synthetic answers serve as references; they are not independent real-world procurement outcomes.
- GPT-6 Sol is an automatic judge and may have model-specific biases.
- The judge reports high residual error and unsupported-claim rates for every model.
- Catalog and review evidence can be stale, incomplete, manipulated, or variant-confounded.
- Latency, throughput, cold-start behavior, and cost per successful answer were not benchmarked here.
- The source snapshot lacks the complete Fireworks fine-tuning job configuration and billed training cost.
- Results should not be generalized to other domains, languages, prompts, or deployment settings without new evaluation.
