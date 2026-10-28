# Dataset generation

## Pipeline

```text
Amazon Reviews 2023
  -> deterministic 5,000-product selection
  -> complete reviews + product metadata
  -> diverse evidence selection
  -> procurement-request authoring
  -> single assistant response
  -> independent audit
  -> approved / pending / rejected routing
  -> product-disjoint train and evaluation exports
```

## Generation stages

1. `gpt-4.1-mini` inspects the product evidence, selects a defensible target decision, and authors the procurement request.
2. The assistant model generates one response. The first 50 canonical products used `gpt-4o`; the remaining 1,772 used `gpt-6-sol`.
3. A separate `gpt-6-sol` invocation audits grounding, attribution, decision precedence, format, and SFT readiness.
4. Only approved examples are exported to training or evaluation.

A second request is allowed only after a technical failure; responses are not regenerated using judge feedback.

## Evidence and labels

Generation follows [DATASET_REVIEW_RULES.md](DATASET_REVIEW_RULES.md). The four decisions are `BUY`, `CONDITIONAL_BUY`, `DO_NOT_BUY`, and `INSUFFICIENT_EVIDENCE`. Evidence selection favors informative and diverse reports, but frequency claims must use complete product-level statistics rather than the selected evidence subset.

## Commands

```bash
python3 scripts/build_source_subset.py

export FOUNDRY_BASE_URL='https://YOUR-RESOURCE.services.ai.azure.com/openai/v1'
export FOUNDRY_API_KEY='...'
python3 scripts/generate_sft_dataset.py \
  --data-dir data/source/raw \
  --output-dir data/generated \
  --examples 50 \
  --product-offset 0 \
  --assistant-model gpt-6-sol
```

The canonical dataset was assembled from restartable batches. Exact batch offsets, status counts, model mix, and timings are retained in `data/dataset/manifest.json`.
