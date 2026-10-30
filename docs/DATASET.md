# Dataset

## Source

The experiment uses the `Industrial_and_Scientific` category of Amazon Reviews 2023. Reviews and product metadata are joined by `parent_asin`; variants belonging to one parent product are never split between training and evaluation.

## Source layer

- 5,000 products: 4,000 core products plus 1,000 deterministically ranked long-tail products.
- 533,879 original review rows.
- All matching source reviews retained, without per-product review caps or rating filters.
- Every selected product has metadata and at least three reviews.

The raw source layer is reconstructed using `data/amazon-reviews/source/product_plan.json`, `data/amazon-reviews/source/manifest.json`, and `scripts/dataset/build_source_subset.py`.

## Canonical derived dataset

| Item | Count |
|---|---:|
| Canonical examples / unique products | 1,822 |
| Approved | 1,220 |
| Approved, training split | 797 |
| Training file of the published model (`fireworks_v2_training.jsonl`) | 731 |
| Evaluation export (`evaluation.jsonl`) | 118 |
| Backup-routed | 483 |
| Held for human review (not exported) | 546 |
| Rejected | 56 |

### Training file

The model evaluated here, `ft-qwen3-4b-amazon-v2`, was trained on **`fireworks_v2_training.jsonl` (731 rows)**,
the approved training export of 2026-10-20, uploaded to Fireworks as `amazon-reviews-v2-train`. Every row maps
to an approved training example of the ledger with an identical request and answer; the system message is the
shorter version used at that time. Its label mix is 300 `DO_NOT_BUY`, 191 `INSUFFICIENT_EVIDENCE`,
157 `CONDITIONAL_BUY`, 83 `BUY`. `make upload-dataset-amazon-reviews` uploads this file.

Each model-visible export contains exactly three messages in order: `system`, `user`, and `assistant`. One example represents one `parent_asin`.

## Approved result distribution

| Result | Training | Evaluation |
|---|---:|---:|
| BUY | 102 | 19 |
| CONDITIONAL_BUY | 204 | 24 |
| DO_NOT_BUY | 300 | 51 |
| INSUFFICIENT_EVIDENCE | 191 | 24 |

The benchmark uses 103 of the 118 evaluation examples. See [BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md) for the frozen benchmark definition.

## Leakage controls

- Split assignment is performed at product level.
- A `parent_asin` appears in only one split.
- All variants and evidence for a product stay in that split.
- Model-visible identifiers are excluded unless required by the task.
- `scripts/results/validate_dataset.py` checks message structure, uniqueness, distributions, and overlap, including zero product overlap between the 731-row v2 training file and both the evaluation export and the 103 benchmark cases.

## Risks

Reviews are self-selected, may be duplicated or manipulated, and may refer to different variants. Catalog attributes are seller claims, prices may be stale, and `verified_purchase` is not proof of correctness. The derived corpus is for research and evaluation, not a substitute for procurement controls.
