# Dataset

## Source

The experiment uses the `Industrial_and_Scientific` category of Amazon Reviews 2023. Reviews and product metadata are joined by `parent_asin`; variants belonging to one parent product are never split between training and evaluation.

## Source layer

- 5,000 products: 4,000 core products plus 1,000 deterministically ranked long-tail products.
- 533,879 original review rows.
- All matching source reviews retained, without per-product review caps or rating filters.
- Every selected product has metadata and at least three reviews.

The raw source layer is reconstructed using `data/source/product_plan.json`, `data/source/manifest.json`, and `scripts/build_source_subset.py`.

## Canonical derived dataset

| Item | Count |
|---|---:|
| Canonical examples / unique products | 1,822 |
| Approved | 1,220 |
| Training export | 797 |
| Evaluation export | 118 |
| Backup-routed | 483 |
| Pending review | 546 |
| Rejected | 56 |

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
- `scripts/validate_dataset.py` checks message structure, uniqueness, distributions, and overlap.

## Risks

Reviews are self-selected, may be duplicated or manipulated, and may refer to different variants. Catalog attributes are seller claims, prices may be stale, and `verified_purchase` is not proof of correctness. The derived corpus is for research and evaluation, not a substitute for procurement controls.
