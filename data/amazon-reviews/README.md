# Amazon reviews data (purchasing copilot)

Derived from the `Industrial_and_Scientific` category of [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).

| Path | Content |
|---|---|
| `dataset/fireworks_v2_training.jsonl` | 731 approved training examples: the training file of `ft-qwen3-4b-amazon-v2` and the one the pipeline uploads |
| `dataset/evaluation.jsonl` | 118 approved evaluation examples; the 103 benchmark cases come from here |
| `dataset/examples_master.jsonl` | Every generated example (1,822) with its audit metadata: the source of truth |
| `dataset/manifest.json` | Counts, label distributions and generation batches |
| `source/manifest.json` | The 5,000-product source layer |
| `source/product_plan.json` | Deterministic product selection used by `scripts/dataset/build_source_subset.py` |

The raw reviews and metadata are downloaded from the upstream source by `scripts/dataset/build_source_subset.py`
into `source/raw/` (git-ignored); newly generated examples go to `generated/` (git-ignored).
Training and evaluation are product-disjoint: keep that isolation when adding data. Details: [docs/DATASET.md](../../docs/DATASET.md).
