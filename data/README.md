# Published data

`dataset/` contains the canonical derived SFT data for the Amazon case. `examples_master.jsonl` is the source of truth (every generated example with its audit metadata). `fireworks_v2_training.jsonl` is the training file of `ft-qwen3-4b-amazon-v2` (731 approved examples) and the one the pipeline uploads; `evaluation.jsonl` is the approved evaluation export the 103 benchmark cases come from (see docs/DATASET.md).

`banking77/` contains the Banking77 splits in chat format (see `banking77/README.md`).

`source/manifest.json` describes the 5,000-product source layer. `source/product_plan.json` is the deterministic core/tail selection input needed by `scripts/build_source_subset.py`. Raw Amazon reviews and metadata are intentionally not redistributed; the build script downloads them from the public upstream source into the ignored `data/source/raw/` directory.

Do not concatenate historical datasets with these exports. Preserve product-level isolation between training and evaluation.
