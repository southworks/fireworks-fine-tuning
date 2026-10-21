# Reproducibility

## Environment

- Python 3.11 or newer.
- Standard library only for published scripts.
- Network access for source reconstruction and model inference.

## Validate the published snapshot

```bash
python3 scripts/validate_dataset.py
python3 scripts/generate_checksums.py --check CHECKSUMS.sha256
```

## Rebuild the source subset

```bash
python3 scripts/build_source_subset.py
```

This downloads the upstream category files and writes the ignored `data/source/raw/` directory. The selection is controlled by the published product plan and seed embedded in the script.

## Generate new examples

```bash
export FOUNDRY_BASE_URL='https://YOUR_RESOURCE.openai.azure.com/openai'
export FOUNDRY_API_KEY='...'
python3 scripts/generate_sft_dataset.py --assistant-model gpt-6-sol
```

## Rerun model inference

```bash
export FIREWORKS_API_KEY='...'
python3 scripts/benchmark.py run \
  --test evaluation/qwen3-4b-amazon-v2/test_evaluation_103.jsonl \
  --output evaluation/qwen3-4b-amazon-v2/reruns/predictions.jsonl \
  --model MODEL_ID --workers 4 --max-tokens 500
```

Historical outputs are preserved rather than overwritten. API/model behavior can change, so record model IDs, date, parameters, and output hashes for every rerun.
