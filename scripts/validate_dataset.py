#!/usr/bin/env python3
"""Validate canonical dataset structure, counts, routing, and benchmark integrity."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset"
EVAL = ROOT / "evaluation" / "qwen3-4b-amazon-v2"
LABEL_RE = re.compile(r"^- Result: (BUY|CONDITIONAL_BUY|DO_NOT_BUY|INSUFFICIENT_EVIDENCE)$", re.M)


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise AssertionError(f"invalid JSON at {path}:{number}: {error}") from error


def validate_messages(row: dict, where: str) -> str:
    messages = row.get("messages")
    assert isinstance(messages, list) and len(messages) == 3, f"{where}: expected three messages"
    assert [item.get("role") for item in messages] == ["system", "user", "assistant"], f"{where}: invalid roles"
    match = LABEL_RE.search(messages[-1].get("content", ""))
    assert match, f"{where}: missing canonical Result"
    return match.group(1)


def main() -> int:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    master = list(rows(DATA / "examples_master.jsonl"))
    train = list(rows(DATA / "training.jsonl"))
    evaluation = list(rows(DATA / "evaluation.jsonl"))

    assert len(master) == manifest["total_examples"] == 1822
    assert len(train) == manifest["training_examples"] == 797
    assert len(evaluation) == manifest["evaluation_examples"] == 118

    ids = [row["metadata"]["example_id"] for row in master]
    products = [row["metadata"]["parent_asin"] for row in master]
    assert len(ids) == len(set(ids)), "duplicate example IDs"
    assert len(products) == len(set(products)) == manifest["unique_products"], "duplicate products"

    statuses = Counter(row["metadata"]["review_status"] for row in master)
    assert dict(statuses) == manifest["statuses"]
    master_by_messages = {json.dumps(row["messages"], sort_keys=True): row for row in master}
    split_products: dict[str, set[str]] = {"train": set(), "evaluation": set()}

    for split, exported in (("train", train), ("evaluation", evaluation)):
        labels = Counter()
        for number, row in enumerate(exported, 1):
            labels[validate_messages(row, f"{split}:{number}")] += 1
            key = json.dumps(row["messages"], sort_keys=True)
            assert key in master_by_messages, f"{split}:{number}: not found in master"
            source = master_by_messages[key]
            assert source["metadata"]["review_status"] == "approved"
            assert source["metadata"]["dataset_split"] == split
            split_products[split].add(source["metadata"]["parent_asin"])
        assert dict(labels) == manifest["split_results"][split]

    assert split_products["train"].isdisjoint(split_products["evaluation"]), "product leakage"

    test_path = EVAL / "test_evaluation_103.jsonl"
    test = list(rows(test_path))
    test_manifest = json.loads((EVAL / "test_evaluation_103.manifest.json").read_text(encoding="utf-8"))
    assert len(test) == test_manifest["total"] == 103
    assert hashlib.sha256(test_path.read_bytes()).hexdigest() == test_manifest["sha256"]
    test_ids = {row["id"] for row in test}
    assert len(test_ids) == 103

    prediction_files = (
        "predictions_qwen3_4b_base.jsonl",
        "predictions_qwen3_4b_finetuned.jsonl",
        "predictions_gpt41mini.jsonl",
    )
    for name in prediction_files:
        predictions = list(rows(EVAL / name))
        assert len(predictions) == 103, f"{name}: expected 103 rows"
        assert {row["id"] for row in predictions} == test_ids, f"{name}: case IDs differ"
        assert all(not row.get("error") and row.get("prediction") for row in predictions), f"{name}: failed rows"

    judgments = list(rows(EVAL / "judgments_gpt6sol.jsonl"))
    assert len(judgments) == 103
    assert {row["id"] for row in judgments} == test_ids
    assert all(not row.get("error") and row.get("judgment") for row in judgments)

    print(json.dumps({
        "status": "ok",
        "canonical_examples": len(master),
        "training_examples": len(train),
        "evaluation_examples": len(evaluation),
        "benchmark_cases": len(test),
        "prediction_rows": 3 * len(test),
        "judgments": len(judgments),
        "train_evaluation_product_overlap": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
