#!/usr/bin/env python3
"""Validate the published datasets, Fireworks job records, and frozen benchmark evidence (offline)."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "amazon-reviews" / "dataset"
EVAL = ROOT / "evaluation" / "qwen3-4b-amazon-v2"
BANKING = ROOT / "data" / "banking77"
FIREWORKS = ROOT / "records" / "fireworks"
V2_TRAINING_SHA256 = "0640754bf0e6630683799d2f32035086c3ab1b9a5b9fba50dd08aa5e6c8b1e11"
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
    evaluation = list(rows(DATA / "evaluation.jsonl"))

    assert len(master) == manifest["total_examples"] == 1822
    assert len(evaluation) == manifest["evaluation_examples"] == 118

    ids = [row["metadata"]["example_id"] for row in master]
    products = [row["metadata"]["parent_asin"] for row in master]
    assert len(ids) == len(set(ids)), "duplicate example IDs"
    assert len(products) == len(set(products)) == manifest["unique_products"], "duplicate products"

    statuses = Counter(row["metadata"]["review_status"] for row in master)
    assert dict(statuses) == manifest["statuses"]
    master_by_messages = {json.dumps(row["messages"], sort_keys=True): row for row in master}
    split_products: dict[str, set[str]] = {
        split: {row["metadata"]["parent_asin"] for row in master
                if row["metadata"]["review_status"] == "approved" and row["metadata"]["dataset_split"] == split}
        for split in ("train", "evaluation")
    }

    labels = Counter()
    for number, row in enumerate(evaluation, 1):
        labels[validate_messages(row, f"evaluation:{number}")] += 1
        key = json.dumps(row["messages"], sort_keys=True)
        assert key in master_by_messages, f"evaluation:{number}: not found in master"
        source = master_by_messages[key]
        assert source["metadata"]["review_status"] == "approved"
        assert source["metadata"]["dataset_split"] == "evaluation"
    assert dict(labels) == manifest["split_results"]["evaluation"]
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
        "predictions_gpt5.jsonl",
    )
    for name in prediction_files:
        predictions = list(rows(EVAL / name))
        assert len(predictions) == 103, f"{name}: expected 103 rows"
        assert {row["id"] for row in predictions} == test_ids, f"{name}: case IDs differ"
        assert all(not row.get("error") and row.get("prediction") for row in predictions), f"{name}: failed rows"

    for name in ("judgments_gpt6sol.jsonl", "judgments_gpt6sol_with_gpt5.jsonl"):
        judgments = list(rows(EVAL / name))
        assert len(judgments) == 103, name
        assert {row["id"] for row in judgments} == test_ids, name
        assert all(not row.get("error") and row.get("judgment") for row in judgments), name

    # Exact training file uploaded for Fireworks job o9pa8ud2 (ft-qwen3-4b-amazon-v2).
    v2_path = DATA / "fireworks_v2_training.jsonl"
    assert hashlib.sha256(v2_path.read_bytes()).hexdigest() == V2_TRAINING_SHA256, "v2 training file changed"
    v2 = list(rows(v2_path))
    master_by_user = {row["messages"][1]["content"]: row for row in master}
    v2_products = set()
    for number, row in enumerate(v2, 1):
        validate_messages(row, f"v2:{number}")
        source = master_by_user.get(row["messages"][1]["content"])
        assert source, f"v2:{number}: request not found in master"
        assert source["messages"][2]["content"] == row["messages"][2]["content"], f"v2:{number}: answer differs from master"
        assert source["metadata"]["review_status"] == "approved" and source["metadata"]["dataset_split"] == "train"
        v2_products.add(source["metadata"]["parent_asin"])
    test_products = {master_by_user[row["messages"][1]["content"]]["metadata"]["parent_asin"] for row in test if row["messages"][1]["content"] in master_by_user}
    assert len(test_products) == 103, "benchmark cases must trace back to the canonical ledger"
    assert v2_products.isdisjoint(split_products["evaluation"]), "v2 training/evaluation product leakage"
    assert v2_products.isdisjoint(test_products), "v2 training/benchmark product leakage"

    job = json.loads((FIREWORKS / "sftj_amazon_v2_o9pa8ud2.json").read_text(encoding="utf-8"))
    model = json.loads((FIREWORKS / "model_ft-qwen3-4b-amazon-v2.json").read_text(encoding="utf-8"))
    dataset = json.loads((FIREWORKS / "dataset_amazon-reviews-v2-train.json").read_text(encoding="utf-8"))
    assert model["fine_tuning_job"].endswith("/o9pa8ud2") and job["output_model"] == model["name"]
    assert int(dataset["example_count"]) == len(v2) == 731
    for path in FIREWORKS.glob("*.json"):
        assert "Signature=" not in path.read_text(encoding="utf-8"), f"{path.name}: unredacted signed URL"

    # Banking77 chat-format splits used for Fireworks SFT and the generative benchmark.
    banking = {split: list(rows(BANKING / f"{split}.jsonl")) for split in ("train", "validation", "test")}
    assert {k: len(v) for k, v in banking.items()} == {"train": 9002, "validation": 1001, "test": 3080}
    label_sets = {}
    for split, items in banking.items():
        for number, row in enumerate(items, 1):
            assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"], f"banking77 {split}:{number}"
        label_sets[split] = {row["messages"][2]["content"] for row in items}
    assert len(label_sets["train"]) == 77 and label_sets["train"] == label_sets["test"] == label_sets["validation"]

    print(json.dumps({
        "status": "ok",
        "canonical_examples": len(master),
        "training_examples": len(v2),
        "evaluation_examples": len(evaluation),
        "benchmark_cases": len(test),
        "prediction_rows": len(prediction_files) * len(test),
        "judge_sessions": 2,
        "train_evaluation_product_overlap": 0,
        "v2_training_benchmark_product_overlap": 0,
        "fireworks_job_records": "consistent, redacted",
        "banking77": {k: len(v) for k, v in banking.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
