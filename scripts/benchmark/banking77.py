#!/usr/bin/env python3
"""Banking77 intent classification: build the generative benchmark and score predictions.

Standard library only. Inference itself reuses the shared runners:
  Fireworks models -> scripts/fireworks/inference.py
  Foundry models   -> scripts/foundry/inference.py

Commands
  prepare-test  Convert data/banking77/test.jsonl (chat SFT rows) into benchmark rows
                {id, task, messages[system,user], reference}. With --label-list the system
                prompt also enumerates the 77 labels (the prompt used for GPT baselines).
  score         Normalize each prediction to one of the 77 labels and report accuracy,
                macro-F1 and invalid outputs for one or more prediction files.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "banking77"
GPT_SYSTEM_PROMPT = (
    "Classify the banking customer query into exactly one Banking77 intent. "
    "Reply with only the exact intent label, with no explanation. Valid labels: "
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def labels() -> list[str]:
    found = sorted({row["messages"][-1]["content"] for row in read_jsonl(DATA / "train.jsonl")})
    assert len(found) == 77, f"expected 77 labels, found {len(found)}"
    return found


def normalize(raw: str, valid: list[str]) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    value = text.strip("`\"'").splitlines()[0].strip() if text else ""
    if value in valid:
        return value
    lowered = value.lower().replace(" ", "_").replace("-", "_").rstrip(". ,;:")
    exact = {label.lower(): label for label in valid}
    if lowered in exact:
        return exact[lowered]
    padded = f" {re.sub(r'[^a-z0-9_]+', ' ', value.lower())} "
    matches = [label for label in valid if f" {label.lower()} " in padded]
    return max(matches, key=len) if matches else None


def prepare_test(source: Path, output: Path, label_list: bool, limit: int | None) -> dict:
    valid = labels()
    rows = read_jsonl(source)[: limit or None]
    out = []
    for index, row in enumerate(rows, 1):
        system, user, assistant = row["messages"]
        if label_list:
            system = {"role": "system", "content": GPT_SYSTEM_PROMPT + ", ".join(valid)}
        out.append({"id": f"banking77-{index:04d}", "task": "intent_classification",
                    "messages": [system, user], "reference": assistant["content"]})
    write_jsonl(output, out)
    return {"output": str(output), "rows": len(out), "label_list_in_prompt": label_list}


def score_file(path: Path, valid: list[str]) -> tuple[str, dict]:
    rows = read_jsonl(path)
    gold, pred = [], []
    invalid = errors = 0
    for row in rows:
        if row.get("error"):
            errors += 1
        guess = normalize(row.get("prediction", ""), valid)
        invalid += guess is None
        gold.append(row["reference"])
        pred.append(guess)
    f1s = []
    for label in valid:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    inp = sum(r.get("usage", {}).get("prompt_tokens", r.get("usage", {}).get("input_tokens", 0)) for r in rows)
    out = sum(r.get("usage", {}).get("completion_tokens", r.get("usage", {}).get("output_tokens", 0)) for r in rows)
    latency = sorted(r.get("latency_seconds", 0) for r in rows)
    return rows[0].get("model", path.stem), {
        "file": path.name,
        "n": len(rows),
        "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(rows), 4) if rows else None,
        "macro_f1": round(statistics.mean(f1s), 4),
        "invalid_outputs": invalid,
        "request_errors": errors,
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": inp + out,
        "latency_mean": round(statistics.mean(latency), 3) if latency else None,
        "latency_p50": round(statistics.median(latency), 3) if latency else None,
        "top_confusions": Counter(f"{g} -> {p}" for g, p in zip(gold, pred) if g != p).most_common(10),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-test")
    prep.add_argument("--source", type=Path, default=DATA / "test.jsonl")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--label-list", action="store_true")
    prep.add_argument("--limit", type=int)
    score = sub.add_parser("score")
    score.add_argument("--predictions", type=Path, nargs="+", required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-test":
        print(json.dumps(prepare_test(args.source, args.output, args.label_list, args.limit), indent=2))
    else:
        valid = labels()
        report = {"labels": len(valid), "models": dict(score_file(p, valid) for p in args.predictions)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({m: {k: v[k] for k in ("n", "accuracy", "macro_f1", "invalid_outputs")} for m, v in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
