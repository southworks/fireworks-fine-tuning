#!/usr/bin/env python3
"""Reference-overlap metrics for benchmark answers: token F1, ROUGE-L and simple language signals.

  python3 scripts/benchmark/overlap_metrics.py --predictions <predictions_*.jsonl> --output <metrics.json>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from typing import Any

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make scripts/ importable

from common.jsonl import read_jsonl

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*", re.I)
RECOMMEND_RE = re.compile(r"\b(?:do not buy|don't buy|not recommended|recommend(?:ed|ation)?|buy|purchase|avoid|suitable|unsuitable)\b", re.I)
UNCERTAINTY_RE = re.compile(r"\b(?:may|might|could|reported|according to|evidence|uncertain|limited|one review|some users?)\b", re.I)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_f1(reference: str, prediction: str) -> float:
    ref, pred = Counter(tokens(reference)), Counter(tokens(prediction))
    overlap = sum((ref & pred).values())
    if not ref or not pred or not overlap:
        return 0.0
    precision, recall = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for index, right in enumerate(b, 1):
            current.append(previous[index - 1] + 1 if left == right else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref, pred = tokens(reference), tokens(prediction)
    if not ref or not pred:
        return 0.0
    lcs = lcs_length(ref, pred)
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def extract_user_evidence(messages: list[dict[str, str]]) -> str:
    return " ".join(m.get("content", "") for m in messages if m.get("role") in {"system", "user"})


def unsupported_token_rate(evidence: str, prediction: str) -> float:
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "for", "is", "are", "be", "with", "that", "this", "it", "as", "on", "by", "from", "should", "product", "users", "reviews"}
    source = set(tokens(evidence))
    meaningful = [t for t in tokens(prediction) if len(t) > 3 and t not in stop]
    return sum(t not in source for t in meaningful) / len(meaningful) if meaningful else 1.0


def row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    prediction, reference = row.get("prediction", ""), row["reference"]
    evidence = extract_user_evidence(row["messages"])
    pred_tokens = tokens(prediction)
    return {
        "id": row["id"], "task": row["task"], "error": row.get("error"),
        "token_f1": token_f1(reference, prediction),
        "rouge_l_f1": rouge_l_f1(reference, prediction),
        "unsupported_token_rate": unsupported_token_rate(evidence, prediction),
        "length_words": len(pred_tokens),
        "empty": not bool(pred_tokens),
        "has_recommendation_language": bool(RECOMMEND_RE.search(prediction)),
        "has_uncertainty_language": bool(UNCERTAINTY_RE.search(prediction)),
    }


def summarize(predictions: list[Path], output: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"models": {}}
    for path in predictions:
        rows = list(read_jsonl(path))
        metrics = [row_metrics(row) for row in rows]
        model = rows[0].get("model", path.stem) if rows else path.stem
        successful = [m for m in metrics if not m["error"]]
        def aggregate(items):
            return {
                "n": len(items),
                "token_f1": round(statistics.mean(m["token_f1"] for m in items), 4) if items else None,
                "rouge_l_f1": round(statistics.mean(m["rouge_l_f1"] for m in items), 4) if items else None,
                "unsupported_token_rate": round(statistics.mean(m["unsupported_token_rate"] for m in items), 4) if items else None,
                "mean_length_words": round(statistics.mean(m["length_words"] for m in items), 1) if items else None,
                "recommendation_language_rate": round(statistics.mean(m["has_recommendation_language"] for m in items), 4) if items else None,
                "uncertainty_language_rate": round(statistics.mean(m["has_uncertainty_language"] for m in items), 4) if items else None,
            }
        report["models"][model] = {
            "overall": aggregate(successful),
            "errors": [m for m in metrics if m["error"]],
        }
    models = list(report["models"])
    if len(models) == 2:
        first, second = models
        report["comparison"] = {
            "first": first, "second": second,
            "delta_second_minus_first": {
                key: round(report["models"][second]["overall"][key] - report["models"][first]["overall"][key], 4)
                for key in ("token_f1", "rouge_l_f1", "unsupported_token_rate")
                if report["models"][first]["overall"][key] is not None and report["models"][second]["overall"][key] is not None
            },
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.predictions, args.output), indent=2))


if __name__ == "__main__":
    main()
