#!/usr/bin/env python3
"""Print a few benchmark cases side by side (offline, from frozen predictions).

Picks illustrative cases: one where every model agrees with the reference, one where
the fine-tuned model is right and GPT-5 chose BUY, and one where the fine-tuned model
is wrong. Use --ids to choose cases explicitly.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation" / "qwen3-4b-amazon-v2"
FILES = {
    "Qwen3-4B FT v2": "predictions_qwen3_4b_finetuned.jsonl",
    "GPT-4.1 mini": "predictions_gpt41mini.jsonl",
    "GPT-5": "predictions_gpt5.jsonl",
    "Qwen3-4B base": "predictions_qwen3_4b_base.jsonl",
}
LABEL_RE = re.compile(r"(?<![A-Z_])(INSUFFICIENT_EVIDENCE|CONDITIONAL_BUY|DO_NOT_BUY|BUY)(?![A-Z_])")
RATIONALE_RE = re.compile(r"Rationale:\s*(.+)")


def read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return {row["id"]: row for row in map(json.loads, filter(str.strip, handle))}


def decision(text: str) -> str:
    match = LABEL_RE.search((text or "").upper())
    return match.group(1) if match else "MISSING"


def rationale(text: str) -> str:
    match = RATIONALE_RE.search(text or "")
    return (match.group(1).strip() if match else (text or "").strip().splitlines()[0] if text else "")[:220]


def pick(test: dict, preds: dict) -> list[str]:
    ids = list(test)
    ref = {i: decision(test[i]["reference"]) for i in ids}
    got = {name: {i: decision(rows[i]["prediction"]) for i in ids} for name, rows in preds.items()}
    ft, g5 = got["Qwen3-4B FT v2"], got["GPT-5"]
    chosen = []
    for rule in (
        lambda i: all(got[n][i] == ref[i] for n in got if n != "Qwen3-4B base"),
        lambda i: ft[i] == ref[i] and g5[i] == "BUY" and ref[i] != "BUY",
        lambda i: ft[i] != ref[i],
    ):
        chosen.append(next(i for i in ids if rule(i) and i not in chosen))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args()
    test = read(EVAL / "test_evaluation_103.jsonl")
    preds = {name: read(EVAL / file) for name, file in FILES.items()}
    for case_id in args.ids or pick(test, preds):
        row = test[case_id]
        user = row["messages"][-1]["content"]
        item = next((l.split(":", 1)[1].strip() for l in user.splitlines() if l.startswith("Item needed:")), "")
        found = re.search(r'"title": "([^"]+)"', user)
        title = f"Request: {item}\nProduct: {(found.group(1) if found else '')[:140]}"
        print("=" * 100)
        print(f"{case_id}  |  reference: {decision(row['reference'])}")
        print(title)
        print("-" * 100)
        for name, rows in preds.items():
            pred = rows[case_id]["prediction"]
            mark = "OK " if decision(pred) == decision(row["reference"]) else "-- "
            print(f"{mark}{name:<16} {decision(pred):<22} {rows[case_id]['latency_seconds']:>6.2f}s")
            print(textwrap.indent(textwrap.fill(rationale(pred), 94), "      "))
    print("=" * 100)
    print("OK = same decision as the reference label. Full answers: evaluation/qwen3-4b-amazon-v2/predictions_*.jsonl")


if __name__ == "__main__":
    main()
