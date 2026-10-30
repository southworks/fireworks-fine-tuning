#!/usr/bin/env python3
"""Mode 1: show the results from the published data and metrics (offline, no credentials).

  python3 scripts/results/show_results.py [--quick]

Steps: validate the data and Fireworks records, recompute every metric from the published
answers and compare byte-for-byte, check RESULTS.md is current, print the Amazon results,
three cases side by side, and the frozen Banking77 results.
"""

from __future__ import annotations

import argparse
import filecmp
import re
import tempfile

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make scripts/ importable

from common.cli import py
from common.paths import ROOT

EVAL = ROOT / "evaluation" / "qwen3-4b-amazon-v2"


def title(text: str) -> None:
    print(f"\n\033[1m== {text}\033[0m")


def section(path: Path, start: str, stop: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(start)}.*?(?=^{re.escape(stop)})", text, re.S | re.M)
    return match.group(0).rstrip() if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the side-by-side cases")
    args = parser.parse_args()

    title("1. Data: validated, leakage-free and traceable to the Fireworks job")
    py("results/validate_dataset.py")
    py("results/generate_checksums.py", "--check", str(ROOT / "CHECKSUMS.sha256"))

    title("2. Metrics recomputed from the raw answers")
    predictions = [str(p) for p in sorted(EVAL.glob("predictions_*.jsonl"))]
    with tempfile.TemporaryDirectory() as tmp:
        decisions, overlap = Path(tmp) / "decision.json", Path(tmp) / "overlap.json"
        py("benchmark/decision_metrics.py", "--test", str(EVAL / "test_evaluation_103.jsonl"), "--predictions", *predictions, "--output", str(decisions), "--table")
        py("benchmark/overlap_metrics.py", "--predictions", *predictions, "--output", str(overlap), quiet=True)
        same = filecmp.cmp(decisions, EVAL / "decision_metrics_all_models.json", shallow=False) and \
            filecmp.cmp(overlap, EVAL / "metrics_all_models.json", shallow=False)
    if not same:
        raise SystemExit("MISMATCH between recomputed and published metrics")
    print("Recomputed metrics match the published files.")
    py("benchmark/build_report.py", "--check")

    title("3. Amazon purchasing copilot: fine-tuned Qwen3-4B vs GPT-4.1 mini vs GPT-5 (103 cases)")
    print(section(EVAL / "RESULTS.md", "## 1.", "## 2."))
    print("\n" + section(EVAL / "RESULTS.md", "## 4.", "Qwen models ran"))

    if not args.quick:
        title("4. What the answers look like")
        py("results/show_cases.py")

    title("5. Banking77 intent classification (frozen results)")
    print(section(ROOT / "evaluation" / "banking77" / "README.md", "## Frozen results", "Training cost"))
    print("\nFull tables: evaluation/qwen3-4b-amazon-v2/RESULTS.md · Business reading: docs/EXECUTIVE_SUMMARY.md")


if __name__ == "__main__":
    main()
