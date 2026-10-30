#!/usr/bin/env python3
"""Generate answers with a Microsoft Foundry deployment for every row of a benchmark file.

  python3 scripts/foundry/inference.py --test <test.jsonl> --output <predictions.jsonl> --model <deployment>

--resume keeps finished rows and retries failed ones. See foundry/client.py for the environment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import time

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make scripts/ importable

from common.jsonl import read_jsonl, write_jsonl
from foundry.client import foundry_endpoint, foundry_key, post_responses


def run_generation(test: Path, output: Path, endpoint: str, model: str, workers: int, resume: bool, limit: int | None, max_output_tokens: int, temperature: float | None) -> None:
    key = foundry_key()
    rows = list(read_jsonl(test))
    if limit:
        rows = rows[:limit]
    retained = {}
    if resume and output.exists():
        retained = {row["id"]: row for row in read_jsonl(output) if not row.get("error")}
        rows = [row for row in rows if row["id"] not in retained]
        print(f"Resuming: retained {len(retained)}; retrying {len(rows)}", flush=True)

    def execute(row):
        started = time.monotonic()
        payload = {
            "model": model,
            "input": row["messages"],
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        try:
            prediction, usage, _ = post_responses(endpoint, key, payload)
            return {**row, "model": model, "prediction": prediction, "usage": usage, "latency_seconds": round(time.monotonic() - started, 3), "error": None}
        except Exception as exc:
            return {**row, "model": model, "prediction": "", "usage": {}, "latency_seconds": round(time.monotonic() - started, 3), "error": str(exc)}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, row): row["id"] for row in rows}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            print(f"[{index}/{len(rows)}] {futures[future]}", flush=True)
    results.extend(retained.values())
    results.sort(key=lambda row: (row["task"], row["id"]))
    write_jsonl(output, results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Foundry deployment name")
    parser.add_argument("--endpoint", help="override $FOUNDRY_BASE_URL/responses")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float)
    args = parser.parse_args()
    run_generation(args.test, args.output, foundry_endpoint(args.endpoint), args.model, args.workers, args.resume,
                   args.limit, args.max_output_tokens, args.temperature)


if __name__ == "__main__":
    main()
