#!/usr/bin/env python3
"""Generate answers with a Fireworks model for every row of a benchmark file.

  python3 scripts/fireworks/inference.py --test <test.jsonl> --output <predictions.jsonl> \
      --model accounts/<account>/models/<model>#accounts/<account>/deployments/<deployment>

Uses the OpenAI-compatible chat completions endpoint (FIREWORKS_API_KEY). Each output row keeps the
answer, token usage, latency and error; --resume keeps finished rows and retries failed ones.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make scripts/ importable

from common.jsonl import read_jsonl, write_jsonl


def call_model(api_key: str, model: str, messages: list[dict[str, str]], max_tokens: int, timeout: int, retries: int, reasoning_effort: str | None) -> tuple[str, dict]:
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    headers = {"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"], body.get("usage", {})
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"Fireworks HTTP {exc.code}: {body[:500]}") from exc
            if attempt == retries:
                raise RuntimeError(f"Fireworks HTTP {exc.code}: {body[:500]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Fireworks request failed: {exc}") from exc
        time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Fireworks request failed after {retries + 1} attempts")


def run_model(test_path: Path, output: Path, model: str, workers: int, max_tokens: int, limit: int | None, reasoning_effort: str | None, resume: bool) -> None:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("Set FIREWORKS_API_KEY in the environment; it is never written to disk.")
    rows = list(read_jsonl(test_path))
    if limit:
        rows = rows[:limit]
    retained: dict[str, dict[str, Any]] = {}
    if resume and output.exists():
        retained = {row["id"]: row for row in read_jsonl(output) if not row.get("error")}
        rows = [row for row in rows if row["id"] not in retained]
        print(f"Resuming: retained {len(retained)} successful rows; retrying {len(rows)} rows", flush=True)

    def execute(row):
        started = time.monotonic()
        try:
            prediction, usage = call_model(api_key, model, row["messages"], max_tokens, 120, 4, reasoning_effort)
            return {**row, "model": model, "prediction": prediction, "usage": usage, "latency_seconds": round(time.monotonic() - started, 3), "error": None}
        except Exception as exc:  # keep partial runs inspectable
            return {**row, "model": model, "prediction": "", "usage": {}, "latency_seconds": round(time.monotonic() - started, 3), "error": str(exc)}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, row): row["id"] for row in rows}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            print(f"[{index}/{len(rows)}] {futures[future]}", flush=True)
    results.extend(retained.values())
    results.sort(key=lambda r: (r["task"], r["id"]))
    write_jsonl(output, results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"))
    parser.add_argument("--resume", action="store_true", help="keep successful rows of an existing output and retry the rest")
    args = parser.parse_args()
    run_model(args.test, args.output, args.model, args.workers, args.max_tokens, args.limit, args.reasoning_effort, args.resume)


if __name__ == "__main__":
    main()
