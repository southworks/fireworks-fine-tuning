#!/usr/bin/env python3
"""Run a benchmark defined in a TOML config: generate answers for every model, judge, score, report.

  python3 scripts/benchmark/run_benchmark.py --config configs/benchmark/amazon-reviews.toml [--limit 10]

Fireworks models are addressed by deployment name (plus the model name for fine-tuned models);
Foundry models by deployment name. Each model step resumes completed rows and retries failed ones
(3 passes); the run stops if any case is still missing, so reports never use partial answers.
Results go to the config `output_dir` (git-ignored); published evidence is never touched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make scripts/ importable

from common.cli import fail, py
from common.config import env, load_config, load_env, require
from common.paths import ROOT
from fireworks.cli import deployment_path, model_path


def read_ids(path: Path, ok_only: bool = False) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {r["id"] for r in rows if not ok_only or (not r.get("error") and r.get("prediction"))}


def prepare(config: dict, out: Path, limit: int) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    tests = {"default": out / "test.jsonl"}
    if config["task"] == "banking77":
        tests["label_list"] = out / "test_label_list.jsonl"
        py("benchmark/banking77.py", "prepare-test", "--source", config["test"], "--output", str(tests["default"]), "--limit", str(limit), quiet=True)
        py("benchmark/banking77.py", "prepare-test", "--source", config["test"], "--output", str(tests["label_list"]), "--label-list", "--limit", str(limit), quiet=True)
    else:
        with (ROOT / config["test"]).open(encoding="utf-8") as source:
            rows = [line for line in source if line.strip()][:limit]
        tests["default"].write_text("".join(rows), encoding="utf-8")
    print(f"test set: {len(read_ids(tests['default']))} cases -> {tests['default'].relative_to(ROOT)}")
    return tests


def generate(model: dict, test: Path, output: Path, workers: int, dry_run: bool) -> None:
    expected = read_ids(test)
    if read_ids(output) - expected:
        fail(f"{output.name} has cases outside this test set (different limit?). Delete it or use the same limit.")
    if model["provider"] == "fireworks":
        target = deployment_path(model["deployment"])
        if model.get("model"):
            target = f"{model_path(model['model'])}#{target}"
        command = ("fireworks/inference.py", "--test", str(test), "--output", str(output), "--model", target,
                   "--reasoning-effort", "none", "--max-tokens", str(model.get("max_tokens", 500)),
                   "--workers", str(workers), "--resume")
    elif model["provider"] == "foundry":
        command = ("foundry/inference.py", "--test", str(test), "--output", str(output),
                   "--model", model["deployment"], "--max-output-tokens", str(model.get("max_tokens", 500)),
                   "--workers", str(workers), "--resume")
        if model.get("temperature") is not None:
            command += ("--temperature", str(model["temperature"]))
    else:
        fail(f"model {model['label']}: provider must be fireworks or foundry")
    for attempt in range(1, 4):
        py(*command, dry_run=dry_run, check=False)
        missing = expected - read_ids(output, ok_only=True)
        if dry_run:
            return
        if not missing:
            print(f"{model['label']}: {len(expected)}/{len(expected)} answers")
            return
        print(f"{model['label']}: {len(missing)} cases failed, retrying ({attempt}/3)")
        time.sleep(10)
    fail(f"{model['label']}: {len(missing)} cases still failing; re-run to resume")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, help="override the number of test cases")
    parser.add_argument("--only", nargs="*", help="run only these model labels (plus scoring)")
    parser.add_argument("--fresh", action="store_true", help="delete previous results in output_dir first")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_env()
    config = load_config(args.config)
    require(config, "task", "test", "output_dir", "models", name=args.config)
    out = ROOT / config["output_dir"]
    if args.fresh and out.exists():
        shutil.rmtree(out)
    limit = args.limit or config.get("limit") or 10**9
    tests = prepare(config, out, limit)
    workers = config.get("workers", 4)

    for model in config["models"]:
        if args.only and model["label"] not in args.only:
            continue
        require(model, "label", "provider", "deployment", name=f"{args.config} model")
        if model["provider"] == "fireworks":
            env("FIREWORKS_API_KEY", required=not args.dry_run)
        else:
            env("FOUNDRY_BASE_URL", required=not args.dry_run)
        test = tests.get(model.get("prompt", "default")) or fail(f"unknown prompt '{model.get('prompt')}'")
        generate(model, test, out / f"predictions_{model['label']}.jsonl", workers, args.dry_run)

    judge = config.get("judge")
    if judge and not args.only:
        files = [str(out / f"predictions_{label}.jsonl") for label in judge["candidates"]]
        output = out / f"judgments_{judge['name']}.jsonl"
        for attempt in range(1, 4):
            py("benchmark/judge.py", "judge", "--test", str(tests["default"]), "--predictions", *files,
               "--output", str(output), "--model", judge["deployment"], "--workers", "3", "--resume", dry_run=args.dry_run, check=False)
            if args.dry_run or read_ids(output, ok_only=False) >= read_ids(tests["default"]):
                break
        py("benchmark/judge.py", "summarize", "--judgments", str(output), "--output", str(out / f"judge_summary_{judge['name']}.json"), dry_run=args.dry_run, quiet=True)

    if args.dry_run:
        return
    predictions = [str(p) for p in sorted(out.glob("predictions_*.jsonl"))]
    if config["task"] == "banking77":
        py("benchmark/banking77.py", "score", "--predictions", *predictions, "--output", str(out / "metrics.json"))
    else:
        py("benchmark/decision_metrics.py", "--test", str(tests["default"]), "--predictions", *predictions,
           "--output", str(out / "decision_metrics_all_models.json"), "--table")
        py("benchmark/overlap_metrics.py", "--predictions", *predictions, "--output", str(out / "metrics_all_models.json"), quiet=True)
        py("benchmark/build_report.py", "--dir", str(out), quiet=True)
    print(f"\nResults: {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
