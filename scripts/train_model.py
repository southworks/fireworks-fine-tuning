#!/usr/bin/env python3
"""Run a supervised fine-tuning job on Fireworks from a TOML config.

  python3 scripts/train_model.py --config configs/training/amazon-reviews.toml

The job ID and the output model use the config `name`. Fixed names: if the model or the job
already exists the script stops, unless --force deletes them first (a deployed model must be
undeployed before it can be deleted). By default it waits for the job and then saves the job
and model records to records/fireworks/ (signed URLs redacted).
"""

from __future__ import annotations

import argparse
import time

from common import firectl, firectl_json, exists, fail, load_config, load_env, model_path, replace_or_fail, require, save_record

FLAGS = {  # config key -> firectl sftj create flag
    "eval_dataset": "--evaluation-dataset",
    "epochs": "--epochs",
    "learning_rate": "--learning-rate",
    "lr_scheduler": "--learning-rate-scheduler",
    "warmup_steps": "--learning-rate-warmup-steps",
    "lora_rank": "--lora-rank",
    "batch_size_samples": "--batch-size-samples",
    "max_context_length": "--max-context-length",
    "display_name": "--display-name",
}


def build_command(config: dict) -> list[str]:
    if config.get("method", "sft") != "sft":
        fail(f"method '{config['method']}' is not supported by this script (only sft)")
    command = ["sftj", "create", "--base-model", config["base_model"], "--dataset", config["dataset"],
               "--output-model", config["name"], "--job-id", config["name"]]
    for key, flag in FLAGS.items():
        if config.get(key) not in (None, ""):
            command += [flag, str(config[key])]
    return command


def wait(job: str, poll: int) -> dict:
    while True:
        record = firectl_json("sftj", "get", job) or {}
        state = record.get("state", "UNKNOWN")
        progress = record.get("job_progress", {}).get("percent", 0)
        print(f"{time.strftime('%H:%M:%S')}  {job}  {state}  {progress}%", flush=True)
        if state == "JOB_STATE_COMPLETED":
            return record
        if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            fail(f"job {job} ended in {state}: {record.get('status', {}).get('message', '')}")
        time.sleep(poll)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="delete an existing model/job with the same name first")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--no-wait", action="store_true", help="return right after the job is created")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    load_env()
    config = load_config(args.config)
    require(config, "name", "base_model", "dataset", name=args.config)
    name = config["name"]

    if not args.dry_run:
        if not exists("dataset", config["dataset"]):
            fail(f"dataset '{config['dataset']}' not found in Fireworks; run make upload-datasets first")
        replace_or_fail("model", name, args.force, args.dry_run)
        replace_or_fail("sftj", name, args.force, args.dry_run, label="fine-tuning job")
    firectl(*build_command(config), dry_run=args.dry_run)
    if args.dry_run or args.no_wait:
        return
    job = wait(name, args.poll_seconds)
    save_record(f"sftj_{name}", job)
    save_record(f"model_{name}", firectl_json("model", "get", name) or {})
    print(f"Trained: {model_path(name)}")


if __name__ == "__main__":
    main()
