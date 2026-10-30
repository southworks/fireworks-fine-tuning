#!/usr/bin/env python3
"""Create a Fireworks on-demand deployment from a TOML config.

  python3 scripts/fireworks/deploy_model.py --config configs/deployment/qwen3-4b-base.toml

Two strategies (config `strategy`):
  dedicated  one deployment serving `model` (a base model, or a fine-tuned LoRA merged live)
  addons     one deployment of `base_model` with add-ons enabled; the fine-tuned models in
             `adapters` are loaded onto it (multi-LoRA: several fine-tuned models share one GPU;
             available through firectl, not in the Fireworks web UI). Use --adapter to load only
             some of them; an existing add-on deployment is reused, so adapters can be added one
             use case at a time:
               deploy_model.py --config configs/deployment/qwen3-4b-ft-adapters.toml --adapter ft-qwen3-4b-banking77

The deployment ID is the config `name`, so it is addressed as accounts/<account>/deployments/<name>.
Fixed names: an existing `dedicated` deployment stops the script, unless --force deletes it first.
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make scripts/ importable

from common.cli import fail
from common.config import load_config, load_env, require
from fireworks.cli import deployment_path, exists, firectl, model_path, replace_or_fail

OPTIONAL = {  # config key -> firectl deployment create flag
    "display_name": "--display-name",
    "accelerator_type": "--accelerator-type",
    "accelerator_count": "--accelerator-count",
    "region": "--region",
    "scale_to_zero_window": "--scale-to-zero-window",
    "scale_down_window": "--scale-down-window",
    "unused_auto_delete": "--unused-auto-delete-duration",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="delete an existing deployment with the same name first")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--adapter", action="append", help="addons strategy: load only this fine-tuned model (repeatable)")
    args = parser.parse_args()
    load_env()
    config = load_config(args.config)
    require(config, "name", "strategy", name=args.config)
    name, strategy = config["name"], config["strategy"]
    if strategy == "dedicated":
        require(config, "model", name=args.config)
        served = model_path(config["model"])
    elif strategy == "addons":
        require(config, "base_model", "adapters", name=args.config)
        served = config["base_model"]
        adapters = args.adapter or config["adapters"]
        unknown = [a for a in adapters if a not in config["adapters"]]
        if unknown:
            fail(f"{args.config}: {', '.join(unknown)} not listed in `adapters`")
    else:
        fail(f"{args.config}: strategy must be 'dedicated' or 'addons'")

    reuse = strategy == "addons" and not args.force and not args.dry_run and exists("deployment", name)
    if reuse:
        print(f"Reusing add-on deployment '{name}'")
    else:
        replace_or_fail("deployment", name, args.force, args.dry_run)
    command = ["deployment", "create", served, "--deployment-id", name,
               "--deployment-shape", str(config.get("shape") or "default"),
               "--min-replica-count", str(config.get("min_replicas", 0)),
               "--max-replica-count", str(config.get("max_replicas", 1)), "--wait"]
    for key, flag in OPTIONAL.items():
        if config.get(key) not in (None, ""):
            command += [flag, str(config[key])]
    if strategy == "addons":
        command.append("--enable-addons")
    if not reuse:
        firectl(*command, dry_run=args.dry_run)

    if strategy == "addons":
        for adapter in adapters:
            firectl("model", "load-lora", model_path(adapter), "--deployment", name, "--wait", dry_run=args.dry_run)
        print("\nQuery the fine-tuned models as:")
        for adapter in adapters:
            print(f"  {model_path(adapter)}#{deployment_path(name)}")
    else:
        print(f"\nQuery as: {deployment_path(name)}")


if __name__ == "__main__":
    main()
