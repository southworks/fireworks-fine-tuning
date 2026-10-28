#!/usr/bin/env python3
"""Create a Fireworks on-demand deployment from a TOML config.

  python3 scripts/deploy_model.py --config configs/deployment/qwen3-4b-base.toml

Two strategies (config `strategy`):
  dedicated  one deployment serving `model` (a base model, or a fine-tuned LoRA merged live)
  addons     one deployment of `base_model` with add-ons enabled; every model in `adapters` is
             loaded onto it (multi-LoRA: several fine-tuned models share one GPU; firectl only,
             the Fireworks UI does not offer it)

The deployment ID is the config `name`, so it is addressed as accounts/<account>/deployments/<name>.
Fixed names: an existing deployment stops the script, unless --force deletes it first.
"""

from __future__ import annotations

import argparse

from common import deployment_path, fail, firectl, load_config, load_env, model_path, replace_or_fail, require

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
    else:
        fail(f"{args.config}: strategy must be 'dedicated' or 'addons'")

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
    firectl(*command, dry_run=args.dry_run)

    if strategy == "addons":
        for adapter in config["adapters"]:
            firectl("model", "load-lora", model_path(adapter), "--deployment", name, "--wait", dry_run=args.dry_run)
        print("\nQuery the fine-tuned models as:")
        for adapter in config["adapters"]:
            print(f"  {model_path(adapter)}#{deployment_path(name)}")
    else:
        print(f"\nQuery as: {deployment_path(name)}")


if __name__ == "__main__":
    main()
