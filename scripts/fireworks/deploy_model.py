#!/usr/bin/env python3
"""Create a Fireworks on-demand deployment for one model from a TOML config.

  python3 scripts/fireworks/deploy_model.py --config configs/deployment/qwen3-4b-base.toml

Each model gets its own deployment: the base model or a fine-tuned model. The deployment ID is the
config `name`, so it is addressed as accounts/<account>/deployments/<name>. Fixed names: an existing
deployment stops the script, unless --force deletes it first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make scripts/ importable

from common.config import load_config, load_env, require
from fireworks.cli import deployment_path, firectl, model_path, replace_or_fail

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
    require(config, "name", "model", name=args.config)
    name = config["name"]

    replace_or_fail("deployment", name, args.force, args.dry_run)
    command = ["deployment", "create", model_path(config["model"]), "--deployment-id", name,
               "--deployment-shape", str(config.get("shape") or "default"),
               "--min-replica-count", str(config.get("min_replicas", 0)),
               "--max-replica-count", str(config.get("max_replicas", 1)), "--wait"]
    for key, flag in OPTIONAL.items():
        if config.get(key) not in (None, ""):
            command += [flag, str(config[key])]
    firectl(*command, dry_run=args.dry_run)
    print(f"\nDeployment: {deployment_path(name)}")


if __name__ == "__main__":
    main()
