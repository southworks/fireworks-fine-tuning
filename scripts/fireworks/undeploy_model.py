#!/usr/bin/env python3
"""Delete Fireworks deployments created from deployment configs (stops GPU billing).

  python3 scripts/fireworks/undeploy_model.py --config configs/deployment/qwen3-4b-base.toml
  python3 scripts/fireworks/undeploy_model.py --all          # every config in configs/deployment/
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make scripts/ importable

from common.config import load_config, load_env
from common.paths import ROOT
from fireworks.cli import exists, firectl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--config", action="append")
    target.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_env()
    paths = sorted((ROOT / "configs" / "deployment").glob("*.toml")) if args.all else args.config
    for path in paths:
        name = load_config(path)["name"]
        if args.dry_run or exists("deployment", name):
            firectl("deployment", "delete", name, dry_run=args.dry_run)
        else:
            print(f"{name}: not deployed")


if __name__ == "__main__":
    main()
