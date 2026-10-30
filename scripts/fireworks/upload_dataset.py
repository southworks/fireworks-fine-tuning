#!/usr/bin/env python3
"""Upload a chat-format JSONL file as a Fireworks dataset.

  python3 scripts/fireworks/upload_dataset.py --name amazon-reviews-train --path data/dataset/fireworks_v2_training.jsonl

The file is checked locally first (valid JSON, a `messages` list with an assistant turn).
Fixed names: if the dataset already exists the script stops, unless --force deletes it first.
"""

from __future__ import annotations

import argparse
import json

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make scripts/ importable

from common.cli import fail
from common.config import load_env
from common.paths import ROOT
from fireworks.cli import firectl, replace_or_fail


def check_file(path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                fail(f"{path.name}:{number}: invalid JSON ({error})")
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
                fail(f"{path.name}:{number}: expected a 'messages' list ending with an assistant turn")
            count += 1
    if count < 3:
        fail(f"{path.name}: Fireworks needs at least 3 examples")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Fireworks dataset ID")
    parser.add_argument("--path", required=True, help="local JSONL file")
    parser.add_argument("--force", action="store_true", help="delete an existing dataset with the same name first")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()
    load_env()
    path = (ROOT / args.path) if not args.path.startswith("/") else __import__("pathlib").Path(args.path)
    if not path.exists():
        fail(f"file not found: {args.path}")
    print(f"{args.path}: {check_file(path)} examples")
    replace_or_fail("dataset", args.name, args.force, args.dry_run)
    firectl("dataset", "create", args.name, str(path), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
