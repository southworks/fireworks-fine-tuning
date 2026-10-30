#!/usr/bin/env python3
"""Generate or verify SHA-256 checksums for the published evidence files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INCLUDED = ("data/banking77", "records", "data/amazon-reviews", "evaluation/banking77", "evaluation/qwen3-4b-amazon-v2")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def current() -> dict[str, str]:
    found = {}
    for name in INCLUDED:
        for path in sorted((ROOT / name).rglob("*")):
            if path.is_file() and path.name != ".DS_Store" and "reruns" not in path.parts:
                found[path.relative_to(ROOT).as_posix()] = digest(path)
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "CHECKSUMS.sha256")
    args = parser.parse_args()
    values = current()
    if args.check:
        expected = {}
        for line in args.check.read_text(encoding="utf-8").splitlines():
            checksum, name = line.split("  ", 1)
            expected[name] = checksum
        if values != expected:
            missing = sorted(set(expected) - set(values))
            added = sorted(set(values) - set(expected))
            changed = sorted(name for name in set(values) & set(expected) if values[name] != expected[name])
            raise SystemExit(f"checksum mismatch: missing={missing}, added={added}, changed={changed}")
        print(f"verified {len(values)} files")
        return 0
    args.output.write_text("".join(f"{checksum}  {name}\n" for name, checksum in values.items()), encoding="utf-8")
    print(f"wrote {len(values)} checksums to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
