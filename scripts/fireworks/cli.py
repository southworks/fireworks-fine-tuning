"""firectl helpers: account, JSON calls, fixed-name handling, records and model/deployment paths."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from common.cli import fail, run
from common.config import env
from common.paths import ROOT

RECORDS = ROOT / "evaluation" / "reruns" / "training"   # records of jobs run by the pipeline (git-ignored)


def account() -> str:
    return env("FW_ACCOUNT", "southworks")


def firectl(*args: str, dry_run: bool = False, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    command = ["firectl", *args, "-a", account()]
    if os.environ.get("FIREWORKS_API_KEY"):
        command += ["--api-key", os.environ["FIREWORKS_API_KEY"]]
    return run(command, dry_run=dry_run, capture=capture, check=check)


def firectl_json(*args: str) -> dict | None:
    """Return the parsed JSON of a firectl get/create call, or None if it fails (e.g. not found)."""
    result = firectl(*args, "-o", "json", capture=True, check=False)
    if result.returncode:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        fail(f"unexpected firectl output: {result.stdout[:500]}")


def exists(kind: str, resource_id: str) -> bool:
    return firectl_json(kind, "get", resource_id) is not None


def replace_or_fail(kind: str, resource_id: str, force: bool, dry_run: bool, label: str | None = None) -> None:
    """Fixed names: an existing resource is an error unless --force, which deletes it first."""
    if dry_run or not exists(kind, resource_id):
        return
    if not force:
        fail(f"{label or kind} '{resource_id}' already exists in Fireworks. Re-run with --force (make ... FORCE=1) "
             "to delete and recreate it, or change the name in the config.")
    print(f"--force: deleting existing {label or kind} '{resource_id}'")
    firectl(kind, "delete", resource_id)
    for _ in range(60):  # the ID stays reserved while the resource is being deleted
        if not exists(kind, resource_id):
            return
        time.sleep(10)
    fail(f"{label or kind} '{resource_id}' is still being deleted; try again in a few minutes")


def redact(data: dict) -> dict:
    for key in [k for k in data if k.endswith("_signed_url")]:
        data[key] = "REDACTED (time-limited signed URL; regenerate with: firectl sftj get <job> -o json)"
    return data


def save_record(name: str, data: dict) -> Path:
    RECORDS.mkdir(parents=True, exist_ok=True)
    path = RECORDS / f"{name}.json"
    path.write_text(json.dumps(redact(data), indent=2) + "\n", encoding="utf-8")
    print(f"saved {path.relative_to(ROOT)}")
    return path


def model_path(name: str) -> str:
    return name if name.startswith("accounts/") else f"accounts/{account()}/models/{name}"


def deployment_path(name: str) -> str:
    return name if name.startswith("accounts/") else f"accounts/{account()}/deployments/{name}"
