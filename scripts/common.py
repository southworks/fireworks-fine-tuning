"""Shared helpers for the process scripts: .env loading, TOML configs, and CLI calls.

Every external call goes through `run()`, which prints the exact command before running it
(API keys masked) and supports dry runs, so the scripts double as documentation of the
firectl / az commands behind each step.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records" / "fireworks"


# ---------------------------------------------------------------- environment

def load_env(path: Path = ROOT / ".env") -> None:
    """Load KEY=value lines from .env without overriding variables already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        fail(f"Missing {name}. Set it in .env (see .env.example).")
    return value or ""


def account() -> str:
    return env("FW_ACCOUNT", "southworks")


# ---------------------------------------------------------------- configs

def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        fail(f"Config not found: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib  # Python 3.11+
        return tomllib.loads(text)
    except ModuleNotFoundError:
        return _parse_simple_toml(text, path)


def _parse_value(raw: str, where: str) -> Any:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        return [_parse_value(item, where) for item in _split_list(inner)] if inner else []
    if raw.startswith('"') and raw.endswith('"'):
        return json.loads(raw)
    if raw in ("true", "false"):
        return raw == "true"
    try:
        return float(raw) if any(c in raw for c in ".eE") else int(raw.replace("_", ""))
    except ValueError:
        fail(f"{where}: unsupported value {raw!r} (Python 3.11+ reads full TOML)")


def _split_list(inner: str) -> list[str]:
    items, current, quoted = [], "", False
    for char in inner:
        if char == '"':
            quoted = not quoted
        if char == "," and not quoted:
            items.append(current)
            current = ""
        else:
            current += char
    if current.strip():
        items.append(current)
    return [item for item in items if item.strip()]


def _parse_simple_toml(text: str, path: Path) -> dict[str, Any]:
    """Subset of TOML used by this repo's configs (tables, arrays of tables, scalars, flat arrays)."""
    data: dict[str, Any] = {}
    target = data
    for number, line in enumerate(text.splitlines(), 1):
        line = re.sub(r'\s+#[^"]*$', "", line).strip()
        if not line or line.startswith("#"):
            continue
        where = f"{path.name}:{number}"
        if line.startswith("[[") and line.endswith("]]"):
            target = {}
            data.setdefault(line[2:-2].strip(), []).append(target)
        elif line.startswith("[") and line.endswith("]"):
            target = data.setdefault(line[1:-1].strip(), {})
        elif "=" in line:
            key, raw = line.split("=", 1)
            target[key.strip()] = _parse_value(raw, where)
        else:
            fail(f"{where}: cannot parse {line!r}")
    return data


def require(config: dict, *keys: str, name: str = "config") -> None:
    missing = [key for key in keys if config.get(key) in (None, "")]
    if missing:
        fail(f"{name}: missing {', '.join(missing)}")


# ---------------------------------------------------------------- CLI calls

def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _masked(command: list[str]) -> str:
    shown, hide = [], False
    for part in command:
        part = "python3" if part == sys.executable else part.replace(str(ROOT) + "/", "")
        shown.append("***" if hide else part)
        hide = part in ("--api-key",)
    return " ".join(shlex.quote(part) for part in shown)


def run(command: list[str], dry_run: bool = False, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {_masked(command)}", flush=True)
    if dry_run:
        return subprocess.CompletedProcess(command, 0, "", "")
    if not shutil.which(command[0]):
        fail(f"{command[0]} not found. " + {
            "firectl": "Install it: https://docs.fireworks.ai/tools-sdks/firectl/firectl",
            "az": "Install it (https://learn.microsoft.com/cli/azure/install-azure-cli), then run az login",
        }.get(command[0], ""))
    result = subprocess.run(command, text=True, capture_output=capture)
    if check and result.returncode:
        detail = (result.stderr or result.stdout or "").strip() if capture else ""
        fail(f"command failed ({result.returncode}){': ' + detail[:2000] if detail else ''}")
    return result


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


def py(script: str, *args: str, dry_run: bool = False, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    """Run a repo script; quiet=True hides its stdout unless it fails."""
    result = run([sys.executable, str(ROOT / "scripts" / script), *args], dry_run=dry_run, check=False, capture=quiet)
    if quiet and result.returncode:
        print(result.stdout, result.stderr, sep="\n", file=sys.stderr)
    if check and result.returncode:
        fail(f"{script} failed ({result.returncode})")
    return result
