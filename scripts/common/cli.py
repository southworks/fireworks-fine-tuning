"""Run external commands and repo scripts, printing each command first (secrets masked)."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys

from common.paths import ROOT


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
        }.get(command[0], ""))
    result = subprocess.run(command, text=True, capture_output=capture)
    if check and result.returncode:
        detail = (result.stderr or result.stdout or "").strip() if capture else ""
        fail(f"command failed ({result.returncode}){': ' + detail[:2000] if detail else ''}")
    return result


def py(script: str, *args: str, dry_run: bool = False, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    """Run a repo script; quiet=True hides its stdout unless it fails."""
    result = run([sys.executable, str(ROOT / "scripts" / script), *args], dry_run=dry_run, check=False, capture=quiet)
    if quiet and result.returncode:
        print(result.stdout, result.stderr, sep="\n", file=sys.stderr)
    if check and result.returncode:
        fail(f"{script} failed ({result.returncode})")
    return result
