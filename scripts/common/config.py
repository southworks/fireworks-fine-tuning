"""Load .env and TOML configs (with ${VAR} expansion from the environment)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from common.cli import fail
from common.paths import ROOT


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


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        fail(f"Config not found: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib  # Python 3.11+
        data = tomllib.loads(text)
    except ModuleNotFoundError:
        data = _parse_simple_toml(text, path)
    return _expand_env(data, path.name)


def _expand_env(value: Any, where: str) -> Any:
    """Replace ${VAR} in config strings with values from the environment / .env (required)."""
    if isinstance(value, dict):
        return {key: _expand_env(item, where) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item, where) for item in value]
    if isinstance(value, str):
        def resolve(match: re.Match) -> str:
            name = match.group(1)
            if not os.environ.get(name):
                fail(f"{where} needs {name}. Set it in .env (see .env.example).")
            return os.environ[name]
        return re.sub(r"\$\{([A-Z0-9_]+)\}", resolve, value)
    return value


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
