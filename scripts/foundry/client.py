"""Minimal Microsoft Foundry Responses API client (standard library).

Environment (never written to disk):
  FOUNDRY_BASE_URL   e.g. https://<resource>.services.ai.azure.com/openai/v1
  FOUNDRY_API_KEY    Foundry key (AZURE_AI_API_KEY is accepted as a fallback)
The Responses endpoint is $FOUNDRY_BASE_URL/responses unless an explicit endpoint is given.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


def response_text(body: dict[str, Any]) -> str:
    if body.get("output_text"):
        return body["output_text"]
    parts = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    parts.append(content.get("text", ""))
    return "".join(parts)


def post_responses(endpoint: str, api_key: str, payload: dict[str, Any], retries: int = 6) -> tuple[str, dict, dict]:
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "api-key": api_key,
        "Authorization": f"Bearer {api_key}",
    }
    for attempt in range(retries + 1):
        request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
            return response_text(body), body.get("usage", {}), body
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt == retries:
                raise RuntimeError(f"Foundry HTTP {exc.code}: {message[:1000]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Foundry request failed: {exc}") from exc
        time.sleep(min(2**attempt, 30))
    raise RuntimeError("Foundry request failed")


def foundry_key() -> str:
    key = os.environ.get("FOUNDRY_API_KEY") or os.environ.get("AZURE_AI_API_KEY")
    if not key:
        raise SystemExit("Set FOUNDRY_API_KEY; the key is never written to disk.")
    return key


def foundry_endpoint(value: str | None) -> str:
    if value:
        return value
    base = os.environ.get("FOUNDRY_BASE_URL")
    if not base:
        raise SystemExit("Set FOUNDRY_BASE_URL or pass --endpoint.")
    return base.rstrip("/") + "/responses"
