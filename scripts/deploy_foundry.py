#!/usr/bin/env python3
"""Ensure the Microsoft Foundry deployments listed in a TOML config exist (Azure CLI).

  python3 scripts/deploy_foundry.py --config configs/foundry/models.toml

Unlike Fireworks resources, Foundry deployments are reused when they already exist: they are
pay-per-token, so an existing deployment costs nothing and recreating it would only lose quota.
--force deletes and recreates them. A blank `version` uses the newest version the account offers.
Needs `az login`, AZ_RESOURCE_GROUP and AZ_FOUNDRY_ACCOUNT. Prints the FOUNDRY_BASE_URL for .env.
"""

from __future__ import annotations

import argparse
import json

from common import env, fail, load_config, load_env, run


def az_json(*args: str) -> object | None:
    result = run(["az", *args, "-o", "json"], capture=True, check=False)
    return json.loads(result.stdout or "null") if result.returncode == 0 else None


def newest_version(group: str, account: str, model: str) -> str:
    models = az_json("cognitiveservices", "account", "list-models", "-g", group, "-n", account) or []
    versions = sorted({m.get("version") for m in models if m.get("name") == model and m.get("version")})
    if not versions:
        fail(f"model {model} is not offered on account {account} (check the region) — set `version` explicitly")
    return versions[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="delete and recreate existing deployments")
    parser.add_argument("--delete", action="store_true", help="delete the deployments listed in the config")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_env()
    config = load_config(args.config)
    group, account = env("AZ_RESOURCE_GROUP", required=True), env("AZ_FOUNDRY_ACCOUNT", required=True)
    base = ["-g", group, "-n", account]

    for item in config.get("deployments", []):
        name, model = item["name"], item.get("model", item["name"])
        current = None if args.dry_run else az_json("cognitiveservices", "account", "deployment", "show", *base, "--deployment-name", name)
        if args.delete or (current and args.force):
            if current or args.dry_run:
                run(["az", "cognitiveservices", "account", "deployment", "delete", *base, "--deployment-name", name], dry_run=args.dry_run)
            if args.delete:
                continue
        elif current:
            info = current.get("properties", {}).get("model", {})
            print(f"reuse {name}: {info.get('name')} {info.get('version')}")
            continue
        version = item.get("version") or ("<newest>" if args.dry_run else newest_version(group, account, model))
        run(["az", "cognitiveservices", "account", "deployment", "create", *base, "--deployment-name", name,
             "--model-name", model, "--model-version", version, "--model-format", "OpenAI",
             "--sku-name", item.get("sku", "GlobalStandard"), "--sku-capacity", str(item.get("capacity", 50))],
            dry_run=args.dry_run)

    if not args.delete and not args.dry_run:
        endpoint = (az_json("cognitiveservices", "account", "show", *base) or {}).get("properties", {}).get("endpoint", "").rstrip("/")
        print(f"\nPut in .env:\n  FOUNDRY_BASE_URL={endpoint}/openai/v1\n  FOUNDRY_API_KEY=<az cognitiveservices account keys list -g {group} -n {account}>")


if __name__ == "__main__":
    main()
