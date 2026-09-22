"""Resolve a recorded server explicitly; never connect or operate hardware."""

import argparse
import copy
import json
import shlex
from pathlib import Path

DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "config" / "servers.json"


def load_registry(path=DEFAULT_REGISTRY):
    registry = json.loads(Path(path).read_text())
    if registry.get("schema_version") != 1 or not isinstance(
        registry.get("sites"), dict
    ):
        raise ValueError("Expected server registry schema_version 1 with sites")
    return registry


def resolve_site(registry, selector):
    """Accept a site id or exact IP; reject missing, unknown or ambiguous targets."""
    if not isinstance(selector, str) or not selector.strip():
        raise ValueError("An explicit server id or IP is required")
    matches = [
        (site_id, profile)
        for site_id, profile in registry["sites"].items()
        if selector in (site_id, profile["ssh"]["host"])
    ]
    if len(matches) != 1:
        raise ValueError("Server selection must match exactly one profile: " + selector)
    site_id, profile = matches[0]
    result = dict(site_id=site_id, **copy.deepcopy(profile))
    ssh = result["ssh"]
    argv = ["ssh", "-p", str(ssh["port"]), "-l", ssh["user"], ssh["host"]]
    result["ssh_argv"] = argv
    result["ssh_command"] = shlex.join(argv)
    result["meaning"] = (
        "Recorded configuration only; no connection or hardware check performed"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    show = commands.add_parser("show")
    show.add_argument("--site", required=True, help="Exact registered server id or IP")
    args = parser.parse_args(argv)
    try:
        registry = load_registry(args.registry)
        if args.command == "list":
            result = [
                {
                    "site_id": key,
                    "host": value["ssh"]["host"],
                    "validation": value["validation"]["level"],
                }
                for key, value in registry["sites"].items()
            ]
        else:
            result = resolve_site(registry, args.site)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
