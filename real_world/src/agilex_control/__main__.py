"""CLI transport for primitive calls and camera-service lifecycle."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    tool = sub.add_parser("call")
    tool.add_argument("--request", type=Path, required=True)
    tool.add_argument("--output", type=Path)
    tool.add_argument("--summary", action="store_true")
    phase = sub.add_parser("phase", help="Execute a finite list of existing primitives")
    phase.add_argument("--request", type=Path, required=True)
    phase.add_argument("--output", type=Path, required=True)
    phase.add_argument("--retry-observation", action="store_true")
    phase.add_argument("--summary", action="store_true")
    camera = sub.add_parser("cameras")
    camera.add_argument("operation", choices=["serve", "status", "stop"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.command == "cameras":
        from .cameras import serve, request

        if args.operation == "serve":
            return serve(config)
        result = request(config["runtime_dir"], {"op": args.operation})
        print(json.dumps(result, indent=2))
        return 0
    message = json.loads(sys.stdin.read() if str(args.request) == "-" else args.request.read_text())
    if args.command == "phase":
        from .phase import run_phase

        result = run_phase(config, message, args.output, retry_observation=args.retry_observation)
    else:
        from .primitives import call

        result = call(config, message["primitive"], message.get("arguments", {}))
    if args.output and args.command == "call":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.summary:
        from .report import summarize

        print(json.dumps(summarize(result, config), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
