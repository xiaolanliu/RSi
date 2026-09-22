#!/usr/bin/env python3
"""Read selected CAN interfaces using the shared get_state implementation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from agilex_control.primitives import state


def receive(interfaces, seconds):
    if not interfaces or len(set(interfaces)) != len(interfaces):
        raise ValueError("Choose distinct CAN interfaces")
    return state({"arms": {name: name for name in interfaces}}, seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interfaces", nargs="+", required=True)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = receive(args.interfaces, args.seconds)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
