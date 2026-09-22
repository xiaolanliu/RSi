"""Native integration fault injection, never an OOD accuracy/success benchmark.

Runs the real pi05, Wan, RSI and RoboDojo. Overrides the alarm bit at one known
step solely to test native command ownership. Raw scores and natural alarm bits
are retained. No GPT call is made; mock recovery is a three-step joint hold.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi_loop.cli import evaluate
from rsi_loop.monitor import CausalMonitor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--resources", default="resources.local.json")
    p.add_argument("--config", default="configs/loop.toml")
    p.add_argument("--at-step", type=int, default=32)
    args = p.parse_args()
    if args.at_step < 1:
        raise ValueError("Inject after the first VLA inference")
    args.mode, args.max_steps, args.allow_live_gpt = "mock", args.at_step+28, False
    original = CausalMonitor.observe
    def test_observe(self, obs):
        value = original(self, obs)
        value["natural_alarm"] = value["alarm"]
        value["alarm"] = obs.step == args.at_step
        value["test_only_alarm_override"] = True
        return value
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    record = dict(test_only=True, api_calls=0, injected_step=args.at_step,
                  purpose="native command handoff, not detection accuracy or recovery skill")
    try:
        with patch.object(CausalMonitor, "observe", test_observe):
            evaluate(args)
        events = [json.loads(line) for line in (output/"events.jsonl").read_text().splitlines()]
        by_step = {event["step"]: event for event in events}
        start = by_step[args.at_step]
        assert start["source"] == "mock" and start["handoff"]["to"] == "mock"
        assert all(by_step[args.at_step+i]["source"] == "mock" for i in range(3))
        resumed = by_step[args.at_step+3]
        assert resumed["source"] == "vla" and resumed["handoff"]["to"] == "vla"
        import numpy as np
        inferred_at = []
        for file in sorted((output/"vla").glob("chunk_*.npz")):
            with np.load(file) as values:
                inferred_at.append(int(values["observation_step"]))
        assert args.at_step+3 in inferred_at
        assert all(e["acknowledged"] and e["next_step"] == e["step"]+1 for e in events)
        record.update(passed=True, discarded_vla_actions=start["discarded_vla_actions"],
                      resumed_at=args.at_step+3, fresh_inference_steps=inferred_at)
    finally:
        if output.exists():
            (output/"control_test.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
