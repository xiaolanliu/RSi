"""Explicit live-GPT simulation test with a marked injected review trigger.

Natural RSI scores and alarms are preserved. This tests real API diagnosis,
bounded native execution and fresh VLA inference, not detector accuracy.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fold_clothes_live.local.toml")
    parser.add_argument("--resources", default="resources.local.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--at-step", type=int, default=175)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--allow-live-gpt", action="store_true")
    args = parser.parse_args()
    if not args.allow_live_gpt:
        raise ValueError("This test makes paid provider calls; explicitly pass --allow-live-gpt")
    if args.at_step < 1:
        raise ValueError("Trigger after the first VLA inference")
    args.mode = "live"
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    record = dict(test_only=True, injected_step=args.at_step, live_gpt=True, passed=False,
                  purpose="real Responses call and native handoff; not detection accuracy or VLA-only success")
    original = CausalMonitor.observe
    def observe(self, obs):
        value = original(self, obs)
        value["natural_alarm"] = value["alarm"]
        value["test_only_alarm_override"] = obs.step == args.at_step
        value["alarm"] = bool(value["alarm"] or obs.step == args.at_step)
        return value
    try:
        with patch.object(CausalMonitor, "observe", observe):
            evaluate(args)
        events = [json.loads(line) for line in (output/"events.jsonl").read_text().splitlines()]
        entries = [event for event in events if event.get("handoff", {}).get("to") == "gpt"]
        returns = [event for event in events if event.get("handoff", {}).get("to") == "vla"]
        import numpy as np
        inferred_at = []
        for path in sorted((output/"vla").glob("chunk_*.npz")):
            with np.load(path) as data:
                inferred_at.append(int(data["observation_step"]))
        assert entries and returns, "No accepted GPT recovery and VLA return were observed"
        assert all(event["step"] in inferred_at for event in returns), "VLA did not re-infer at recovery return"
        assert all(event["acknowledged"] and event["next_step"] == event["step"]+1 for event in events)
        record.update(passed=True, gpt_execution_steps=sum(e["source"] == "gpt" for e in events),
            recovery_entries=[e["step"] for e in entries], fresh_vla_returns=[e["step"] for e in returns],
            natural_alarm_steps=sum(e["risk"]["natural_alarm"] for e in events),
            discarded_vla_actions=[e["discarded_vla_actions"] for e in entries])
    except BaseException as error:
        record["error_class"] = type(error).__name__
        raise
    finally:
        if output.exists():
            attempts = sorted((output/"recovery").glob("api_attempt_*.json"))
            record["api_attempts"] = len(attempts)
            record["api_usage"] = [json.loads(p.read_text()) for p in attempts]
            (output/"control_test.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
