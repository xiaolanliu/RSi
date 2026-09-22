"""VLA-first loop with RSi OOD takeover and a short GPT recovery.

``load_chain`` is the runtime that a 4090 host should start: each quiet tick
calls the local pi0.5 ``infer``, then RSi ``OnlineMonitor``. ``alarm`` is the
bundle's calibration-tail decision (default worst 5%). On alarm, GPT receives
the demo and the alarm note, returns one short deg/mm phase, and control
returns to pi0.5.

RSi does not move the robot. GPT does not stay in the loop. 5555/5556 are not
used. Wan2.2 visual features are required for a live alarm; recorded fixtures
already contain ``visual48``. The 9MB OOD head stays on CPU. The VAE does not
share the 8GB card with the Pi0.5 JAX pool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .rsi_observe import assert_rsi_state14, piper_deg_mm_to_rsi_state14

RECOVERY_SYSTEM_PROMPT = """你是短时恢复控制器，不是全程策略。
先根据当前图像和上文示范视频判断现场：物体是否掉落、夹持是否失败、布料是否滑出夹爪。
只给出一段很短的恢复动作，然后立刻把主导权交还给本地 VLA（pi0.5）。
不要规划完整任务，不要反复观察，不要调用 geometry。

只输出一个 JSON：
{
  "decision": "phase",
  "reason": "看见了什么错误，恢复要做什么",
  "hand_back": true,
  "phase": {
    "steps": [{"primitive": "move_right", "arguments": {"target": {"xyz_mm": [0,0,0]}, "duration_s": 5, "settle_s": 1}}],
    "observe_after": true,
    "context": {"purpose": "short_recovery"}
  }
}
phase.steps 最多 2 步。hand_back 必须为 true。
"""

class PolicyVla:
    """One tick of the local pi0.5 server. ``infer`` owns the websocket call."""

    def __init__(self, infer):
        self.infer = infer

    def step(self, observation):
        action = self.infer(observation)
        return {
            "owner": "vla",
            "decision": "pi05_step",
            "frame": observation.get("frame"),
            "action": action,
            "reason": "local pi0.5 chunk",
        }


def validate_recovery(decision):
    """Accept one short deg/mm phase that returns control to pi0.5."""
    if not isinstance(decision, dict):
        raise ValueError("GPT recovery must be a JSON object")
    if decision.get("decision") != "phase" or decision.get("hand_back") is not True:
        raise ValueError("GPT recovery must be decision=phase with hand_back true")
    phase = decision.get("phase")
    steps = phase.get("steps") if isinstance(phase, dict) else None
    if not isinstance(steps, list) or not 1 <= len(steps) <= 2:
        raise ValueError("GPT recovery phase must contain 1 or 2 steps")
    for step in steps:
        arguments = step.get("arguments") if isinstance(step, dict) else None
        target = arguments.get("target") if isinstance(arguments, dict) else None
        if not isinstance(target, dict):
            raise ValueError("GPT recovery step needs arguments.target")
        if "state14" in target or any(key.endswith("_rad") for key in target):
            raise ValueError("GPT recovery targets stay in joint_deg / xyz_mm / gripper_mm")
        if not any(key in target for key in ("joint_deg", "xyz_mm", "gripper_mm")):
            raise ValueError("GPT recovery target needs joint_deg, xyz_mm, or gripper_mm")
    return decision


class GptTakeover:
    """On alarm, ask GPT for one short recovery. ``decide`` performs the HTTP call."""

    def __init__(self, decide, demo_path=None):
        self.decide = decide
        self.demo_path = None if demo_path is None else str(demo_path)

    def begin(self, observation, ood):
        note = {
            "frame": observation.get("frame"),
            "image": observation.get("image"),
            "alarm": None if ood is None else ood.get("alarm"),
            "risk_score": None if ood is None else ood.get("risk_score"),
            "risk_percentile": None if ood is None else ood.get("risk_percentile"),
            "threshold": None if ood is None else ood.get("threshold"),
        }
        messages = recovery_messages(self.demo_path, note)
        decision = validate_recovery(self.decide(messages))
        decision = dict(decision)
        decision["demo"] = self.demo_path
        decision["ood_frame"] = None if ood is None else ood.get("frame_index")
        return decision

    def continue_(self, observation):
        del observation
        return {"decision": "hand_back", "reason": "恢复步已结束", "hand_back": True}


class ChunkedPi05:
    """Hold one pi0.5 action chunk. Call the server only when it is empty.

    After GPT returns control, ``reset`` drops the old chunk so the next tick
    infers from the recovered scene.
    """

    def __init__(self, infer_chunk):
        self.infer_chunk = infer_chunk
        self.pending = []

    def reset(self):
        self.pending = []

    def step(self, observation):
        if not self.pending:
            chunk = self.infer_chunk(observation)
            if not isinstance(chunk, (list, tuple)) or len(chunk) < 1:
                raise ValueError("pi0.5 infer_chunk must return a non-empty action list")
            self.pending = list(chunk)
        return {
            "owner": "vla",
            "decision": "pi05_step",
            "frame": observation.get("frame"),
            "action": self.pending.pop(0),
            "reason": "local pi0.5 chunk",
        }


class ScriptedVla:
    """Records that pi0.5 would have owned this tick. Does not load OpenPI."""

    def __init__(self):
        self.steps = 0

    def step(self, observation):
        self.steps += 1
        return {
            "owner": "vla",
            "decision": "pi05_step",
            "frame": observation.get("frame"),
            "reason": "local pi0.5 chunk",
        }


class ScriptedRecovery:
    """One short phase, then hand the loop back. No network call."""

    def __init__(self, demo_path=None):
        self.demo_path = None if demo_path is None else str(demo_path)

    def begin(self, observation, ood):
        return {
            "decision": "phase",
            "reason": "OOD 报警，按示范做一次短恢复后交还 VLA",
            "hand_back": True,
            "demo": self.demo_path,
            "ood_frame": None if ood is None else ood.get("frame_index"),
            "phase": {
                "steps": [{
                    "primitive": "move_right",
                    "arguments": {
                        "target": {"joint_deg": [0, 20, -20, 0, 20, 0]},
                        "duration_s": 5,
                        "settle_s": 1,
                    },
                }],
                "observe_after": True,
                "context": {"purpose": "short_recovery"},
            },
        }

    def continue_(self, observation):
        del observation
        return {"decision": "hand_back", "reason": "恢复步已结束", "hand_back": True}


class RsiOodHead:
    """Streaming wrapper around the cloned RSi V11 bundle."""

    def __init__(self, checkpoint, rsi_root, device="cpu", layout="joints12_grippers2"):
        root = str(Path(rsi_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from agent_closed_loop.monitor import OnlineMonitor

        self.monitor = OnlineMonitor(str(checkpoint), device=device)
        self.layout = layout

    def reset(self):
        self.monitor.reset()

    def step(self, state14, visual48):
        state14 = assert_rsi_state14(state14)
        result = self.monitor.step(state14, visual48, state_layout=self.layout)
        tail = result.get("risk_percentile")
        return {
            "alarm": bool(result["alarm"]),
            "frame_index": int(result["frame_index"]),
            "risk_score": float(result["risk_score"]),
            "risk_percentile": None if tail is None else float(tail),
            "confirmed_phase": int(result["confirmed_phase"]),
            "threshold": float(result["threshold"]),
        }


class SequenceOod:
    """Test double: alarm becomes true at ``alarm_at`` and stays true."""

    def __init__(self, alarm_at):
        self.alarm_at = int(alarm_at)
        self.frame = 0

    def reset(self):
        self.frame = 0

    def step(self, state14, visual48):
        del state14, visual48
        index = self.frame
        self.frame += 1
        return {
            "alarm": index >= self.alarm_at,
            "frame_index": index,
            "risk_score": 1.0 if index >= self.alarm_at else 0.0,
            "confirmed_phase": 1,
            "threshold": 0.5,
        }


class AgentInLoop:
    """Own the tick: VLA unless RSi alarms, then a bounded GPT recovery."""

    def __init__(self, vla, ood, gpt, max_recovery_steps=2, cooldown_frames=30):
        if max_recovery_steps < 1:
            raise ValueError("max_recovery_steps must be >= 1")
        self.vla = vla
        self.ood = ood
        self.gpt = gpt
        self.max_recovery_steps = int(max_recovery_steps)
        self.cooldown_frames = int(cooldown_frames)
        self.owner = "vla"
        self.cooldown = 0
        self.recovery_steps = 0
        self.events = []

    def tick(self, observation):
        handed_back = False
        ood = None
        if self.owner == "gpt":
            command = self.gpt.continue_(observation)
            self.recovery_steps += 1
            owner_before = "gpt"
            handed_back = bool(command.get("hand_back") or self.recovery_steps >= self.max_recovery_steps)
        else:
            ood = self.ood.step(observation["state14"], observation["visual48"])
            if ood["alarm"] and self.cooldown <= 0:
                command = self.gpt.begin(observation, ood)
                self.recovery_steps = 1
                owner_before = "vla"
                handed_back = bool(command.get("hand_back") or self.recovery_steps >= self.max_recovery_steps)
            else:
                if self.cooldown > 0:
                    self.cooldown -= 1
                command = self.vla.step(observation)
                owner_before = "vla"
        if handed_back:
            self.owner = "vla"
            self.cooldown = self.cooldown_frames
            self.ood.reset()
            reset = getattr(self.vla, "reset", None)
            if callable(reset):
                reset()
            command = dict(command)
            command["handed_back"] = True
        elif owner_before == "vla" and ood is not None and ood["alarm"]:
            self.owner = "gpt"
        event = {
            "owner_before": owner_before,
            "owner_after": self.owner,
            "handed_back": handed_back,
            "ood": ood,
            "command": command,
            "frame": observation.get("frame"),
        }
        self.events.append(event)
        return event


def replay_observations(loop, states, visual, limit=None, tail_after_handback=3):
    """Feed aligned RSi frames. Keep a few VLA ticks after the first hand-back."""
    total = len(states)
    if limit is not None:
        total = min(total, int(limit))
    tail = None
    for index in range(total):
        event = loop.tick({
            "frame": index,
            "state14": np.asarray(states[index], dtype=np.float32),
            "visual48": np.asarray(visual[index], dtype=np.float32),
        })
        if event["handed_back"]:
            tail = int(tail_after_handback)
        elif tail is not None:
            tail -= 1
            if tail <= 0:
                break
    return loop.events


def load_rsi_episode(path):
    data = np.load(path)
    if "states" not in data.files or "visual" not in data.files:
        raise ValueError("RSi episode needs states[T,14] and visual[T,48]")
    return data["states"], data["visual"]


def recovery_messages(demo_note, observation_note):
    """System contract plus the demo and the live failure, for one GPT call."""
    user = {
        "demo": demo_note,
        "observation": observation_note,
        "instruction": "判断当前错误，给出不超过两步的恢复，并交还 VLA。",
    }
    return [
        {"role": "system", "content": RECOVERY_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def summarize_events(events):
    alarms = [
        event["ood"]["frame_index"]
        for event in events
        if event["ood"] and event["ood"]["alarm"] and event["owner_before"] == "vla"
    ]
    handbacks = [event["frame"] for event in events if event["handed_back"]]
    return {
        "ticks": len(events),
        "vla_commands": sum(1 for event in events if event["command"].get("decision") == "pi05_step"),
        "gpt_commands": sum(1 for event in events if event["command"].get("decision") == "phase"),
        "first_alarm_frame": None if not alarms else min(alarms),
        "handback_frames": handbacks,
        "final_owner": None if not events else events[-1]["owner_after"],
    }


def fixture_recovery(messages):
    """Offline stand-in for the GPT HTTP call. Live runs pass openai_decide instead."""
    if not messages or messages[0].get("role") != "system":
        raise ValueError("recovery call is missing the system contract")
    return {
        "decision": "phase",
        "reason": "fixture：报警后一步短恢复，然后交还 pi0.5",
        "hand_back": True,
        "phase": {
            "steps": [{
                "primitive": "move_right",
                "arguments": {
                    "target": {"xyz_mm": [0, 0, 10]},
                    "duration_s": 5,
                    "settle_s": 1,
                },
            }],
            "observe_after": True,
            "context": {"purpose": "short_recovery"},
        },
    }


def load_chain(checkpoint, rsi_root, infer, decide, demo_path=None, device="cpu"):
    """Real load order: pi0.5 infer, RSi OnlineMonitor, GPT decide."""
    return AgentInLoop(
        PolicyVla(infer),
        RsiOodHead(checkpoint, rsi_root, device=device),
        GptTakeover(decide, demo_path),
    )


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rsi-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episode", type=Path, default=None)
    parser.add_argument("--demo", type=Path, default=None, help="Prepared MP4 shown to GPT on takeover")
    parser.add_argument("--device", default="cpu", help="RSi head device. Wan VAE is separate.")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("runs/agent-in-loop/fixture.json"))
    args = parser.parse_args(argv)
    root = args.rsi_root
    checkpoint = args.checkpoint or (root / "models/v11/fold_all.pt")
    episode = args.episode or (root / "examples/episodes/pant_fail_ep000000.npz")
    states, visual = load_rsi_episode(episode)
    trace = []

    def infer(observation):
        trace.append("vla")
        return {"frame": observation.get("frame")}

    def decide(messages):
        trace.append("gpt")
        return fixture_recovery(messages)

    loop = load_chain(checkpoint, root, infer, decide, args.demo, device=args.device)
    replay_observations(loop, states, visual, limit=args.limit, tail_after_handback=5)
    summary = summarize_events(loop.events)
    gpt_at = trace.index("gpt") if "gpt" in trace else -1
    summary["chain"] = trace[:gpt_at + 1][-3:] + (["..."] if gpt_at >= 0 else [])
    summary["chain_ok"] = (
        gpt_at > 0
        and trace[gpt_at - 1] == "vla"
        and "vla" in trace[gpt_at + 1:]
        and summary["final_owner"] == "vla"
    )
    summary["demo"] = None if args.demo is None else str(args.demo)
    summary["episode"] = str(episode)
    summary["checkpoint"] = str(checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["chain_ok"] or summary["gpt_commands"] < 1:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
