"""Offline handoff tests: VLA runs until OOD, GPT recovers once, VLA resumes."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agilex_control.agent_in_loop import (
    RECOVERY_SYSTEM_PROMPT,
    AgentInLoop,
    ChunkedPi05,
    GptTakeover,
    PolicyVla,
    ScriptedRecovery,
    ScriptedVla,
    SequenceOod,
    piper_deg_mm_to_rsi_state14,
    recovery_messages,
    replay_observations,
    summarize_events,
    validate_recovery,
)
from agilex_control.rsi_observe import assert_rsi_state14, audit_unit_handoff, shared_gpu_budget


class StateTests(unittest.TestCase):
    def test_piper_degrees_become_rsi_radians_and_metres(self):
        values = {
            "left_joint_1.pos": 180.0,
            **{f"left_joint_{i}.pos": 0.0 for i in range(2, 7)},
            **{f"right_joint_{i}.pos": 0.0 for i in range(1, 7)},
            "left_gripper.pos": 70.0,
            "right_gripper.pos": 0.0,
        }
        state = piper_deg_mm_to_rsi_state14(values)
        self.assertEqual(state.shape, (14,))
        self.assertAlmostEqual(float(state[0]), np.pi, places=5)
        self.assertAlmostEqual(float(state[12]), 0.07, places=5)
        self.assertAlmostEqual(float(state[13]), 0.0, places=5)
        report = audit_unit_handoff()
        self.assertAlmostEqual(report["left_gripper_m"], 0.07, places=5)
        self.assertAlmostEqual(report["left_joint_1_rad"], float(np.pi), places=5)
        self.assertEqual(report["gpt_recovery"], "joint_deg and xyz_mm, not the RSi radian vector")
        assert_rsi_state14(state)
        with self.assertRaises(ValueError):
            assert_rsi_state14(np.array([180.0] + [0.0] * 11 + [70.0, 0.0], dtype=np.float32))

    def test_pi05_and_wan_do_not_fit_on_8gb(self):
        budget = shared_gpu_budget(8188)
        self.assertFalse(budget["fits"])
        self.assertLess(budget["leftover_mib"], budget["wan_peak_mib"])
        self.assertGreater(budget["weights_plus_wan_mib"], 8188)


class HandoffTests(unittest.TestCase):
    def test_vla_runs_until_alarm_then_gpt_returns_control(self):
        states = np.zeros((12, 14), dtype=np.float32)
        visual = np.zeros((12, 48), dtype=np.float32)
        loop = AgentInLoop(
            ScriptedVla(),
            SequenceOod(alarm_at=4),
            ScriptedRecovery(),
            cooldown_frames=2,
        )
        replay_observations(loop, states, visual, tail_after_handback=2)
        summary = summarize_events(loop.events)
        self.assertEqual(summary["first_alarm_frame"], 4)
        self.assertEqual(summary["gpt_commands"], 1)
        self.assertGreaterEqual(summary["vla_commands"], 4)
        self.assertEqual(summary["final_owner"], "vla")
        self.assertEqual(summary["handback_frames"], [4])
        self.assertIn("交还给本地 VLA", RECOVERY_SYSTEM_PROMPT)
        messages = recovery_messages("demo.mp4", {"alarm": True})
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("demo.mp4", messages[1]["content"])

    def test_cooldown_blocks_immediate_retake(self):
        states = np.zeros((8, 14), dtype=np.float32)
        visual = np.zeros((8, 48), dtype=np.float32)
        loop = AgentInLoop(
            ScriptedVla(),
            SequenceOod(alarm_at=1),
            ScriptedRecovery(),
            cooldown_frames=10,
        )
        events = replay_observations(loop, states, visual, tail_after_handback=4)
        gpt = [event for event in events if event["command"].get("decision") == "phase"]
        self.assertEqual(len(gpt), 1)
        self.assertEqual(events[-1]["owner_after"], "vla")

    def test_loaded_chain_is_vla_then_alarm_then_gpt(self):
        calls = []

        def infer(observation):
            calls.append(("vla", observation["frame"]))
            return {"chunk": True}

        def decide(messages):
            calls.append(("gpt", messages[1]["content"]))
            return {
                "decision": "phase",
                "reason": "物体掉落",
                "hand_back": True,
                "phase": {
                    "steps": [{
                        "primitive": "move_right",
                        "arguments": {"target": {"xyz_mm": [0, 0, 10]}, "duration_s": 5},
                    }],
                },
            }

        states = np.zeros((8, 14), dtype=np.float32)
        visual = np.zeros((8, 48), dtype=np.float32)
        loop = AgentInLoop(
            PolicyVla(infer),
            SequenceOod(alarm_at=3),
            GptTakeover(decide, "demo.mp4"),
            cooldown_frames=10,
        )
        replay_observations(loop, states, visual, tail_after_handback=2)
        kinds = [item[0] for item in calls]
        gpt_at = kinds.index("gpt")
        self.assertGreater(gpt_at, 0)
        self.assertEqual(kinds[gpt_at - 1], "vla")
        self.assertIn("vla", kinds[gpt_at + 1:])
        self.assertIn("demo.mp4", calls[gpt_at][1])
        self.assertEqual(loop.events[3]["ood"]["alarm"], True)
        self.assertEqual(loop.events[3]["command"]["decision"], "phase")
        with self.assertRaises(ValueError):
            validate_recovery({"decision": "phase", "hand_back": True, "phase": {"steps": [
                {"primitive": "move_right", "arguments": {"target": {"state14": [0] * 14}}},
            ]}})

    def test_chunk_refills_only_when_empty_and_clears_after_takeover(self):
        calls = []

        def infer_chunk(observation):
            del observation
            calls.append("infer")
            return [{"a": 0}, {"a": 1}]

        vla = ChunkedPi05(infer_chunk)
        loop = AgentInLoop(vla, SequenceOod(alarm_at=2), ScriptedRecovery(), cooldown_frames=5)
        states = np.zeros((6, 14), dtype=np.float32)
        visual = np.zeros((6, 48), dtype=np.float32)
        replay_observations(loop, states, visual, tail_after_handback=2)
        self.assertEqual(calls, ["infer", "infer"])
        self.assertEqual(loop.events[0]["command"]["action"], {"a": 0})
        self.assertEqual(loop.events[1]["command"]["action"], {"a": 1})


if __name__ == "__main__":
    unittest.main()
