"""Hardware-free tests for the on-host OpenAI decision loop."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agilex_control.local_loop import (
    parse_json_object,
    sanitize_phase,
    validate_decision,
    compact_summary,
    run_loop,
    resolve_provider,
    load_api_key,
    message_text,
    prepare_output,
    SYSTEM_PROMPT,
    LocalLoop,
)
from agilex_control.policy_context import compile_user_content, load_task_package

CONFIG = {
    "arms": {"left": "can_left", "right": "can_right"},
    "joint_limits_deg": [[-150, 150], [0, 180], [-170, 0], [-100, 100], [-70, 70], [-120, 120]],
    "gripper_limits_mm": [0, 70],
}

MOVE = {
    "primitive": "move_right",
    "arguments": {
        "target": {"xyz_mm": [200.0, 0.0, 250.0], "rpy_deg": [180.0, 0.0, 0.0]},
        "duration_s": 4,
        "settle_s": 1,
    },
}


def observe_result():
    arm = dict(
        complete=True, stale=[], pose_mm_deg=[200, 0, 250, 180, 0, 0],
        joint_deg=[0, 10, -20, 0, 30, 0], arm_status=0, error_code=0,
        gripper_fault_bits=0, gripper_mm=20.0, gripper_effort_nm=0.1,
    )
    return {
        "primitive": "observe", "ok": True, "started_unix": 1, "finished_unix": 2,
        "error": None,
        "data": {"states": {"left": dict(arm), "right": dict(arm)}, "cameras": {}},
    }


class ParseTests(unittest.TestCase):
    def test_fenced_and_embedded_json(self):
        self.assertEqual(parse_json_object('```json\n{"decision": "done"}\n```')["decision"], "done")
        self.assertEqual(parse_json_object('note {"decision": "abort", "x": 1} trailing')["decision"], "abort")

    def test_rejects_non_object(self):
        with self.assertRaises(ValueError):
            parse_json_object("[1]")

    def test_message_text_accepts_list_content(self):
        text = message_text({
            "choices": [{"message": {"content": [{"type": "text", "text": '{"decision": "done"}'}]}}]
        })
        self.assertIn("done", text)


class ProviderTests(unittest.TestCase):
    def test_deepseek_flash_endpoint(self):
        profile = resolve_provider("deepseek")
        self.assertEqual(profile["model"], "deepseek-flash")
        self.assertEqual(profile["base_url"], "https://api.deepseek.com")
        self.assertEqual(profile["extra_body"]["thinking"]["type"], "disabled")

    def test_hardcoded_deepseek_key_is_used(self):
        with patch("agilex_control.local_loop.DEEPSEEK_API_KEY", "sk-test-deepseek"):
            self.assertEqual(load_api_key(provider="deepseek"), "sk-test-deepseek")

    def test_missing_key_fails_closed(self):
        with patch("agilex_control.local_loop.DEEPSEEK_API_KEY", ""), \
             patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                load_api_key(provider="deepseek")


class OutputTests(unittest.TestCase):
    def test_prepare_output_overwrites_existing_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "task"
            target.mkdir()
            (target / "old.json").write_text("{}")
            path = prepare_output(target)
            self.assertTrue(path.is_dir())
            self.assertFalse((path / "old.json").exists())

    def test_prepare_output_rejects_protected_paths(self):
        with self.assertRaises(ValueError):
            prepare_output("/home/agilex/lwy_astra_agx")


class ValidationTests(unittest.TestCase):
    def test_phase_strips_output_and_dry_run(self):
        cleaned = sanitize_phase(CONFIG, {
            "steps": [{
                "primitive": "move_right",
                "arguments": {**MOVE["arguments"], "output": "/tmp/x", "dry_run": True, "arm": "right"},
            }],
            "observe_after": True,
            "context": {"purpose": "lift"},
        })
        self.assertNotIn("output", cleaned["steps"][0]["arguments"])
        self.assertNotIn("dry_run", cleaned["steps"][0]["arguments"])
        self.assertNotIn("arm", cleaned["steps"][0]["arguments"])

    def test_conservative_phase_unfolds_folded_arm_instead_of_cartesian(self):
        summary = {
            "arms": {
                "right": {
                    "actual_pose_mm_deg": [54.417, 5.341, 187.243, 136.0, 73.2, 140.5],
                    "joint_deg": [5.944, 0.206, -0.151, -0.7, 16.915, 12.506],
                }
            }
        }
        cleaned = sanitize_phase(CONFIG, {
            "steps": [{
                "primitive": "move_right",
                "arguments": {
                    "target": {"xyz_mm": [114.417, 5.341, 187.243], "rpy_deg": [136.0, 73.2, 140.5]},
                    "duration_s": 4,
                    "settle_s": 1,
                },
            }],
            "observe_after": True,
            "context": {"purpose": "approach"},
        }, last_summary=summary)
        target = cleaned["steps"][0]["arguments"]["target"]
        self.assertIn("joint_deg", target)
        self.assertNotIn("xyz_mm", target)
        self.assertAlmostEqual(target["joint_deg"][1], 15.206, places=3)
        self.assertAlmostEqual(target["joint_deg"][2], -15.151, places=3)
        self.assertGreaterEqual(cleaned["steps"][0]["arguments"]["duration_s"], 5)
        self.assertFalse(cleaned["steps"][0]["arguments"]["require_cameras"])

    def test_conservative_phase_clips_cartesian_step_and_drops_rpy(self):
        summary = {
            "arms": {
                "right": {
                    "actual_pose_mm_deg": [54.417, 5.341, 187.243],
                    "joint_deg": [10, 40, -40, 0, 20, 0],
                }
            }
        }
        cleaned = sanitize_phase(CONFIG, {
            "steps": [{
                "primitive": "move_right",
                "arguments": {
                    "target": {"xyz_mm": [114.417, 5.341, 187.243], "rpy_deg": [136.0, 73.2, 140.5]},
                    "duration_s": 4,
                    "settle_s": 1,
                },
            }],
            "observe_after": True,
            "context": {},
        }, last_summary=summary)
        target = cleaned["steps"][0]["arguments"]["target"]
        self.assertNotIn("rpy_deg", target)
        self.assertAlmostEqual(target["xyz_mm"][0] - 54.417, 40.0, places=2)

    def test_phase_without_execute_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_decision(CONFIG, {
                "decision": "phase",
                "reason": "move",
                "phase": {"steps": [MOVE], "observe_after": True, "context": {}},
            }, execute=False)

    def test_dry_run_and_done_are_allowed_without_execute(self):
        dry = validate_decision(CONFIG, {
            "decision": "dry_run",
            "reason": "plan a small lift",
            "dry_run": MOVE,
        }, execute=False)
        self.assertEqual(dry["dry_run"]["primitive"], "move_right")
        done = validate_decision(CONFIG, {"decision": "done", "reason": "task looks complete"}, execute=False)
        self.assertEqual(done["decision"], "done")

    def test_unknown_primitive_rejected(self):
        with self.assertRaises(ValueError):
            sanitize_phase(CONFIG, {"steps": [{"primitive": "enable", "arguments": {}}]})

    def test_compact_summary_drops_image_blobs(self):
        compact = compact_summary({
            "ok": True,
            "arms": {"right": {"error_code": 0}},
            "images": [{"camera": "front", "rgb": "/tmp/a.jpg", "depth": "/tmp/a.npy"}],
        })
        self.assertEqual(compact["images"][0]["camera"], "front")
        self.assertNotIn("depth", compact["images"][0])


class LoopTests(unittest.TestCase):
    def test_initial_observe_then_done_never_moves(self):
        decisions = [({"decision": "done", "reason": "scene is already fine"}, {"id": "fake"})]

        def decide(*args, **kwargs):
            return decisions.pop(0)

        calls = []

        def fake_call(config, name, arguments):
            calls.append(name)
            self.assertEqual(name, "observe")
            return observe_result()

        with tempfile.TemporaryDirectory() as tmp, patch(
            "agilex_control.local_loop.call", side_effect=fake_call
        ):
            journal = run_loop(
                CONFIG, "inspect only", Path(tmp) / "run", Path(tmp) / "site.json",
                api_key="test", execute=False, yes=True, decide=decide,
            )
        self.assertEqual(journal["status"], "completed")
        self.assertEqual(calls, ["observe"])
        self.assertEqual(journal["steps"][-1]["decision"]["decision"], "done")

    def test_phase_without_execute_fails_closed(self):
        def decide(*args, **kwargs):
            return ({
                "decision": "phase",
                "reason": "lift",
                "phase": {"steps": [MOVE], "observe_after": True, "context": {}},
            }, {"id": "fake"})

        with tempfile.TemporaryDirectory() as tmp, patch(
            "agilex_control.local_loop.call", return_value=observe_result()
        ), patch("agilex_control.local_loop.run_phase") as phase:
            journal = run_loop(
                CONFIG, "lift", Path(tmp) / "run", Path(tmp) / "site.json",
                api_key="test", execute=False, yes=True, decide=decide,
            )
        self.assertEqual(journal["status"], "failed")
        self.assertIn("--execute", journal["error"])
        phase.assert_not_called()

    def test_dry_run_is_forced_true(self):
        seen = []

        def fake_call(config, name, arguments):
            seen.append((name, dict(arguments)))
            if name == "observe":
                return observe_result()
            return {
                "primitive": name, "ok": True, "started_unix": 1, "finished_unix": 2,
                "error": None, "data": {"executed": False, "plans": {}},
            }

        sequential = [
            ({"decision": "dry_run", "reason": "plan", "dry_run": MOVE}, {"id": "1"}),
            ({"decision": "done", "reason": "planned"}, {"id": "2"}),
        ]

        def decide_seq(*args, **kwargs):
            return sequential.pop(0)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "agilex_control.local_loop.call", side_effect=fake_call
        ):
            journal = run_loop(
                CONFIG, "plan", Path(tmp) / "run", Path(tmp) / "site.json",
                api_key="test", execute=False, yes=True, decide=decide_seq,
            )
        self.assertEqual(journal["status"], "completed")
        self.assertEqual(seen[1][0], "move_right")
        self.assertTrue(seen[1][1]["dry_run"])
        self.assertTrue(seen[1][1]["output"].endswith("plan.json"))

    def test_execute_dispatches_phase_once(self):
        sequential = [
            ({
                "decision": "phase",
                "reason": "small lift after plan",
                "phase": {"steps": [MOVE], "observe_after": True, "context": {"purpose": "lift"}},
            }, {"id": "1"}),
            ({"decision": "done", "reason": "moved"}, {"id": "2"}),
        ]

        def decide_seq(*args, **kwargs):
            return sequential.pop(0)

        phase_calls = []

        def fake_phase(config, request, output_dir, retry_observation=False):
            phase_calls.append(request)
            return {
                "ok": True, "status": "completed", "action_status": "completed",
                "started_unix": 1, "finished_unix": 2, "steps": [],
                "observation": {"status": "completed", "result": observe_result()},
                "output_dir": str(output_dir),
            }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "agilex_control.local_loop.call", return_value=observe_result()
        ), patch("agilex_control.local_loop.run_phase", side_effect=fake_phase):
            journal = run_loop(
                CONFIG, "lift", Path(tmp) / "run", Path(tmp) / "site.json",
                api_key="test", execute=True, yes=True, decide=decide_seq,
            )
        self.assertEqual(journal["status"], "completed")
        self.assertEqual(len(phase_calls), 1)
        self.assertEqual(phase_calls[0]["steps"][0]["primitive"], "move_right")

    def test_prompt_is_visuomotor_without_calibration(self):
        self.assertIn("禁止 geometry", SYSTEM_PROMPT)
        self.assertIn("禁止等待 calibration.json", SYSTEM_PROMPT)
        self.assertIn("上下文学习", SYSTEM_PROMPT)

    def test_icl_thread_keeps_prior_decision(self):
        seen = []
        sequential = [
            ({"decision": "observe", "reason": "need a closer look"}, {"id": "1"}),
            ({"decision": "done", "reason": "still fine"}, {"id": "2"}),
        ]

        def decide_seq(api_key, model, messages, **kwargs):
            seen.append([item["role"] for item in messages])
            return sequential.pop(0)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "agilex_control.local_loop.call", return_value=observe_result()
        ):
            journal = run_loop(
                CONFIG, "inspect only", Path(tmp) / "run", Path(tmp) / "site.json",
                api_key="test", execute=False, yes=True, decide=decide_seq,
            )
            self.assertTrue((Path(journal["output"]) / "input.json").is_file())
            self.assertTrue(journal.get("icl"))
        self.assertEqual(journal["status"], "completed")
        self.assertEqual(seen[0][:2], ["system", "user"])
        self.assertGreaterEqual(len(seen[1]), 4)
        self.assertEqual(seen[1][2], "assistant")

    def test_geometry_is_refused_in_the_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = LocalLoop(CONFIG, Path(tmp) / "run", Path(tmp) / "site.json",
                             execute=False, confirm=False)
            outcome = loop.geometry({"operation": "observation_points"})
        self.assertFalse(outcome["ok"])
        self.assertIn("不使用手眼标定", outcome["result"]["error"])

    def test_cancel_stops_after_initial_observe(self):
        import threading

        cancel = threading.Event()
        cancel.set()
        called = []

        def fake_call(config, name, arguments):
            called.append(name)
            if name == "observe":
                return observe_result()
            return {
                "primitive": name, "ok": True, "started_unix": 1, "finished_unix": 2,
                "error": None, "data": {},
            }

        def decide(*args, **kwargs):
            raise AssertionError("model should not run after forced stop")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "agilex_control.local_loop.call", side_effect=fake_call
        ):
            journal = run_loop(
                CONFIG, "inspect only", Path(tmp) / "run", Path(tmp) / "site.json",
                api_key="test", execute=False, yes=True, decide=decide, cancel=cancel,
            )
        self.assertEqual(journal["status"], "stopped")
        self.assertIn("observe", called)
        self.assertIn("stop", called)


class ContextCompilerTests(unittest.TestCase):
    def test_task_package_requires_instruction(self):
        with self.assertRaises(ValueError):
            load_task_package(instruction="  ")

    def test_input_json_and_goal_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "goal.jpg"
            image.write_bytes(
                b"\xff\xd8\xff\xdb\x00C\x00" + bytes(64) + b"\xff\xd9"
            )
            manifest = Path(tmp) / "task.json"
            manifest.write_text(json.dumps({
                "instruction": "match the goal layout",
                "content": [{"type": "image", "path": str(image), "role": "target_image"}],
            }))
            package = load_task_package(manifest)
            self.assertEqual(package["instruction"], "match the goal layout")
            self.assertEqual(package["content"][0]["type"], "image")

    def test_turn_payload_uses_previous_result_not_answer_dump(self):
        content = compile_user_content(
            "pick the bottle",
            execute=False,
            step=2,
            last_outcome={
                "kind": "observe", "ok": True, "executed": False,
                "output_dir": "/tmp/obs", "summary": {"ok": True, "arms": {}},
            },
            live_images=[],
        )
        payload = json.loads(content[0]["text"])
        self.assertEqual(payload["instruction"], "pick the bottle")
        self.assertEqual(payload["previous_result"]["kind"], "observe")
        self.assertFalse(payload["calibration_required"])


if __name__ == "__main__":
    unittest.main()
