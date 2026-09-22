"""Hardware-free tests for live dashboard captions and event hub."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agilex_control.dashboard import Hub
from agilex_control.local_loop import run_loop
from agilex_control.process_video import caption_from_decision, command_payload

from test_local_loop import CONFIG, MOVE, observe_result


class CaptionTests(unittest.TestCase):
    def test_dry_run_caption_includes_purpose_and_command(self):
        caption = caption_from_decision({
            "decision": "dry_run",
            "reason": "先规划一次抬升",
            "dry_run": MOVE,
        }, status="command")
        self.assertEqual(caption["status"], "command")
        self.assertEqual(caption["decision"], "dry_run")
        self.assertIn("抬升", caption["reason"])
        self.assertIn("move_right", caption["command"])
        self.assertEqual(command_payload({"decision": "done", "reason": "ok"}), None)


class HubTests(unittest.TestCase):
    def test_step_events_upsert_and_decision_shows_command(self):
        with tempfile.TemporaryDirectory() as tmp, patch("agilex_control.dashboard.PROJECT", Path(tmp)):
            hub = Hub(CONFIG, Path(tmp) / "site.json")
            hub.publish({
                "kind": "status",
                "status": "running",
                "journal": {"status": "running", "task": "inspect", "steps": []},
            })
            hub.publish({
                "kind": "decision",
                "step": 1,
                "decision": {"decision": "dry_run", "reason": "plan", "dry_run": MOVE},
                "caption": caption_from_decision(
                    {"decision": "dry_run", "reason": "plan", "dry_run": MOVE}, status="command"),
            })
            first = hub.snapshot()
            self.assertEqual(first["status"], "command")
            self.assertEqual(len(first["journal"]["steps"]), 1)
            self.assertIsNone(first["journal"]["steps"][0]["outcome"])
            self.assertIn("move_right", first["caption"]["command"])
            hub.publish({
                "kind": "step",
                "step": {
                    "index": 1,
                    "decision": {"decision": "dry_run", "reason": "plan", "dry_run": MOVE},
                    "outcome": {"kind": "dry_run", "ok": True},
                },
            })
            second = hub.snapshot()
            self.assertEqual(len(second["journal"]["steps"]), 1)
            self.assertTrue(second["journal"]["steps"][0]["outcome"]["ok"])

    def test_start_requires_task(self):
        with tempfile.TemporaryDirectory() as tmp, patch("agilex_control.dashboard.PROJECT", Path(tmp)):
            hub = Hub(CONFIG, Path(tmp) / "site.json")
            with self.assertRaises(ValueError):
                hub.start({"task": "  "})

    def test_halt_requests_stop(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "agilex_control.dashboard.PROJECT", Path(tmp)
        ), patch(
            "agilex_control.primitives.call",
            return_value={"ok": True, "error": None, "data": {"stop_signal_sent": False}},
        ):
            hub = Hub(CONFIG, Path(tmp) / "site.json")
            result = hub.halt()
        self.assertTrue(hub.cancel.is_set())
        self.assertTrue(result["ok"])


class LoopEventTests(unittest.TestCase):
    def test_run_loop_emits_live_events_and_optional_video(self):
        events = []
        decisions = [({"decision": "done", "reason": "scene is already fine"}, {"id": "fake"})]

        def decide(*args, **kwargs):
            return decisions.pop(0)

        class FakeRecorder:
            def __init__(self, output):
                self.output = Path(output)
                self.output.mkdir(parents=True, exist_ok=True)
                self.video_path = self.output / "front-process.mp4"
                self.preview_path = self.output / "front.jpg"
                self.overlays = []
                self.stopped = False

            def set_overlay(self, caption):
                self.overlays.append(caption)

            def start(self):
                self.preview_path.write_bytes(b"jpeg")
                return {"video": str(self.video_path), "preview": str(self.preview_path)}

            def stop(self):
                self.stopped = True
                self.video_path.write_bytes(b"mp4")
                return {"video": str(self.video_path), "preview": str(self.preview_path)}

        with tempfile.TemporaryDirectory() as tmp, patch(
            "agilex_control.local_loop.call", return_value=observe_result()
        ), patch("agilex_control.local_loop.ProcessVideo", FakeRecorder):
            journal = run_loop(
                CONFIG, "inspect only", Path(tmp) / "run", Path(tmp) / "site.json",
                api_key="test", execute=False, yes=True, decide=decide,
                on_event=events.append, save_video=True,
            )
            self.assertTrue((Path(journal["output"]) / "process-video" / "front-process.mp4").is_file())
            self.assertTrue(journal["save_video"])
        kinds = [event["kind"] for event in events]
        self.assertEqual(journal["status"], "completed")
        self.assertIn("status", kinds)
        self.assertIn("thinking", kinds)
        self.assertIn("decision", kinds)
        self.assertIn("video", kinds)
        decision = next(event for event in events if event["kind"] == "decision")
        self.assertEqual(decision["caption"]["decision"], "done")
        self.assertEqual(decision["caption"]["reason"], "scene is already fine")


if __name__ == "__main__":
    unittest.main()
