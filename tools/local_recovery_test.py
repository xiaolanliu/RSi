"""Loopback-only structured recovery fixture for native integration tests."""
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import threading
from unittest.mock import patch

from rsi_loop.demonstration import Demonstration
from rsi_loop.kinematics import DualMotion
from rsi_loop.recovery import ContextRecovery, ResponsesTransport


class VideoFixture(Demonstration):
    def content(self):
        content = super().content()
        content[0]["text"] = "PROTOCOL TEST VIDEO ONLY; this is NOT a successful demonstration."
        return content


class LocalStructuredRecovery:
    duration = 10

    def __init__(self, output, paths, video):
        self.output, self.paths, self.video = Path(output), paths, Path(video)
        self.received = []

    def __enter__(self):
        received, output, duration = self.received, self.output, self.duration
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/v1/responses":
                    self.send_error(404)
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append(body)
                (output/"local_http_request.json").write_text(json.dumps(body, indent=2))
                texts = [p["text"] for p in body["input"][0]["content"] if p["type"] == "input_text"]
                context = json.loads(next(t.removeprefix("Current robot context: ") for t in texts
                                          if t.startswith("Current robot context: ")))
                state = context["state_left7_right7"]
                result = dict(status="recover", diagnosis="LOCAL PROTOCOL TEST: 1 mm upward, not a model diagnosis",
                    duration_steps=duration, **{name: dict(translation_m=[0, 0, .001],
                        rotation_vector_rad=[0, 0, 0], gripper_opening=state[i*7+6])
                        for i, name in enumerate(("left", "right"))})
                data = json.dumps(dict(status="completed", usage=dict(external_api_calls=0),
                    output=[dict(type="message", content=[dict(type="output_text", text=json.dumps(result))])])).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *args):
                pass
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        import os
        self.key_patch = patch.dict(os.environ, {"RSI_LOCAL_PROTOCOL_TEST_KEY": "local-test-only"})
        self.key_patch.start()
        return self

    def factory(self, steps):
        sys.path.insert(0, self.paths["gpt_policy"]+"/src")
        fixture = VideoFixture(self.video, task="protocol fixture", cache=self.output/"video_fixture",
                               provenance=dict(test_only=True, success_demonstration=False))
        def motion(obs, plan):
            meta = json.loads((self.output/"native_metadata.json").read_text())
            return DualMotion(meta["robot_descriptions"], meta["control_dt"])(obs, plan)
        transport = ResponsesTransport(model="LOCAL_STUB_NOT_GPT", enabled=True,
            base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
            key_env="RSI_LOCAL_PROTOCOL_TEST_KEY", timeout=10)
        recovery = ContextRecovery(transport, motion, demo=fixture, output=self.output/"recovery")
        class FixtureRecovery:
            def plan(self, obs, risk, history):
                return replace(recovery.plan(obs, risk, history), source="mock")
        return FixtureRecovery()

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.key_patch.stop()
