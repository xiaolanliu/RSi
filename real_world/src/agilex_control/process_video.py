"""Record the front RGB stream for one local-loop run, with language+command overlay."""

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from .motion import _atomic_json

WORKER = Path(__file__).with_name("ros_front_record_worker.py")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def command_payload(decision):
    name = (decision or {}).get("decision")
    if name == "dry_run":
        return decision.get("dry_run")
    if name == "phase":
        return decision.get("phase")
    if name == "geometry":
        return decision.get("geometry")
    if name == "get_state":
        return {"seconds": decision.get("get_state_seconds", 1)}
    return None


def caption_from_decision(decision, status="thinking"):
    decision = decision or {}
    command = command_payload(decision)
    return {
        "status": status,
        "decision": decision.get("decision") or "idle",
        "reason": (decision.get("reason") or "").strip(),
        "command": json.dumps(command, ensure_ascii=False, indent=2) if command else "",
    }


class ProcessVideo:
    def __init__(self, output, namespace="/camera_f", fps=15, topic=None):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.topic = topic or (namespace.rstrip("/") + "/color/image_raw")
        self.fps = fps
        self.overlay_path = self.output / "overlay.json"
        self.stop_path = self.output / "STOP"
        self.preview_path = self.output / "front.jpg"
        self.video_path = self.output / "front-process.mp4"
        self.proc = None
        self.set_overlay(caption_from_decision({"decision": "observe", "reason": "starting"}))

    def set_overlay(self, caption):
        payload = dict(caption)
        payload["updated_unix"] = time.time()
        _atomic_json(self.overlay_path, payload)

    def start(self):
        inner = shlex.join([
            "/usr/bin/python3", str(WORKER.resolve()),
            "--output", str(self.video_path),
            "--overlay", str(self.overlay_path),
            "--preview", str(self.preview_path),
            "--stop", str(self.stop_path),
            "--topic", self.topic,
            "--font", FONT,
            "--fps", str(self.fps),
        ])
        command = "source /opt/ros/noetic/setup.bash && " + inner
        self.stop_path.unlink(missing_ok=True)
        self.proc = subprocess.Popen(
            ["bash", "-lc", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        deadline = time.time() + 8
        while time.time() < deadline:
            if self.preview_path.is_file() or (self.proc.poll() is not None):
                break
            time.sleep(0.1)
        if self.proc.poll() is not None:
            err = (self.proc.stderr.read() if self.proc.stderr else "") or "recorder exited"
            raise RuntimeError("Process video recorder failed: " + err.strip())
        return {"video": str(self.video_path), "preview": str(self.preview_path)}

    def stop(self):
        self.stop_path.write_text("stop\n")
        if self.proc is None:
            return {"video": None}
        try:
            self.proc.wait(timeout=12)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        stderr = ""
        if self.proc.stderr:
            stderr = self.proc.stderr.read().strip()
        record = {
            "video": str(self.video_path) if self.video_path.is_file() else None,
            "preview": str(self.preview_path) if self.preview_path.is_file() else None,
            "returncode": self.proc.returncode,
            "stderr": stderr[-2000:],
        }
        _atomic_json(self.output / "record.json", record)
        return record
