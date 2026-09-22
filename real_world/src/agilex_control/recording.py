"""Separate action-video service; reads camera previews and never opens CAN/USB.

Each phase explicitly starts one clip and finalizes it on exit. The camera owner
retains USB ownership. A frame ledger preserves real camera timestamps and gaps.
"""

import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .cameras import request as camera_request
from .motion import _atomic_json


def request(runtime, message, timeout=12):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(Path(runtime) / "recording.sock"))
        client.sendall((json.dumps(message) + "\n").encode())
        client.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        result = json.loads(b"".join(chunks))
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result


def stage_summary(value):
    """An optional, deliberately written public action summary, not a trace."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"title", "action", "reason"}:
        raise ValueError("Video stage requires title, action and reason")
    for text in value.values():
        if not isinstance(text, str) or not text.strip() or len(text) > 300:
            raise ValueError("Video stage fields require 1-300 characters")
    return {key: text.strip() for key, text in value.items()}


class Clip:
    def __init__(self, output, identity, camera_runtime, fps, max_seconds, capture=None, stage=None):
        if not math.isfinite(fps) or not 1 <= fps <= 30:
            raise ValueError("Recording FPS must be between 1 and 30")
        if not math.isfinite(max_seconds) or not 1 <= max_seconds <= 1800:
            raise ValueError("Recording duration must be between 1 and 1800 seconds")
        stage = stage_summary(stage)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.identity, self.fps, self.max_seconds = identity, fps, max_seconds
        self.capture = capture or (lambda: camera_request(camera_runtime, {"op": "preview"}, timeout=3))
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.record = dict(schema_version=1, identity=identity, output=str(self.output),
                           kind="live RGB video sampled during one execution interval",
                           fps=fps, max_seconds=max_seconds, status="starting",
                           cross_camera_hardware_sync=False, frames=0, error=None,
                           cameras={}, repeated_samples=0, padded_frames=0, max_sample_gap_s=0)
        self.thread = threading.Thread(target=self.run, name="action-video", daemon=True)
        if stage is not None:
            self.record["stage"] = stage

    def start(self):
        self.thread.start()
        if not self.ready.wait(timeout=8) or self.record["error"]:
            self.stop_event.set()
            self.thread.join(timeout=5)
            raise RuntimeError(self.record["error"] or "Recorder did not produce its first frame")
        return self.status()

    def status(self):
        return dict(self.record)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=8)
        if self.thread.is_alive():
            raise RuntimeError("Recorder has not finalized; inspect the existing clip")
        return self.status()

    def run(self):
        import cv2
        import numpy as np

        writers, previous_frames, previous_numbers = {}, {}, {}
        start = time.monotonic()
        previous_sample = None
        ledger = None
        count = 0
        try:
            ledger = (self.output / "frames.jsonl").open("x")
            self.record.update(started_unix=time.time(), status="recording")
            while not self.stop_event.is_set():
                elapsed = time.monotonic() - start
                if elapsed >= self.max_seconds:
                    self.record["status"] = "duration_limit"
                    break
                packet = self.capture()
                sampled = time.monotonic()
                slot = max(count, round((sampled - start) * self.fps)) if writers else 0
                if previous_sample is not None:
                    self.record["max_sample_gap_s"] = max(
                        self.record["max_sample_gap_s"], sampled - previous_sample)
                names = tuple(packet["cameras"])
                if writers and set(names) != set(writers):
                    raise RuntimeError("Camera set changed during a clip")
                frames, stamps, numbers = {}, {}, {}
                for name, item in packet["cameras"].items():
                    if not name or Path(name).name != name or name in {".", ".."}:
                        raise ValueError("Invalid camera key")
                    frame = cv2.imdecode(np.frombuffer(base64.b64decode(
                        item["jpeg_b64"], validate=True), np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError("Unreadable camera preview: " + name)
                    meta = item["metadata"]
                    height, width = frame.shape[:2]
                    if name not in writers:
                        writer = cv2.VideoWriter(str(self.output / (name + ".mp4")),
                            cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
                        if not writer.isOpened():
                            raise OSError("Video encoder failed: " + name)
                        writers[name] = writer
                        self.record["cameras"][name] = dict(serial=meta["serial"],
                            width=width, height=height, file=name + ".mp4")
                    shape = self.record["cameras"][name]
                    if (width, height) != (shape["width"], shape["height"]) or meta["serial"] != shape["serial"]:
                        raise RuntimeError("Camera identity or dimensions changed: " + name)
                    frames[name] = frame
                    stamps[name] = meta["color_timestamp_ms"]
                    numbers[name] = meta["color_frame_number"]
                if not frames:
                    raise RuntimeError("No cameras in the preview")
                # Retain elapsed time if a fetch was late; do not invent motion.
                while count < slot and previous_frames:
                    for name, writer in writers.items():
                        writer.write(previous_frames[name])
                    ledger.write(json.dumps(dict(index=count, repeated_for_gap=True)) + "\n")
                    count += 1
                    self.record["padded_frames"] += 1
                for name, writer in writers.items():
                    writer.write(frames[name])
                ledger.write(json.dumps(dict(index=count, sampled_unix=time.time(),
                    elapsed_s=sampled-start, camera_color_timestamp_ms=stamps,
                    camera_frame_number=numbers, repeated_for_gap=False)) + "\n")
                ledger.flush()
                self.record["repeated_samples"] += int(numbers == previous_numbers)
                count += 1
                self.record["frames"] = count
                previous_frames, previous_numbers, previous_sample = frames, numbers, sampled
                self.ready.set()
                self.stop_event.wait(max(0, start + count / self.fps - time.monotonic()))
            if self.record["status"] == "recording":
                self.record["status"] = "completed"
        except Exception as exc:
            self.record.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        finally:
            for writer in writers.values():
                writer.release()
            if ledger:
                ledger.close()
            self.record.update(finished_unix=time.time(), frames=count,
                               video_seconds=count/self.fps)
            _atomic_json(self.output / "clip.json", self.record)
            self.ready.set()


def serve(camera_runtime, runtime, output_root, fps=15, max_seconds=180):
    runtime, output_root = Path(runtime).resolve(), Path(output_root).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    sockpath, healthpath = runtime / "recording.sock", runtime / "recording-health.json"
    active, running = None, True

    def stop(signum, frame):
        nonlocal running
        running = False

    old = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    with (runtime / "recording.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                sockpath.unlink(missing_ok=True)
                server.bind(str(sockpath))
                os.chmod(sockpath, 0o600)
                server.listen(4)
                server.settimeout(.2)
                while running:
                    health = dict(pid=os.getpid(), updated_unix=time.time(), ready=True,
                                  active=active.status() if active else None,
                                  camera_runtime=str(camera_runtime), output_root=str(output_root))
                    _atomic_json(healthpath, health)
                    try:
                        client, _ = server.accept()
                    except socket.timeout:
                        continue
                    with client:
                        client.settimeout(10)
                        try:
                            cmd = json.loads(client.makefile("rb").readline(8192))
                            op = cmd["op"]
                            if op == "status":
                                reply = health
                            elif op == "start":
                                identity = cmd["identity"]
                                stage = stage_summary(cmd.get("stage"))
                                if not isinstance(identity, str) or not identity:
                                    raise ValueError("A clip identity is required")
                                if active and active.thread.is_alive():
                                    if active.identity != identity:
                                        raise RuntimeError("Another clip is active")
                                    if active.record.get("stage") != stage:
                                        raise ValueError("Active identity has a different stage summary")
                                    reply = active.status()
                                else:
                                    destination = output_root / hashlib.sha256(identity.encode()).hexdigest()[:24]
                                    if destination.exists():
                                        raise FileExistsError("Clip identity already used; inspect it without restarting")
                                    active = Clip(destination, identity, camera_runtime, fps, max_seconds, stage=stage)
                                    reply = active.start()
                            elif op == "stop":
                                if not active or active.identity != cmd["identity"]:
                                    raise ValueError("Stop requires the current clip identity")
                                reply = active.stop()
                            elif op == "shutdown":
                                if active:
                                    active.stop()
                                running, reply = False, {"stopping": True}
                            else:
                                raise ValueError("Unknown recording operation")
                        except Exception as exc:
                            reply = {"error": type(exc).__name__ + ": " + str(exc)}
                        try:
                            client.sendall(json.dumps(reply).encode())
                        except (OSError, socket.timeout):
                            pass
        finally:
            if active:
                active.stop()
            sockpath.unlink(missing_ok=True)
            _atomic_json(healthpath, dict(ready=False, stopped=True, updated_unix=time.time()))
            for sig, handler in old.items():
                signal.signal(sig, handler)


@contextmanager
def phase_recording(config, root, record, save):
    """Required recording fails before actuation; stop errors preserve action truth."""
    setting = config.get("action_recording")
    if not setting:
        yield
        return
    runtime = setting["runtime_dir"]
    identity = str(Path(root).resolve())
    try:
        stage = stage_summary(record.get("request", {}).get("context", {}).get("video"))
        message = {"op": "start", "identity": identity}
        if stage is not None:
            message["stage"] = stage
        started = request(runtime, message)
        if stage is not None and started.get("stage") != stage:
            request(runtime, {"op": "stop", "identity": identity})
            raise RuntimeError("Recorder did not retain the requested stage summary")
    except Exception as exc:
        record["recording"] = dict(status="start_failed", error=str(exc))
        record["action_status"] = "failed"
        save()
        raise
    record["recording"] = dict(status="recording", start=started)
    save()
    try:
        yield
    finally:
        try:
            stopped = request(runtime, {"op": "stop", "identity": identity})
            record["recording"] = dict(status=stopped["status"], result=stopped)
        except Exception as exc:
            record["recording"] = dict(status="finalization_uncertain", start=started, error=str(exc))
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["serve", "status", "start", "stop", "shutdown"])
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--camera-runtime", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--identity")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--max-seconds", type=float, default=180)
    args = parser.parse_args()
    if args.operation == "serve":
        if args.camera_runtime is None or args.output_root is None:
            parser.error("serve requires camera-runtime and output-root")
        serve(args.camera_runtime, args.runtime, args.output_root, args.fps, args.max_seconds)
    else:
        message = {"op": args.operation}
        if args.identity:
            message["identity"] = args.identity
        print(json.dumps(request(args.runtime, message), indent=2))


if __name__ == "__main__":
    main()
