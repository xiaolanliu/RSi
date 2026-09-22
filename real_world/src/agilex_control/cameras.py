"""One owner per RGBD camera; a local Unix socket requests snapshots.

Camera acquisition runs independently of CAN target timing. Images and depth
are aligned within each camera; cameras are not hardware synchronized.
"""

import json
import base64
import os
import signal
import socket
import threading
import time
from pathlib import Path


class RgbdCamera:
    def __init__(self, name, serial, resolution):
        self.name, self.serial, self.resolution = name, serial, resolution
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest = None
        self.error = None
        self.thread = threading.Thread(target=self.acquire, name=name, daemon=True)
        self.thread.start()

    def acquire(self):
        import pyrealsense2 as rs
        import numpy as np

        pipeline = rs.pipeline()
        started = False
        try:
            width, height, fps = self.resolution
            cfg = rs.config()
            cfg.enable_device(self.serial)
            cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            profile = pipeline.start(cfg)
            started = True
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            align = rs.align(rs.stream.color)
            count = 0
            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(2000)
                frames = align.process(frames)
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    continue
                count += 1
                if count < 30:
                    continue
                rgb = np.asanyarray(color.get_data()).copy()
                dep = np.asanyarray(depth.get_data()).copy()
                i = color.profile.as_video_stream_profile().get_intrinsics()
                meta = dict(
                    name=self.name,
                    serial=self.serial,
                    received_unix=time.time(),
                    color_frame_number=color.get_frame_number(),
                    depth_frame_number=depth.get_frame_number(),
                    color_timestamp_ms=color.get_timestamp(),
                    depth_timestamp_ms=depth.get_timestamp(),
                    timestamp_domain=str(color.get_frame_timestamp_domain()),
                    width=i.width,
                    height=i.height,
                    fx=i.fx,
                    fy=i.fy,
                    cx=i.ppx,
                    cy=i.ppy,
                    coeffs=i.coeffs,
                    model=str(i.model),
                    depth_scale_m=scale,
                )
                with self.lock:
                    self.latest = (rgb, dep, meta)
        except BaseException as exc:
            text = type(exc).__name__ + ": " + str(exc)
            if "busy" in text.lower() or "errno=16" in text or "VIDIOC_S_FMT" in text:
                text += (
                    " USB is already owned (this host: ROS realsense2_camera "
                    "camera_f/l/r). Stop that launch before cameras serve, or use observe "
                    "which falls back to ROS topics."
                )
            self.error = text
        finally:
            if started:
                pipeline.stop()

    def snapshot(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.name + ": " + self.error)
            if self.latest is None or time.time() - self.latest[2]["received_unix"] > 1:
                raise RuntimeError(self.name + ": no fresh RGBD frame")
            return self.latest

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)


def request(runtime, message, timeout=10):
    sockpath = Path(runtime) / "cameras.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect(str(sockpath))
        except FileNotFoundError:
            raise FileNotFoundError(
                "Camera service is not running; missing %s. Start it with: "
                "PYTHONPATH=src python -m agilex_control --config <site.json> cameras serve"
                % sockpath
            ) from None
        except ConnectionRefusedError:
            raise ConnectionRefusedError(
                "Camera socket exists but is not accepting connections: %s" % sockpath
            ) from None
        s.sendall((json.dumps(message) + "\n").encode())
        s.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            b = s.recv(65536)
            if not b:
                break
            chunks.append(b)
        response = json.loads(b"".join(chunks))
        if "error" in response:
            raise RuntimeError(response["error"])
        return response


def preview(cameras):
    """Read current RGB frames for a local recorder without another USB owner."""
    import cv2

    frames = {}
    for name, camera in cameras.items():
        rgb, _, meta = camera.snapshot()
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise OSError("Preview JPEG encode failed: " + name)
        frames[name] = {"metadata": meta, "jpeg_b64": base64.b64encode(encoded).decode("ascii")}
    return {"captured_unix": time.time(), "cameras": frames,
            "cross_camera_hardware_sync": False}


def serve(config):
    import cv2
    import fcntl
    import numpy as np

    runtime = Path(config["runtime_dir"])
    runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
    sockpath = runtime / "cameras.sock"
    healthpath = runtime / "camera_health.json"
    cameras = {}
    running = True
    lock = (runtime / "cameras.lock").open("a+")
    owns_runtime = False

    def stop(signum, frame):
        nonlocal running
        running = False

    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}

    def health():
        result = dict(pid=os.getpid(), updated_unix=time.time(), ready=True, cameras={})
        for name, camera in cameras.items():
            try:
                _, _, m = camera.snapshot()
                result["cameras"][name] = m
            except RuntimeError as exc:
                result["ready"] = False
                result["cameras"][name] = {"error": str(exc)}
        return result

    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owns_runtime = True
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            sockpath.unlink(missing_ok=True)
            server.bind(str(sockpath))
            os.chmod(sockpath, 0o600)
            server.listen(4)
            server.settimeout(0.2)
            for name, serial in config["cameras"].items():
                resolution = config.get("camera_resolutions", {}).get(
                    name, config["camera_resolution"]
                )
                cameras[name] = RgbdCamera(name, serial, resolution)
            while running:
                status = health()
                temp = healthpath.with_suffix(".tmp")
                temp.write_text(json.dumps(status))
                temp.replace(healthpath)
                try:
                    client, _ = server.accept()
                except socket.timeout:
                    continue
                with client:
                    client.settimeout(2)
                    try:
                        message = client.makefile("rb").readline(4096)
                        cmd = json.loads(message)
                        if cmd["op"] == "status":
                            reply = health()
                        elif cmd["op"] == "preview":
                            reply = preview(cameras)
                        elif cmd["op"] == "stop":
                            running = False
                            reply = {"stopping": True}
                        elif cmd["op"] == "capture":
                            out = Path(cmd["output"])
                            out.mkdir(parents=True, exist_ok=True)
                            reply = dict(captured_unix=time.time(), cameras={})
                            for name, camera in cameras.items():
                                rgb, dep, meta = camera.snapshot()
                                folder = out / name
                                folder.mkdir(exist_ok=True)
                                if not cv2.imwrite(
                                    str(folder / "rgb.jpg"),
                                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                ):
                                    raise OSError("Image write failed")
                                np.save(folder / "depth.npy", dep)
                                (folder / "metadata.json").write_text(
                                    json.dumps(meta, indent=2) + "\n"
                                )
                                reply["cameras"][name] = meta
                            (out / "observation.json").write_text(
                                json.dumps(reply, indent=2) + "\n"
                            )
                        else:
                            raise ValueError("Unknown camera operation")
                    except Exception as exc:
                        reply = {"error": type(exc).__name__ + ": " + str(exc)}
                    try:
                        client.sendall(json.dumps(reply).encode())
                    except (OSError, socket.timeout):
                        pass  # A departed snapshot client does not own the cameras.
    finally:
        for camera in cameras.values():
            camera.close()
        if owns_runtime:
            sockpath.unlink(missing_ok=True)
            healthpath.write_text(
                json.dumps(dict(updated_unix=time.time(), ready=False, stopped=True))
            )
        lock.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
