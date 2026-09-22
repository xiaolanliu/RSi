"""Reuse a live ROS RealSense graph when USB pipelines cannot open."""

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

DEFAULT_NAMESPACES = {
    "front": "/camera_f",
    "left_wrist": "/camera_l",
    "right_wrist": "/camera_r",
}

WORKER = Path(__file__).with_name("ros_capture_worker.py")


def should_fallback(exc):
    text = str(exc)
    needles = (
        "Device or resource busy",
        "VIDIOC_S_FMT",
        "errno=16",
        "Camera service is not running",
        "not accepting connections",
        "no fresh RGBD frame",
    )
    return any(item in text for item in needles)


def _namespaces(config):
    configured = config.get("ros_cameras") or {}
    mapping = {}
    for name, serial in config["cameras"].items():
        namespace = configured.get(name) or DEFAULT_NAMESPACES.get(name)
        if not namespace:
            raise ValueError("No ROS namespace for camera " + name)
        mapping[name] = {"namespace": str(namespace), "serial": str(serial)}
    return mapping


def capture(config, output, timeout=8.0):
    """Write the same per-camera files as cameras serve capture, via ROS topics."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = {"cameras": _namespaces(config)}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(spec, handle)
        spec_path = handle.name
    worker = str(WORKER.resolve())
    inner = shlex.join(
        ["/usr/bin/python3", worker, "--output", str(output), "--spec", spec_path,
         "--timeout", str(timeout)]
    )
    command = "source /opt/ros/noetic/setup.bash && " + inner
    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout * max(3, len(spec["cameras"])) + 10,
            env=os.environ.copy(),
        )
    finally:
        Path(spec_path).unlink(missing_ok=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() or "no ROS worker output"
        raise RuntimeError("ROS camera capture failed: " + detail)
    reply_path = output / "capture_reply.json"
    if not reply_path.is_file():
        detail = (result.stderr or result.stdout or "").strip() or str(reply_path)
        raise RuntimeError("ROS camera capture wrote no reply: " + detail)
    payload = json.loads(reply_path.read_text())
    payload["source"] = "ros"
    payload["usb_owner"] = "realsense2_camera"
    return payload
