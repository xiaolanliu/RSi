"""Capture one RGBD snapshot from an already-running ROS RealSense graph.

USB pipelines must not be opened while realsense2_camera nodelets own the
devices. This worker uses system ROS Python and is launched as a subprocess.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image


def _wait(topic, cls, timeout):
    return rospy.wait_for_message(topic, cls, timeout=timeout)


def _color_meta(info, serial, name, stamp, extra=None):
    k = info.K
    meta = dict(
        name=name,
        serial=serial,
        received_unix=time.time(),
        color_timestamp_ms=stamp.to_sec() * 1000.0,
        depth_timestamp_ms=stamp.to_sec() * 1000.0,
        timestamp_domain="ros_header",
        width=info.width,
        height=info.height,
        fx=k[0],
        fy=k[4],
        cx=k[2],
        cy=k[5],
        coeffs=list(info.D),
        model=info.distortion_model,
        depth_scale_m=0.001,
        aligned_to_color=False,
        source="ros",
        frame_id=info.header.frame_id,
    )
    if extra:
        meta.update(extra)
    return meta


def capture_one(name, namespace, serial, output, timeout, bridge):
    ns = namespace.rstrip("/")
    color_topic = ns + "/color/image_raw"
    info_topic = ns + "/color/camera_info"
    depth_topic = ns + "/depth/image_rect_raw"
    color = _wait(color_topic, Image, timeout)
    info = _wait(info_topic, CameraInfo, timeout)
    try:
        depth = _wait(depth_topic, Image, timeout)
    except rospy.ROSException:
        depth = None
    rgb = bridge.imgmsg_to_cv2(color, desired_encoding="rgb8")
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(folder / "rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError("Image write failed: " + name)
    extra = {
        "ros_namespace": ns,
        "color_topic": color_topic,
        "color_frame_number": int(color.header.seq),
        "depth_frame_number": int(depth.header.seq) if depth is not None else None,
    }
    if depth is not None:
        dep = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        np.save(folder / "depth.npy", dep)
        extra["depth_topic"] = depth_topic
        extra["depth_encoding"] = depth.encoding
    meta = _color_meta(info, serial, name, color.header.stamp, extra)
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rospy.init_node("agilex_ros_capture", anonymous=True, disable_signals=True)
    bridge = CvBridge()
    reply = {"captured_unix": time.time(), "cameras": {}, "source": "ros"}
    for name, item in spec["cameras"].items():
        reply["cameras"][name] = capture_one(
            name, item["namespace"], item["serial"], output, args.timeout, bridge
        )
    (output / "observation.json").write_text(json.dumps(reply, indent=2) + "\n")
    (output / "capture_reply.json").write_text(json.dumps(reply, indent=2) + "\n")


if __name__ == "__main__":
    main()
