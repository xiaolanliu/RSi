#!/usr/bin/env python3
"""Read one RobotIOServer observation from camera port 5555; no action connection."""
import argparse
import datetime
import io
import json
import pickle
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np


class ObservationUnpickler(pickle.Unpickler):
    """Only NumPy reconstruction globals are allowed in the observation envelope."""

    def find_class(self, module, name):
        core = getattr(np, "_core", None)
        if core is None or not all(hasattr(core, key) for key in ("multiarray", "numeric")):
            core = np.core
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(core.multiarray, name)
        if module in ("numpy.core.numeric", "numpy._core.numeric") and name == "_frombuffer":
            return core.numeric._frombuffer
        raise pickle.UnpicklingError("Unsupported observation global: " + module + "." + name)


def decode(payload):
    obj = ObservationUnpickler(io.BytesIO(payload)).load()
    if not isinstance(obj, dict) or not isinstance(obj.get("observation"), dict):
        raise ValueError("Expected RobotIOServer {seq, t_server, observation} envelope")
    return obj


def validate_endpoint(endpoint):
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "tcp" or parsed.port != 5555 or not parsed.hostname
            or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
        raise ValueError("This camera reader accepts only tcp://HOST:5555")
    return endpoint


def receive(endpoint, timeout):
    import zmq
    validate_endpoint(endpoint)
    if not 0.5 <= timeout <= 10:
        raise ValueError("Timeout must be between 0.5 and 10 seconds")
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    try:
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        subscriber.setsockopt(zmq.CONFLATE, 1)
        subscriber.setsockopt(zmq.RCVHWM, 1)
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.MAXMSGSIZE, 32 * 1024 * 1024)
        subscriber.connect(endpoint)
        if not subscriber.poll(int(timeout * 1000), zmq.POLLIN):
            raise TimeoutError("No observation received on camera port 5555; server was not restarted")
        payload = subscriber.recv()
        return decode(payload), len(payload)
    finally:
        subscriber.close(linger=0)
        context.term()


def summarize(envelope, size, endpoint, output_dir=None):
    report = {"captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "endpoint": endpoint, "action_port_connected": False,
              "server_seq": envelope.get("seq"), "t_server": envelope.get("t_server"),
              "payload_bytes": size, "images": {}, "other_field_count": 0}
    timestamp = envelope.get("t_server")
    if isinstance(timestamp, (int, float)):
        report["envelope_age_seconds_at_read"] = round(time.time() - timestamp, 4)
    for key, value in envelope["observation"].items():
        if not isinstance(value, np.ndarray) or value.ndim not in (2, 3):
            report["other_field_count"] += 1
            continue
        details = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if value.size and np.issubdtype(value.dtype, np.number):
            details["min"] = float(np.nanmin(value))
            details["max"] = float(np.nanmax(value))
        if output_dir and value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] == 3:
            import cv2
            # This server's RealSense cameras are configured for RGB output.
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(key)).strip(".") or "camera"
            target = output_dir / (name + ".jpg")
            ok = cv2.imwrite(str(target), cv2.cvtColor(value, cv2.COLOR_RGB2BGR),
                             [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise OSError("Could not save " + str(target))
            details["saved_image"] = str(target)
            details["interpreted_color_order"] = "RGB; verify server configuration before reuse"
        report["images"][str(key)] = details
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    envelope, size = receive(args.endpoint, args.timeout)
    report = summarize(envelope, size, args.endpoint, args.output_dir)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_dir:
        (args.output_dir / "camera_observation.json").write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
