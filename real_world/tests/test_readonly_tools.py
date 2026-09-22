"""Offline regression checks. No hardware, network, SDK initialization or motion."""

import importlib.util
import pickle
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/wcx-agilex-control/scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


camera = load("read_camera_5555")
state_reader = load("read_state")
planner = load("plan_lift")


class CameraTests(unittest.TestCase):
    def test_numpy_protocols_and_rejected_global(self):
        expected = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        for protocol in (4, 5):
            envelope = camera.decode(
                pickle.dumps({"observation": {"front": expected}}, protocol)
            )
            np.testing.assert_array_equal(envelope["observation"]["front"], expected)
        # Serializing a global reference does not execute it. The decoder must reject it.
        with self.assertRaises(pickle.UnpicklingError):
            camera.decode(pickle.dumps(eval))

    def test_action_and_metadata_ports_are_rejected(self):
        for endpoint in (
            "tcp://127.0.0.1:5556",
            "tcp://127.0.0.1:5557",
            "tcp://user@127.0.0.1:5555",
            "ipc:///tmp/camera",
        ):
            with self.assertRaises(ValueError):
                camera.validate_endpoint(endpoint)

    def fake_zmq(self, has_message):
        events = []

        class Subscriber:
            def setsockopt(self, *args):
                pass

            def connect(self, endpoint):
                events.append(("connect", endpoint))

            def poll(self, timeout, flags):
                return has_message

            def recv(self):
                events.append(("recv",))
                return pickle.dumps({"seq": 1, "observation": {}})

            def close(self, linger):
                events.append(("close", linger))

        class Context:
            def socket(self, kind):
                events.append(("socket", kind))
                return Subscriber()

            def term(self):
                events.append(("term",))

        module = types.SimpleNamespace(
            Context=Context,
            SUB=2,
            SUBSCRIBE=6,
            CONFLATE=54,
            RCVHWM=24,
            LINGER=17,
            MAXMSGSIZE=22,
            POLLIN=1,
        )
        return module, events

    def test_one_subscriber_one_receive_and_cleanup(self):
        zmq, events = self.fake_zmq(True)
        with patch.dict(sys.modules, {"zmq": zmq}):
            camera.receive("tcp://127.0.0.1:5555", 0.5)
        self.assertEqual(
            events,
            [
                ("socket", 2),
                ("connect", "tcp://127.0.0.1:5555"),
                ("recv",),
                ("close", 0),
                ("term",),
            ],
        )

    def test_timeout_and_wrong_port_do_not_fall_back_to_actions(self):
        zmq, events = self.fake_zmq(False)
        with patch.dict(sys.modules, {"zmq": zmq}):
            with self.assertRaises(TimeoutError):
                camera.receive("tcp://127.0.0.1:5555", 0.5)
            count = len(events)
            with self.assertRaises(ValueError):
                camera.receive("tcp://127.0.0.1:5556", 0.5)
            self.assertEqual(len(events), count)
        self.assertNotIn(("recv",), events)
        self.assertEqual(events[-2:], [("close", 0), ("term",)])


class StateAndPlanTests(unittest.TestCase):
    def frames(self):
        frames = {cid: bytes(8) for cid in state_reader.REQUIRED}
        frames[0x2A1] = bytes([1, 0, 1, 0, 0, 0, 0, 0])
        for cid in range(0x261, 0x267):
            frames[cid] = bytes([0, 240, 0, 0, 0, 64, 0, 0])
        return frames

    def test_fault_byte_order_and_stale_feedback(self):
        frames = self.frames()
        stamps = {cid: 100 for cid in frames}
        good = state_reader.decode_state(frames, stamps, 100.1)
        self.assertTrue(good["feedback_checks_pass"])
        self.assertFalse(good["motion_verified"])
        self.assertFalse(
            state_reader.decode_state(frames, stamps, 100.3)["feedback_checks_pass"]
        )
        frames[0x2A1] = bytes([1, 5, 1, 0, 0, 0, 0, 0x13])
        fault = state_reader.decode_state(frames, stamps, 100.1)
        self.assertEqual(fault["joint_communication_errors"], [1, 2, 5])
        self.assertEqual(fault["joint_limit_errors"], [])
        self.assertFalse(fault["feedback_checks_pass"])

    def test_fault_and_out_of_bounds_refused_before_fk_loading(self):
        state = {
            "complete": True,
            "stale": [],
            "arm_status": 0,
            "error_code": 0,
            "joint_deg": [0, 1, -1, 0, 0, 0],
            "pose_mm_deg": [0] * 6,
        }
        cases = [
            dict(state, arm_status=5, error_code=0x13),
            dict(state, stale=["0x2a1"]),
            dict(state, complete=False),
            dict(state, joint_deg=[0, -1, 1, 0, 0, 0]),
        ]
        for case in cases:
            with self.assertRaises(ValueError):
                planner.plan(case, "can_right", 20, "/nonexistent/fk.py")
        for lift in (0, -1, 21, float("nan")):
            with self.assertRaises(ValueError):
                planner.plan(state, "can_right", lift, "/nonexistent/fk.py")


if __name__ == "__main__":
    unittest.main()
