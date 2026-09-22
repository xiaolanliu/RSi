"""Hardware-free tests of frame routing, finite trajectories, and tool results."""

import fcntl
import math
import signal
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agilex_control import motion
from agilex_control.primitives import call
from agilex_control.protocol import (
    joint_frames,
    gripper_frame,
    decode_state,
    REQUIRED,
    MODE_FRAME,
)

LIMITS = [[-150, 150], [0, 180], [-170, 0], [-100, 100], [-70, 70], [-120, 120]]
START = [0, 10, -20, 0, 30, 0]


class Clock:
    def __init__(self):
        self.now = 0

    def monotonic(self):
        return self.now


class FakeBus:
    instances = []
    clock = None
    fail_id = None
    fail_arm = None
    unhealthy_arm = None
    stall_once = False
    stop_after_first_tick = False

    def __init__(self, *args, **kwargs):
        self.events = []
        self.sent = []
        self.joints = {"left": START.copy(), "right": START.copy()}
        self.widths = {"left": 30, "right": 30}
        self.arm = None
        self.writers = {}
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def pump(self, seconds):
        self.clock.now += max(0.002, seconds)
        if self.stall_once and self.sent:
            self.clock.now += 0.5
            self.stall_once = False
        if self.stop_after_first_tick and self.sent:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            self.stop_after_first_tick = False

    def state(self):
        return {
            arm: dict(
                complete=True,
                stale=[],
                ctrl_mode=1,
                arm_status=1 if arm == self.unhealthy_arm else 0,
                error_code=0,
                status_raw=[1, 0, 1, 0, 0, 0, 0, 0],
                driver_status_bytes=[64] * 6,
                joint_deg=q.copy(),
                pose_mm_deg=[50, 0, 250, 180, 45, 180],
                gripper_mm=self.widths[arm],
                gripper_fault_bits=0,
            )
            for arm, q in self.joints.items()
        }

    def open_writer(self, arm, interface):
        self.arm = arm
        self.interface = interface
        self.writers[arm] = interface

    def send(self, cid, data, arm=None):
        if arm is not None:
            self.arm = arm
        if cid == self.fail_id and (self.fail_arm is None or self.arm == self.fail_arm):
            raise OSError("injected CAN send failure")
        self.sent.append((self.arm, cid, data))
        if cid == 0x159:
            # A 25 mm object prevents full closure; this is expected contact.
            self.widths[self.arm] = max(25, struct.unpack(">i", data[:4])[0] / 1000)
        if 0x155 <= cid <= 0x157:
            i = (cid - 0x155) * 2
            self.joints[self.arm][i : i + 2] = [
                v / 1000 for v in struct.unpack(">ii", data)
            ]


class PrimitivesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = dict(
            arms={"left": "can_left", "right": "can_right"},
            joint_limits_deg=LIMITS,
            runtime_dir=str(self.root),
        )
        self.clock = Clock()
        FakeBus.clock = self.clock
        FakeBus.instances = []
        FakeBus.fail_id = None
        FakeBus.fail_arm = None
        FakeBus.unhealthy_arm = None
        FakeBus.stall_once = False
        FakeBus.stop_after_first_tick = False

    def tearDown(self):
        self.temp.cleanup()

    def run_move(self, arm="right", settle=0):
        target = [1, 11, -21, 2, 31, -1]
        with (
            patch.object(motion, "CanBus", FakeBus),
            patch.object(motion.time, "monotonic", self.clock.monotonic),
        ):
            return motion.move(
                self.config, arm, target, 0.04, settle, self.root / "run.json"
            ), target

    def test_both_arms_have_same_frame_contract_and_zero_settle_reaches_endpoint(self):
        for arm in ["left", "right"]:
            result, target = self.run_move(arm)
            bus = FakeBus.instances[-1]
            self.assertEqual(result["status"], "command_stream_completed")
            self.assertEqual(bus.interface, self.config["arms"][arm])
            self.assertTrue(all(a == arm for a, _, _ in bus.sent))
            self.assertEqual(
                [x[1:] for x in bus.sent[-3:]], joint_frames(target, LIMITS)
            )
            self.assertEqual(bus.joints[arm], target)
            self.assertEqual(bus.joints["left" if arm == "right" else "right"], START)
            self.assertEqual(bus.sent[0][1:], (0x151, MODE_FRAME))

    def test_partial_send_failure_stops_further_commands(self):
        FakeBus.fail_id = 0x156
        result, _ = self.run_move()
        self.assertEqual(result["status"], "aborted")
        self.assertEqual(
            [cid for _, cid, _ in FakeBus.instances[-1].sent], [0x151, 0x155]
        )

    def test_concurrent_rejected_move_does_not_overwrite_owner_telemetry(self):
        output = self.root / "run.json"
        output.write_text("owner telemetry")
        with (self.root / "motion.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result, _ = self.run_move()
            self.assertEqual(result["status"], "aborted")
            self.assertEqual(output.read_text(), "owner telemetry")
            self.assertEqual(FakeBus.instances, [])

    def test_wire_roundtrip_signed_units(self):
        q = [-10.123, 12.456, -130.789, 3.001, -20.002, 40.003]
        decoded = [
            v / 1000
            for _, data in joint_frames(q, LIMITS)
            for v in struct.unpack(">ii", data)
        ]
        self.assertEqual(q, decoded)

    def test_gripper_frames_route_only_to_selected_gripper_and_accept_contact(self):
        for arm in ["left", "right"]:
            with (
                patch.object(motion, "CanBus", FakeBus),
                patch.object(motion.time, "monotonic", self.clock.monotonic),
            ):
                result = motion.set_gripper(
                    self.config, arm, 0, 1.0, 0.04, 0, self.root / "gripper.json"
                )
            bus = FakeBus.instances[-1]
            self.assertEqual(result["status"], "command_stream_completed")
            self.assertTrue(all(a == arm and cid == 0x159 for a, cid, _ in bus.sent))
            self.assertEqual(bus.sent[-1][2], bytes.fromhex("0000000003e80100"))
            self.assertEqual(result["final"][arm]["gripper_mm"], 25)
            self.assertEqual(bus.joints, {"left": START, "right": START})

    def test_gripper_homed_bit_and_signed_effort_are_decoded(self):
        frames = {cid: bytes(8) for cid in REQUIRED}
        frames[0x2A8] = struct.pack(">ihBB", 25123, -450, 0xC0, 0)
        result = decode_state(frames, {cid: 1 for cid in REQUIRED}, 1)
        self.assertEqual(result["gripper_mm"], 25.123)
        self.assertEqual(result["gripper_effort_nm"], -0.45)
        self.assertEqual(result["gripper_fault_bits"], 0)
        self.assertTrue(result["gripper_homed"])
        for width, effort in [(71, 1), (-1, 1), (30, 5.1), (math.nan, 1)]:
            with self.assertRaises(ValueError):
                gripper_frame(width, effort)

    def test_primitive_dry_run_opens_no_hardware_and_errors_are_structured(self):
        result = call(
            self.config,
            "move_left",
            dict(arm="left", target={"joint_deg": START}, dry_run=True),
        )
        self.assertTrue(result["ok"])
        self.assertFalse(result["data"]["executed"])
        self.assertFalse(call(self.config, "unknown", {})["ok"])
        for q in [[math.nan] * 6, [1000] * 6]:
            r = call(
                self.config,
                "move_right",
                dict(arm="right", target={"joint_deg": q}, dry_run=True),
            )
            self.assertFalse(r["ok"])
            self.assertIsNotNone(r["error"])

    def test_fixed_arm_tool_rejects_other_arm_override(self):
        r = call(
            self.config,
            "move_left",
            dict(arm="right", target={"joint_deg": START}, dry_run=True),
        )
        self.assertFalse(r["ok"])
        self.assertIn("cannot be overridden", r["error"])

    def test_second_camera_service_does_not_delete_live_owner_files(self):
        from agilex_control.cameras import serve

        sock = self.root / "cameras.sock"
        health = self.root / "camera_health.json"
        sock.write_text("existing socket marker")
        health.write_text("owner health")
        with (self.root / "cameras.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.dict(sys.modules, {"cv2": types.ModuleType("cv2")}):
                with self.assertRaises(BlockingIOError):
                    serve(self.config)
        self.assertEqual(sock.read_text(), "existing socket marker")
        self.assertEqual(health.read_text(), "owner health")

    def run_pair(self):
        targets = {"left": [2, 12, -22, 4, 32, -2], "right": [-1, 11, -21, -2, 31, 1]}
        with (
            patch.object(motion, "CanBus", FakeBus),
            patch.object(motion.time, "monotonic", self.clock.monotonic),
        ):
            report = motion.move_both(
                self.config, targets, 0.1, 0, self.root / "pair.json"
            )
        return report, targets

    def test_pair_uses_one_progress_clock_and_reaches_both_endpoints_after_host_stall(
        self,
    ):
        FakeBus.stall_once = True
        report, targets = self.run_pair()
        bus = FakeBus.instances[-1]
        self.assertEqual(report["status"], "command_stream_completed")
        self.assertEqual(bus.joints, targets)
        self.assertEqual(bus.writers, self.config["arms"])
        self.assertNotIn(0x159, [cid for _, cid, _ in bus.sent])
        progress = [row["trajectory_time"] for row in report["commands"]]
        self.assertTrue(
            all(b - a <= 1 / 30 + 1e-9 for a, b in zip(progress, progress[1:]))
        )
        for row in report["commands"]:
            for arm in targets:
                self.assertEqual(
                    row["q"][arm],
                    motion.interpolate(
                        START, targets[arm], row["trajectory_time"], 0.1
                    ),
                )
        for arm in targets:
            sent = [(cid, data) for a, cid, data in bus.sent if a == arm]
            self.assertEqual(sent[-3:], joint_frames(targets[arm], LIMITS))

    def test_pair_second_arm_send_failure_stops_both_without_later_ticks(self):
        FakeBus.fail_arm, FakeBus.fail_id = "right", 0x156
        report, _ = self.run_pair()
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(
            [(a, cid) for a, cid, _ in FakeBus.instances[-1].sent],
            [
                ("left", 0x151),
                ("left", 0x155),
                ("left", 0x156),
                ("left", 0x157),
                ("right", 0x151),
                ("right", 0x155),
            ],
        )

    def test_pair_requires_both_feedback_ready_before_any_write(self):
        FakeBus.unhealthy_arm = "right"
        report, _ = self.run_pair()
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(FakeBus.instances[-1].sent, [])
        self.assertEqual(FakeBus.instances[-1].writers, {})

    def test_pair_joint_dry_run_validates_both_before_hardware(self):
        targets = {"left": {"joint_deg": START}, "right": {"joint_deg": START}}
        with patch.object(
            motion, "CanBus", side_effect=AssertionError("No CAN expected")
        ):
            good = call(self.config, "move_both", dict(targets=targets, dry_run=True))
            self.assertTrue(good["ok"])
            self.assertEqual(set(good["data"]["plans"]), {"left", "right"})
            self.assertFalse(good["data"]["executed"])
            for invalid in [
                {"left": targets["left"]},
                {**targets, "rear": targets["left"]},
                {**targets, "right": {"joint_deg": [1000] * 6}},
            ]:
                self.assertFalse(
                    call(self.config, "move_both", dict(targets=invalid, dry_run=True))[
                        "ok"
                    ]
                )

    def test_pair_stop_signal_ceases_both_target_streams(self):
        FakeBus.stop_after_first_tick = True
        report, _ = self.run_pair()
        self.assertEqual(report["status"], "aborted")
        self.assertIn("Stop requested", report["error"])
        self.assertEqual(len(report["commands"]), 1)
        self.assertEqual(len(FakeBus.instances[-1].sent), 8)

    def test_can_writers_route_by_config_and_reject_ambiguous_send(self):
        from unittest.mock import MagicMock
        from agilex_control import can_bus

        sockets = [MagicMock() for _ in range(4)]
        for sock in sockets:
            sock.send.return_value = 16
        with (
            patch.object(can_bus.socket, "socket", side_effect=sockets),
            patch.object(can_bus.socket, "PF_CAN", 29, create=True),
            patch.object(can_bus.socket, "CAN_RAW", 1, create=True),
        ):
            with can_bus.CanBus(self.config["arms"]) as bus:
                with self.assertRaises(ValueError):
                    bus.open_writer("right", "can_left")
                bus.open_writer("left", "can_left")
                bus.open_writer("right", "can_right")
                with self.assertRaises(ValueError):
                    bus.send(0x151, MODE_FRAME)
                bus.send(0x151, MODE_FRAME, arm="right")
                sockets[2].send.assert_not_called()
                sockets[3].send.assert_called_once_with(
                    struct.pack("=IB3x8s", 0x151, 8, MODE_FRAME)
                )
                self.assertEqual(list(bus.expected), [("right", 0x151, MODE_FRAME)])
            for sock in sockets:
                sock.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
