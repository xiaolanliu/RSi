import unittest

from agilex_control.trajectory import GripperTrajectory


class GripperFeedbackBoundaryTests(unittest.TestCase):
    def test_open_from_negative_closed_feedback(self):
        trajectory = GripperTrajectory(65, 1, {"gripper_limits_mm": [0, 70]})
        state = {"gripper_mm": -0.07, "gripper_fault_bits": 0}
        start = trajectory.start(state)
        self.assertEqual(start, [0])
        self.assertEqual(state["gripper_mm"], -0.07)
        self.assertEqual(trajectory.describe(start)["width_start_mm"], -0.07)
        self.assertEqual(trajectory.describe(start)["width_command_start_mm"], 0)
        trajectory.frames(start)

    def test_close_from_above_per_arm_command_range(self):
        config = {"gripper_limits_mm": {"left": [0, 70], "right": [0, 100]}}
        trajectory = GripperTrajectory(0, 1, config, "right")
        self.assertEqual(trajectory.start({"gripper_mm": 100.2, "gripper_fault_bits": 0}), [100])

    def test_invalid_target_and_faults_still_rejected(self):
        with self.assertRaises(ValueError):
            GripperTrajectory(-0.07, 1, {"gripper_limits_mm": [0, 70]})
        trajectory = GripperTrajectory(65, 1, {})
        with self.assertRaises(RuntimeError):
            trajectory.start({"gripper_mm": -0.07, "gripper_fault_bits": 1})
        with self.assertRaises(ValueError):
            trajectory.start({"gripper_mm": float("nan"), "gripper_fault_bits": 0})
