"""Offline checks for ROS camera fallback. Does not start ROS or open USB."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agilex_control.ros_capture import should_fallback, _namespaces


class FallbackTests(unittest.TestCase):
    def test_busy_and_missing_socket_fall_back(self):
        self.assertTrue(should_fallback(RuntimeError("xioctl(VIDIOC_S_FMT) failed, errno=16 Last Error: Device or resource busy")))
        self.assertTrue(should_fallback(FileNotFoundError("Camera service is not running; missing /tmp/x/cameras.sock")))
        self.assertFalse(should_fallback(RuntimeError("Image write failed")))

    def test_default_namespaces(self):
        mapping = _namespaces({
            "cameras": {"front": "239622301704", "left_wrist": "346522074444", "right_wrist": "346522074314"},
        })
        self.assertEqual(mapping["front"]["namespace"], "/camera_f")
        self.assertEqual(mapping["left_wrist"]["namespace"], "/camera_l")
        self.assertEqual(mapping["right_wrist"]["serial"], "346522074314")


if __name__ == "__main__":
    unittest.main()
