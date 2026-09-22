"""Offline regressions against the site's installed pure FK model; no hardware."""

import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agilex_control.kinematics import Kinematics

CONFIG = json.loads((ROOT / "config/site.json").read_text())
CONFIG["fk_file"] = os.environ.get("AGILEX_FK_FILE", CONFIG["fk_file"])


@unittest.skipUnless(Path(CONFIG["fk_file"]).is_file(), "Site pure FK model absent")
class KinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = Kinematics(CONFIG)

    def state(self, q, arm="left"):
        return {
            arm: dict(
                joint_deg=list(q),
                pose_mm_deg=self.kin.pose(q).tolist(),
                complete=True,
                stale=[],
                arm_status=0,
                error_code=0,
            )
        }

    def assert_endpoint(self, plan, xyz, rotation):
        result = self.kin.transform(plan["q_target_deg"])
        self.assertLess(np.linalg.norm(result[:3, 3] - xyz), 0.01)
        error = (rotation.inv() * Rotation.from_matrix(result[:3, :3])).magnitude()
        self.assertLess(error, 1e-4)

    def test_observed_false_unreachable_from_same_initial_state_both_arms(self):
        q = [-35.763, 80.548, -48.588, -10.863, -28.322, 9.641]
        xyz, rpy = [230, -190, 322], [0, 90, -31]
        for arm in ("left", "right"):
            plan = self.kin.plan(self.state(q, arm), arm, xyz, rpy)
            self.assert_endpoint(
                plan, xyz, Rotation.from_euler("xyz", rpy, degrees=True)
            )

    def test_old_euler_zero_residual_is_not_true_rotation_success(self):
        q = [
            -43.27916187,
            76.52292496,
            -46.87942572,
            -27.58947887,
            -27.33562414,
            24.50342597,
        ]
        target = Rotation.from_euler("xyz", [0, 90, -31], degrees=True)
        raw = Rotation.from_matrix(self.kin.transform(q)[:3, :3])
        self.assertAlmostEqual(
            np.rad2deg((target.inv() * raw).magnitude()), 0.392233396, places=5
        )

    def test_both_pitch_singularities_and_neighborhood(self):
        perturbation = np.array([0.8, -1, -2, 2, 1, -1])
        for q in (
            [20, 80, -50, 0, -25, 0],
            [20, 140, -5, 0, 50, 0],
            [20, 80, -50, 0, -25.5, 0],
            [20, 140, -5, 0, 50.5, 0],
        ):
            pose = self.kin.pose(q)
            rotation = Rotation.from_matrix(self.kin.transform(q)[:3, :3])
            plan = self.kin.plan(
                self.state(np.array(q) + perturbation), "left", pose[:3], pose[3:]
            )
            self.assert_endpoint(plan, pose[:3], rotation)

    def test_omitted_orientation_preserves_untruncated_rotation(self):
        for q in ([20, 80, -50, 0, -25.5, 0], [20, 140, -5, 0, 50.5, 0]):
            transform = self.kin.transform(q)
            xyz = transform[:3, 3] + [2, 0, 0]
            plan = self.kin.plan(self.state(q), "left", xyz)
            self.assert_endpoint(plan, xyz, Rotation.from_matrix(transform[:3, :3]))

    def test_copied_start_euler_retries_current_rotation_on_translation(self):
        q = [20, 80, -50, 0, -25.5, 0]
        state = self.state(q, "right")
        pose = state["right"]["pose_mm_deg"]
        xyz = [pose[0] + 40.0, pose[1], pose[2]]
        plan = self.kin.plan(state, "right", xyz, pose[3:])
        self.assert_endpoint(
            plan, xyz, Rotation.from_matrix(self.kin.transform(q)[:3, :3])
        )

    def test_modified_dh_contract_matches_sdk_away_from_truncated_band(self):
        lo, hi = np.array(CONFIG["joint_limits_deg"]).T
        for q in np.random.default_rng(20260912).uniform(lo, hi, (100, 6)):
            transform = self.kin.transform(q)
            old = np.array(self.kin.fk.CalFK(np.deg2rad(q).tolist())[-1])
            np.testing.assert_allclose(transform[:3, 3], old[:3], atol=1e-9)
            if abs(transform[2, 0]) <= 0.9999:
                rotation = Rotation.from_euler("xyz", old[3:], degrees=True)
                error = (
                    rotation.inv() * Rotation.from_matrix(transform[:3, :3])
                ).magnitude()
                self.assertLess(error, 1e-9)

    def test_bad_feedback_and_unsolved_target_still_rejected(self):
        state = self.state([20, 80, -50, 0, -25, 0])
        with self.assertRaisesRegex(ValueError, "did not converge"):
            self.kin.plan(state, "left", [5000, 5000, 5000], [0, 90, 0])
        state["left"]["stale"] = ["joint_feedback"]
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.kin.plan(state, "left", [200, 0, 300])


if __name__ == "__main__":
    unittest.main()
