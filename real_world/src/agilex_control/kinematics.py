"""Offline FK/IK. Pose positions are link6 coordinates in millimetres."""

import importlib.util
import math
import warnings
import numpy as np
from .protocol import validate_joints


def mdh_link(alpha, a, theta, d):
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return [ct, -st, 0, a, ca * st, ca * ct, -sa, -sa * d,
            sa * st, sa * ct, ca, ca * d, 0, 0, 0, 1]


class Kinematics:
    def __init__(self, config):
        self.limits = config["joint_limits_deg"]
        if "mdh_mm_rad" in config:
            if "fk_file" in config:
                raise ValueError("Choose one geometry source: mdh_mm_rad or fk_file")
            # Per-link rows: d(mm), a(mm), alpha(rad), theta_offset(rad).
            mdh = np.asarray(config["mdh_mm_rad"], dtype=float)
            if mdh.shape != (6, 4) or not np.all(np.isfinite(mdh)):
                raise ValueError("mdh_mm_rad requires six finite four-value rows")
            self._dh = mdh.T[[2, 1, 3, 0]]
            self._link_transform = mdh_link
            return
        spec = importlib.util.spec_from_file_location(
            "piper_fk_offline", config["fk_file"]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.fk = module.C_PiperForwardKinematics(1)
        # Reuse the installed SDK's modified-DH model, without its lossy
        # matrix-to-Euler conversion near +/-90 degrees.
        self._link_transform = getattr(
            self.fk, "_C_PiperForwardKinematics__LinkTransformtion"
        )
        self._dh = np.array(
            [self.fk._alpha, self.fk._a, self.fk._theta, self.fk._d], dtype=float
        )
        if self._dh.shape != (4, 6) or not np.all(np.isfinite(self._dh)):
            raise ValueError("Unsupported SDK modified-DH parameter contract")

    def transform(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("Six finite joint angles required")
        result = np.eye(4)
        for i, joint in enumerate(np.deg2rad(q)):
            alpha, a, theta, d = self._dh[:, i]
            link = self._link_transform(alpha, a, joint + theta, d)
            result = result @ np.asarray(link, dtype=float).reshape(4, 4)
        return result

    def pose(self, q):
        from scipy.spatial.transform import Rotation

        transform = self.transform(q)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Gimbal lock detected.*")
            rpy = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True)
        return np.r_[transform[:3, 3], rpy]

    def plan(self, state, arm, xyz, rpy=None):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        s = state[arm]
        q0 = np.array(s["joint_deg"])
        validate_joints(q0, self.limits)
        if (
            not s.get("complete")
            or s.get("stale")
            or s["arm_status"]
            or s["error_code"]
        ):
            raise ValueError(
                "Planning requires complete, fresh, fault-free captured feedback"
            )
        if (
            len(xyz) != 3
            or not np.all(np.isfinite(xyz))
            or not np.all(np.isfinite(s["pose_mm_deg"]))
        ):
            raise ValueError("Finite Cartesian coordinates required")
        p0 = self.pose(q0)
        if np.linalg.norm(p0[:3] - s["pose_mm_deg"][:3]) > 0.1:
            raise ValueError("FK and feedback mismatch")
        current_rotation = Rotation.from_matrix(self.transform(q0)[:3, :3])
        target_rotation = (
            current_rotation
            if rpy is None
            else Rotation.from_euler("xyz", rpy, degrees=True)
        )

        def residual(q, rotation):
            transform = self.transform(q)
            return np.r_[
                (transform[:3, 3] - xyz) / 100,
                (
                    rotation.inv() * Rotation.from_matrix(transform[:3, :3])
                ).as_rotvec(),
            ]

        lo, hi = np.array(self.limits).T

        def solve(rotation):
            return least_squares(
                lambda q: residual(q, rotation),
                q0,
                bounds=(lo, hi),
                xtol=1e-11,
                ftol=1e-11,
                gtol=1e-11,
                max_nfev=1000,
            )

        fit = solve(target_rotation)
        if np.linalg.norm(fit.fun) > 1e-4 and rpy is not None:
            retry = solve(current_rotation)
            if np.linalg.norm(retry.fun) <= 1e-4:
                fit = retry
        if np.linalg.norm(fit.fun) > 1e-4:
            raise ValueError(
                "IK did not converge within configured limits: "
                f"position residual {np.linalg.norm(fit.fun[:3]) * 100:.3f} mm, "
                f"rotation residual {np.rad2deg(np.linalg.norm(fit.fun[3:])):.3f} deg"
            )
        path = np.array(
            [self.pose(q0 + (fit.x - q0) * u) for u in np.linspace(0, 1, 101)]
        )
        return dict(
            arm=arm,
            q_start_deg=q0.tolist(),
            q_target_deg=fit.x.tolist(),
            pose_start_mm_deg=p0.tolist(),
            pose_target_mm_deg=self.pose(fit.x).tolist(),
            joint_path_xyz_min_mm=path[:, :3].min(axis=0).tolist(),
            joint_path_xyz_max_mm=path[:, :3].max(axis=0).tolist(),
            validation="offline IK and sampled joint interpolation; no collision certification",
        )
