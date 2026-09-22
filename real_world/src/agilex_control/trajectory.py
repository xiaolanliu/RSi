"""Actuator-specific encoding; both trajectories use the same execution loop."""

from dataclasses import dataclass
from math import isfinite
from .protocol import MODE_FRAME, gripper_frame, joint_frames, validate_joints


@dataclass
class JointTrajectory:
    target: list
    config: dict
    kind = "joints"
    units = "deg"
    refresh_mode = MODE_FRAME

    def __post_init__(self):
        validate_joints(self.target, self.config["joint_limits_deg"])

    def start(self, state):
        values = state["joint_deg"]
        validate_joints(values, self.config["joint_limits_deg"])
        return values

    def frames(self, values):
        return joint_frames(values, self.config["joint_limits_deg"])

    def check(self, state):
        pass

    def describe(self, start):
        return dict(q_start_deg=start, q_target_deg=self.target)


@dataclass
class GripperTrajectory:
    width_mm: float
    effort_nm: float
    config: dict
    arm: str | None = None
    kind = "gripper"
    units = "mm"
    refresh_mode = None

    def __post_init__(self):
        limits = self.config.get("gripper_limits_mm", [0, 70])
        self.limits = limits[self.arm] if isinstance(limits, dict) else limits
        self.target = [self.width_mm]
        self.frames(self.target)

    def start(self, state):
        self.check(state)
        measured = state["gripper_mm"]
        if not isfinite(measured):
            raise ValueError("Gripper feedback width must be finite")
        self.measured_start_mm = measured
        # Feedback can overshoot a command boundary; never encode that overshoot.
        # The original measurement remains in the execution baseline.
        return [min(self.limits[1], max(self.limits[0], measured))]

    def frames(self, values):
        return [
            (
                0x159,
                gripper_frame(
                    values[0],
                    self.effort_nm,
                    self.limits,
                ),
            )
        ]

    def check(self, state):
        if state["gripper_fault_bits"]:
            raise RuntimeError(
                "Gripper drive reports fault bits: " + str(state["gripper_fault_bits"])
            )

    def describe(self, start):
        return dict(
            width_start_mm=self.measured_start_mm,
            width_command_start_mm=start[0],
            width_target_mm=self.width_mm,
            effort_target_nm=self.effort_nm,
        )
