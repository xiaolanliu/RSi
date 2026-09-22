"""Explicit boundaries between observations, monitors and command producers."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Observation:
    episode_id: str
    step: int
    time: float
    state: np.ndarray  # L6,gL,R6,gR; simulator grippers are normalized openings.
    images: dict[str, np.ndarray]  # RGB uint8; never simulator object poses.
    instruction: str
    eef_positions: np.ndarray | None = None  # robot proprioception only
    eef_quaternions: np.ndarray | None = None  # wxyz
    terminated: bool = False
    truncated: bool = False
    success: bool = False
    measured_gripper_openings: np.ndarray | None = None  # physical joints, separate from native command-state

    @property
    def identity(self):
        return self.episode_id, self.step


def action_array(value):
    a = np.asarray(value, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 14 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Expected nonempty finite [T,14] absolute joint targets")
    if np.any((a[:, [6, 13]] < 0) | (a[:, [6, 13]] > 1)):
        raise ValueError("Gripper targets must be normalized openings in [0,1]")
    return a


class Environment(Protocol):
    def reset(self, seed: int) -> Observation: ...
    def step(self, action: np.ndarray, source: str) -> Observation: ...
    def close(self) -> None: ...


class Policy(Protocol):
    def reset(self) -> None: ...
    def infer(self, observation: Observation) -> np.ndarray: ...


class Monitor(Protocol):
    def reset(self) -> None: ...
    def observe(self, observation: Observation) -> dict: ...


@dataclass(frozen=True)
class RecoveryPlan:
    observation_identity: tuple[str, int]
    diagnosis: str
    actions: np.ndarray
    source: str  # mock / gpt; a mock hold must never be called a GPT repair.
