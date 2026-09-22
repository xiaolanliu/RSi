"""Unit boundary between Piper feedback, pi0.5, RSi, and GPT recovery.

Piper SDK integers are 0.001 degree and 0.001 mm. clothesfoldingv2's
PiperAdapter multiplies by 1e-3 and exposes degrees and millimetres.
OpenPI then converts joints with deg2rad and grippers with *1e-3, and the
action postprocessor converts the policy output back before JointCtrl.

RSi expects that same radians/metres vector in joints12_grippers2 order:
L1..L6, R1..R6, gL, gR. It does not want left7_right7, and it does not want
Cartesian millimetres. GPT recovery stays on agilex_control's degree/mm
commands and must not be given the RSi vector.
"""

from __future__ import annotations

import hashlib
import sys
from collections import deque
from pathlib import Path

import numpy as np

WAN_VAE_SHA256 = "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
# Measured on this RTX 4060: fp32 Wan encode peaks at 3091 MiB and takes ~92 ms.
# bf16 weights drop that peak to 1748 MiB, which still does not fit beside Pi0.5.
WAN_CUDA_PEAK_MIB = 3200
PI05_WEIGHT_MIB = 6.2 * 1024
PI05_JAX_FRACTION = 0.88
DEFAULT_VAE_PATH = Path(__file__).resolve().parents[3] / "weights" / "Wan2.2_VAE.pth"
PIPER_STATE_KEYS = tuple(
    [f"{side}_joint_{index}.pos" for side in ("left", "right") for index in range(1, 7)]
    + ["left_gripper.pos", "right_gripper.pos"]
)


def piper_deg_mm_to_rsi_state14(values):
    """Match clothesfolding ``encode_openpi_request``: rad joints, metre grippers."""
    if isinstance(values, dict):
        values = [float(values[key]) for key in PIPER_STATE_KEYS]
    raw = np.asarray(values, dtype=np.float32).reshape(14)
    if not np.isfinite(raw).all():
        raise ValueError("Piper state14 contains a non-finite value")
    state = np.empty(14, dtype=np.float32)
    state[:12] = np.deg2rad(raw[:12])
    state[12:] = raw[12:] * np.float32(1e-3)
    return state


def assert_rsi_state14(state14):
    """Reject a vector that is still in Piper degrees or millimetres.

    Legal Piper poses stay inside about +/-180 deg, so a joint component
    above 3.6 rad, or a gripper opening above 0.15 m, is still in robot units.
    """
    state = np.asarray(state14, dtype=np.float32).reshape(14)
    if not np.isfinite(state).all():
        raise ValueError("RSi state14 contains a non-finite value")
    if float(np.max(np.abs(state[:12]))) > 3.6:
        raise ValueError(
            "state14 joints are still in degrees; convert with piper_deg_mm_to_rsi_state14"
        )
    if float(np.max(np.abs(state[12:]))) > 0.15:
        raise ValueError(
            "state14 grippers are still in millimetres; convert with piper_deg_mm_to_rsi_state14"
        )
    return state


def rsi_observation(robot_state_deg_mm, rgb, encoder):
    """Build one live RSi frame from Piper deg/mm feedback and an RGB image."""
    state14 = assert_rsi_state14(piper_deg_mm_to_rsi_state14(robot_state_deg_mm))
    visual48 = np.asarray(encoder.encode(rgb), dtype=np.float32)
    if visual48.shape != (48,) or not np.isfinite(visual48).all():
        raise ValueError("Wan encoder must return a finite visual48 vector")
    return {"state14": state14, "visual48": visual48}


def shared_gpu_budget(total_mib=8188, wan_peak_mib=WAN_CUDA_PEAK_MIB):
    """Return whether this card can hold Pi0.5 and the Wan encoder at once.

    ``serve_local_policy.py`` locks JAX to 88% of an 8GB card so the 6.2GiB
    Pi0.5 restore has one contiguous pool. The leftover slice is for that
    process's cuBLAS handle, not for a second model.
    """
    jax_pool = PI05_JAX_FRACTION * float(total_mib)
    leftover = float(total_mib) - jax_pool
    weights_plus_wan = PI05_WEIGHT_MIB + float(wan_peak_mib)
    return {
        "fits": leftover >= wan_peak_mib and weights_plus_wan < float(total_mib),
        "jax_pool_mib": jax_pool,
        "leftover_mib": leftover,
        "wan_peak_mib": float(wan_peak_mib),
        "pi05_weight_mib": PI05_WEIGHT_MIB,
        "weights_plus_wan_mib": weights_plus_wan,
    }


def verify_wan_vae(path=DEFAULT_VAE_PATH):
    """Return the official checkpoint path only when the published SHA256 matches."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Wan2.2 VAE checkpoint is missing: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != WAN_VAE_SHA256:
        raise ValueError(
            "Wan2.2 VAE checksum mismatch: expected %s, got %s" % (WAN_VAE_SHA256, actual)
        )
    return path


class CausalWanVisual:
    """Online visual48 using RSi's letterbox and five-frame Wan window."""

    def __init__(self, vae_path=DEFAULT_VAE_PATH, rsi_root=None, device="cuda", size=256):
        import torch

        self.path = verify_wan_vae(vae_path)
        root = Path(rsi_root) if rsi_root else self.path.parents[1]
        root_text = str(root.resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from agent_closed_loop.visual_features import letterbox_frame
        from third_party.wan22_vae.vae2_2 import Wan2_2_VAE

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        if str(device).startswith("cuda"):
            free_mib = torch.cuda.mem_get_info()[0] / (1024 ** 2)
            if free_mib < WAN_CUDA_PEAK_MIB:
                budget = shared_gpu_budget()
                raise RuntimeError(
                    "Wan2.2 VAE needs %.0f MiB free and this GPU has %.0f MiB. "
                    "Pi0.5 keeps a %.0f MiB JAX pool on an 8GB card, leaving %.0f MiB, "
                    "so the two cannot stay resident together. Keep OnlineMonitor on CPU "
                    "and stop the policy server before encoding live visual48."
                    % (
                        WAN_CUDA_PEAK_MIB,
                        free_mib,
                        budget["jax_pool_mib"],
                        budget["leftover_mib"],
                    )
                )
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self.device = device
        self.size = int(size)
        self.letterbox = letterbox_frame
        self.vae = Wan2_2_VAE(vae_pth=str(self.path), dtype=dtype, device=str(device))
        self.frames = deque(maxlen=5)

    def reset(self):
        self.frames.clear()

    def encode(self, rgb):
        import torch

        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError("Wan encoder expects one RGB uint8 frame [H,W,3]")
        self.frames.append(frame)
        window = list(self.frames)
        while len(window) < 5:
            window.insert(0, window[0])
        clip = torch.stack([self.letterbox(item, self.size) for item in window], dim=1)
        clip = clip.to(device=self.device, dtype=self.vae.dtype)
        device_type = torch.device(self.device).type
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device_type,
            dtype=self.vae.dtype,
            enabled=device_type == "cuda",
        ):
            encoded = self.vae.model.encode(clip.unsqueeze(0), self.vae.scale).float()
        pooled = encoded[0].mean(dim=(1, 2, 3)).detach().cpu().numpy()
        if pooled.shape != (48,):
            raise RuntimeError("Unexpected Wan latent shape: %s" % (pooled.shape,))
        return pooled.astype(np.float32)


def audit_unit_handoff():
    """Check the numeric boundaries used on the real robot. Raises on a mismatch."""
    sample = {
        "left_joint_1.pos": 180.0,
        "left_joint_2.pos": 90.0,
        "left_joint_3.pos": -45.0,
        "left_joint_4.pos": 0.0,
        "left_joint_5.pos": 10.0,
        "left_joint_6.pos": -10.0,
        "right_joint_1.pos": 1.0,
        "right_joint_2.pos": 2.0,
        "right_joint_3.pos": 3.0,
        "right_joint_4.pos": 4.0,
        "right_joint_5.pos": 5.0,
        "right_joint_6.pos": 6.0,
        "left_gripper.pos": 70.0,
        "right_gripper.pos": 0.0,
    }
    state = piper_deg_mm_to_rsi_state14(sample)
    expected_joints = np.deg2rad(np.asarray(
        [180, 90, -45, 0, 10, -10, 1, 2, 3, 4, 5, 6], dtype=np.float32
    ))
    if not np.allclose(state[:12], expected_joints, rtol=0, atol=1e-6):
        raise AssertionError("joint deg->rad does not match OpenPI encode_openpi_request")
    if not np.isclose(state[12], np.float32(0.07)) or not np.isclose(state[13], np.float32(0.0)):
        raise AssertionError("gripper mm->m does not match OpenPI metres")
    if np.isclose(state[12], np.deg2rad(np.float32(70.0))):
        raise AssertionError("gripper was converted as an angle")
    rsi_root = DEFAULT_VAE_PATH.parents[1]
    if str(rsi_root) not in sys.path:
        sys.path.insert(0, str(rsi_root))
    from agent_closed_loop.action_units import packed_state

    packed = packed_state(state, "joints12_grippers2")
    if not np.array_equal(packed, state):
        raise AssertionError("joints12_grippers2 must keep L6,R6,gL,gR")
    permuted = packed_state(state, "left7_right7")
    if np.array_equal(permuted, state):
        raise AssertionError("left7_right7 unexpectedly matched the Piper order")
    return {
        "piper_feedback": "0.001 deg and 0.001 mm, then *1e-3 -> deg/mm",
        "pi05_and_rsi_state": "deg2rad joints, mm*1e-3 grippers, order L6 R6 gL gR",
        "pi05_action_back": "rad->deg, m*1000->mm, then Piper *1000 integer command",
        "gpt_recovery": "joint_deg and xyz_mm, not the RSi radian vector",
        "visual": "RGB uint8, 5-frame letterbox 256, Wan2.2 pool to float32[48]",
        "left_gripper_m": float(state[12]),
        "left_joint_1_rad": float(state[0]),
    }
