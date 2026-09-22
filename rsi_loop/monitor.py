"""Causal simulator clock and unit adapter for the frozen 30 Hz RSI head."""
from collections import deque
import numpy as np


class StreamingWan:
    def __init__(self, checkpoint, device="cuda:1"):
        import torch
        from third_party.wan22_vae.vae2_2 import Wan2_2_VAE
        self.device = torch.device(device)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.vae = Wan2_2_VAE(vae_pth=str(checkpoint), dtype=dtype, device=str(device))
        self.frames = deque(maxlen=5)

    def reset(self):
        self.frames.clear()

    def step(self, rgb):
        import torch
        from agent_closed_loop.visual_features import letterbox_frame
        self.frames.append(letterbox_frame(rgb, 256))
        frames = list(self.frames)
        frames = [frames[0]] * (5-len(frames)) + frames
        clip = torch.stack(frames, dim=1).to(self.device, self.vae.dtype)
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.vae.dtype,
                                                   enabled=self.device.type == "cuda"):
            latent = self.vae.model.encode(clip[None], self.vae.scale)[0].float()
        return latent.mean(dim=(1, 2, 3)).cpu().numpy()


class CausalMonitor:
    """Zero-order hold at 30 Hz, using only frames acquired by the sample time.

    This transfers the frozen head to simulation; it does NOT transfer its
    real-world normal-tail calibration guarantee. Telemetry records this fact.
    """
    def __init__(self, head, visual, gripper_open_width_m, camera="cam_high"):
        widths = np.asarray(gripper_open_width_m, np.float32)
        if widths.shape != (2,) or not np.isfinite(widths).all() or np.any(widths <= 0):
            raise ValueError("Provide two measured simulator full-open gripper widths in meters")
        self.head, self.visual, self.widths, self.camera = head, visual, widths, camera
        self.reset()

    def reset(self):
        self.head.reset()
        self.visual.reset()
        self.previous = None
        self.index = 0
        self.result = None
        self.start_time = None

    def observe(self, obs):
        if self.previous is not None:
            if obs.episode_id != self.previous.episode_id or obs.time <= self.previous.time:
                raise ValueError("Monitor requires strictly increasing episode time; reset first")
        else:
            self.start_time = obs.time
        emitted = 0
        while self.start_time + self.index / 30 <= obs.time + 1e-8:
            tick = self.start_time + self.index / 30
            sample = obs if abs(tick-obs.time) < 1e-8 or self.previous is None else self.previous
            state = np.asarray(sample.state, np.float32).copy()
            if state.shape != (14,) or np.any((state[[6, 13]] < 0) | (state[[6, 13]] > 1)):
                raise ValueError("Invalid normalized simulator gripper observation")
            if sample.measured_gripper_openings is not None:
                physical = np.asarray(sample.measured_gripper_openings, np.float32)
                if physical.shape != (2,) or not np.isfinite(physical).all():
                    raise ValueError("Invalid physical gripper joint measurement")
                state[[6, 13]] = physical
            state[[6, 13]] *= self.widths
            visual = self.visual.step(sample.images[self.camera])
            raw = self.head.step(state, visual, state_layout="left7_right7")
            self.result = {k: raw[k] for k in ("frame_index", "alarm", "risk_score", "risk_percentile",
                                              "confirmed_phase", "accepted_phase", "ood_signals")}
            self.result.update(sample_time=tick, source_time=sample.time)
            self.index += 1
            emitted += 1
        self.previous = obs
        return dict(self.result, emitted_samples=emitted, calibration_domain="real_folding_transferred_to_sim",
                    gripper_source="physical_joint" if obs.measured_gripper_openings is not None else "native_state",
                    clock_adapter="causal_zero_order_hold_30hz")
