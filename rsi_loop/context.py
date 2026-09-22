"""Identical causal history for online requests and offline inspection."""
from pathlib import Path
import numpy as np

from .contracts import Observation


def history_entry(obs, source, alarm):
    from PIL import Image
    image = Image.fromarray(obs.images["cam_high"])
    image.thumbnail((320, 240))
    return dict(step=obs.step, time=obs.time, source=source, alarm=bool(alarm),
                state=obs.state.tolist(), image_top=np.asarray(image),
                measured_gripper_openings=None if obs.measured_gripper_openings is None
                else obs.measured_gripper_openings.tolist())


def select_history(history, now):
    past = [entry for entry in history if now-3 <= entry["time"] < now]
    indices = np.linspace(0, len(past)-1, min(3, len(past)), dtype=int) if past else []
    return [past[index] for index in indices]


def read_observation(run, step):
    from .recording import CAMERAS, read_rgb
    with np.load(Path(run)/"observations"/f"{step:06d}.npz", allow_pickle=False) as data:
        images = {name: data[name] if name in data else read_rgb(run, name, step, data[f"{name}_sha256"])
                  for name in CAMERAS}
        return Observation(str(data["episode_id"]), step, float(data["time"]), data["state"],
            images, str(data["instruction"]),
            data["eef_positions"], data["eef_quaternions"],
            measured_gripper_openings=data["measured_gripper_openings"] if "measured_gripper_openings" in data else None)
