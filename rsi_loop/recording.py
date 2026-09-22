"""Lossless RGB storage with a per-observation byte identity check."""
import hashlib
from pathlib import Path

import numpy as np


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class LosslessRGB:
    def __init__(self, output, fps):
        self.output, self.fps = Path(output), fps
        self.output.mkdir(parents=True, exist_ok=True)
        self.writers = {}

    def append(self, images):
        import imageio.v2 as imageio
        hashes = {}
        for name, rgb in images.items():
            if name not in CAMERAS:
                raise ValueError("Unknown native camera")
            if name not in self.writers:
                self.writers[name] = imageio.get_writer(str(self.output/f"{name}.mkv"),
                    fps=self.fps, codec="libx264rgb", pixelformat="rgb24", macro_block_size=1,
                    output_params=["-crf", "0", "-preset", "fast", "-threads", "2"])
            self.writers[name].append_data(rgb)
            hashes[f"{name}_sha256"] = hashlib.sha256(np.asarray(rgb, np.uint8).tobytes()).hexdigest()
        return hashes

    def close(self):
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()


def read_rgb(run, name, step, expected_sha256):
    import av
    with av.open(str(Path(run)/"rgb"/f"{name}.mkv")) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == step:
                rgb = frame.to_ndarray(format="rgb24")
                if hashlib.sha256(rgb.tobytes()).hexdigest() != str(expected_sha256):
                    raise ValueError(f"RGB checksum mismatch: {name}, step {step}")
                return rgb
    raise ValueError(f"Missing RGB frame: {name}, step {step}")
