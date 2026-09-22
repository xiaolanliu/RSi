"""Optional successful MP4 context using the pinned GPT-Policy extractor."""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np


class Demonstration:
    def __init__(self, video, *, task, cache, max_frames=8, provenance=None):
        from gpt_policy.input.video import FfmpegVideoExtractor, VideoProcessingConfig
        self.video = Path(video).resolve()
        self.task = task
        if not 1 <= max_frames <= 24:
            raise ValueError("Demo context must use 1..24 frames")
        with self.video.open("rb") as stream:
            self.sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self.cache = Path(cache) / self.sha256
        extractor = FfmpegVideoExtractor(VideoProcessingConfig(max_candidates=24, max_keyframes=max_frames))
        metadata = extractor.probe(self.video)
        pts = extractor.frame_times(self.video)
        times = np.linspace(pts[0], pts[-1], min(max_frames, len(pts)))
        self.frames = extractor.extract_at(self.video, times, self.cache, 640, pts)
        self.provenance = provenance or {"success_source": "user_provided_success_demonstration"}
        (self.cache / "manifest.json").write_text(json.dumps(dict(
            video=str(self.video), sha256=self.sha256, task=task, metadata=metadata.record(),
            frames=[frame.record(self.cache) for frame in self.frames], provenance=self.provenance,
            selection="deterministic_uniform_pts_no_llm"), indent=2))

    def content(self):
        content = [dict(type="input_text", text="SUCCESS DEMONSTRATION (reference only): " + self.task)]
        for frame in self.frames:
            content.append(dict(type="input_text", text=f"Demo timestamp {frame.timestamp_s:.3f}s"))
            content.append(dict(type="input_image", detail="auto", image_url="data:image/jpeg;base64," +
                                base64.b64encode(frame.path.read_bytes()).decode()))
        return content


def promote_success(run, destination):
    """Only native, complete VLA-only successes may bootstrap a success demo."""
    import shutil
    run, destination = Path(run), Path(destination)
    if (run/"control_test.json").exists():
        raise ValueError("A fault-injection control test cannot become a success demonstration")
    loop = json.loads((run / "loop_summary.json").read_text())
    native = json.loads((run / "native_outcome.json").read_text())
    if not (loop["complete"] and loop["native_success"] and native["valid_for_success_rate"]
            and native["native_success"] and loop["interventions"] == 0):
        raise ValueError("This run is not a complete native VLA-only success; cannot promote demo")
    source = run / "sensors.mp4"
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("Success has no recorded video")
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, destination / "success.mp4")
    (destination / "provenance.json").write_text(json.dumps(dict(
        source_run=str(run.resolve()), source="pi05_native_success", loop=loop, native=native), indent=2))
    return destination / "success.mp4"
