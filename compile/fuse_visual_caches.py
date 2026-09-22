"""Concatenate aligned per-camera visual caches without re-encoding frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--camera-keys", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input_dirs) != len(args.camera_keys):
        parser.error("--input-dirs and --camera-keys must have equal lengths")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_names = sorted(path.name for path in args.input_dirs[0].glob("episode_*.npz"))
    if not episode_names:
        raise FileNotFoundError(f"No episode caches in {args.input_dirs[0]}")
    manifest: dict[str, object] = {
        "camera_keys": args.camera_keys,
        "input_dirs": [str(path.resolve()) for path in args.input_dirs],
        "output_dir": str(args.output_dir.resolve()),
        "episodes": [],
    }
    for name in episode_names:
        arrays = []
        frame_indices = None
        for directory in args.input_dirs:
            path = directory / name
            if not path.exists():
                raise FileNotFoundError(f"Missing aligned camera cache: {path}")
            with np.load(path) as cached:
                features = np.asarray(cached["features"], dtype=np.float32)
                current_indices = np.asarray(
                    cached["frame_indices"] if "frame_indices" in cached else np.arange(len(features)),
                    dtype=np.int64,
                )
            if frame_indices is None:
                frame_indices = current_indices
            elif not np.array_equal(frame_indices, current_indices):
                raise ValueError(f"Camera frame indices do not align for {name}")
            arrays.append(features)
        if len({array.shape[0] for array in arrays}) != 1:
            raise ValueError(f"Camera feature lengths do not align for {name}")
        fused = np.concatenate(arrays, axis=1)
        output = args.output_dir / name
        np.savez_compressed(output, features=fused, frame_indices=frame_indices)
        manifest["episodes"].append(
            {
                "cache": name,
                "length": int(fused.shape[0]),
                "feature_dim": int(fused.shape[1]),
            }
        )
    manifest["feature_dim"] = int(sum(array.shape[1] for array in arrays))
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"episodes": len(episode_names), "feature_dim": manifest["feature_dim"]}))


if __name__ == "__main__":
    main()
