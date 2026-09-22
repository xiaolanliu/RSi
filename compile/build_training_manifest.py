"""Validate and index per-root visual caches for multi-root training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-parent", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--camera-key", default="global_image")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    records = []
    incomplete = []
    for root in sorted(args.data_parent.iterdir()):
        info_path = root / "meta" / "info.json"
        if not root.is_dir() or not info_path.is_file() or not any((root / "data").glob("chunk-*/*.parquet")):
            continue
        info = json.loads(info_path.read_text())
        expected = int(info.get("total_episodes", 0))
        cache_dir = args.cache_root / "cache" / root.name / args.camera_key
        cache_files = sorted(cache_dir.glob("episode_*.npz"))
        missing = []
        bad = []
        total_frames = 0
        for episode_index in range(expected):
            path = cache_dir / f"episode_{episode_index:06d}.npz"
            if not path.is_file():
                missing.append(episode_index)
                continue
            try:
                with np.load(path) as cached:
                    features = np.asarray(cached["features"])
                    frame_indices = np.asarray(cached["frame_indices"] if "frame_indices" in cached else [])
                    if features.ndim != 2 or features.shape[1] != 48 or features.shape[0] != len(frame_indices):
                        bad.append({"episode": episode_index, "shape": list(features.shape), "frame_indices": len(frame_indices)})
                    else:
                        total_frames += int(features.shape[0])
            except Exception as exc:
                bad.append({"episode": episode_index, "error": f"{type(exc).__name__}: {exc}"})
        valid = not missing and not bad
        if not valid:
            incomplete.append({"data_root": str(root.resolve()), "missing": missing, "bad": bad})
        records.append({
            "data_root": str(root.resolve()),
            "dataset_name": root.name,
            "visual_cache_dir": str(cache_dir.resolve()),
            "camera_key": args.camera_key,
            "episodes": expected,
            "cached_files": len(cache_files),
            "cached_frames": total_frames,
            "valid": valid,
        })
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(f"Incomplete visual caches for {len(incomplete)} roots; use --allow-incomplete only for inspection")
    selected = [record for record in records if record["valid"]]
    report = {
        "data_parent": str(args.data_parent.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "camera_key": args.camera_key,
        "feature_dim": 48,
        "roots": selected,
        "all_roots": records,
        "incomplete": incomplete,
        "episodes": sum(record["episodes"] for record in selected),
        "frames": sum(record["cached_frames"] for record in selected),
        "complete": not incomplete,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": report["complete"], "roots": len(selected), "episodes": report["episodes"], "frames": report["frames"], "incomplete_roots": len(incomplete)}, indent=2))


if __name__ == "__main__":
    main()
