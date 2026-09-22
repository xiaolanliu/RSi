"""Audit all camera episode mappings below a LeRobot dataset parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .vision import resolve_video_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-keys", nargs="+", default=["global_image", "left_image", "right_image"])
    args = parser.parse_args()
    records = []
    for root in sorted(args.data_parent.iterdir()):
        info_path = root / "meta" / "info.json"
        if not root.is_dir() or not info_path.is_file() or not any((root / "data").glob("chunk-*/*.parquet")):
            continue
        info = json.loads(info_path.read_text())
        expected = int(info.get("total_episodes", 0))
        cameras = {}
        errors = []
        for key in args.camera_keys:
            try:
                episodes = resolve_video_episodes(root, key)
                missing_paths = [str(episode.video_path) for episode in episodes if not episode.video_path.is_file()]
                cameras[key] = {
                    "episodes": len(episodes),
                    "expected_episodes": expected,
                    "all_paths_exist": not missing_paths,
                    "missing_paths": missing_paths,
                    "length_sum": sum(episode.length for episode in episodes),
                }
                if len(episodes) != expected or missing_paths:
                    errors.append(f"{key}: episode/path mismatch")
            except Exception as exc:
                cameras[key] = {"error": f"{type(exc).__name__}: {exc}"}
                errors.append(f"{key}: {type(exc).__name__}")
        records.append({
            "data_root": str(root.resolve()),
            "dataset_name": root.name,
            "episodes": expected,
            "frames": int(info.get("total_frames", 0)),
            "fps": info.get("fps"),
            "cameras": cameras,
            "valid": not errors,
            "errors": errors,
        })
        print(json.dumps({"dataset": root.name, "valid": not errors, "errors": errors}), flush=True)
    report = {
        "data_parent": str(args.data_parent.resolve()),
        "camera_keys": args.camera_keys,
        "roots": records,
        "valid_roots": [record["data_root"] for record in records if record["valid"]],
        "invalid_roots": [record["data_root"] for record in records if not record["valid"]],
        "total_valid_episodes": sum(record["episodes"] for record in records if record["valid"]),
        "total_valid_frames": sum(record["frames"] for record in records if record["valid"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("total_valid_episodes", "total_valid_frames", "invalid_roots")}, indent=2))


if __name__ == "__main__":
    main()
