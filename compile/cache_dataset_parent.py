"""Plan and run resumable multi-camera Wan VAE caching for a dataset parent."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def discover_roots(parent: Path) -> list[tuple[Path, int]]:
    roots = []
    for root in sorted(parent.iterdir()):
        info_path = root / "meta" / "info.json"
        if not root.is_dir() or not info_path.is_file() or not any((root / "data").glob("chunk-*/*.parquet")):
            continue
        info = json.loads(info_path.read_text())
        roots.append((root.resolve(), int(info.get("total_frames", 0))))
    return roots


def balanced_shards(roots: list[tuple[Path, int]], count: int) -> list[list[tuple[Path, int]]]:
    shards: list[list[tuple[Path, int]]] = [[] for _ in range(count)]
    totals = [0] * count
    for item in sorted(roots, key=lambda value: (-value[1], value[0].name)):
        index = min(range(count), key=lambda candidate: totals[candidate])
        shards[index].append(item)
        totals[index] += item[1]
    return shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-parent", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--camera-keys", nargs="+", default=["global_image", "left_image", "right_image"])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=None,
                        help="Alias for num-shards when launching many independent workers.")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--vae-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.shard_count is not None:
        args.num_shards = args.shard_count
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    roots = discover_roots(args.data_parent)
    shards = balanced_shards(roots, args.num_shards)
    selected = shards[args.shard_index]
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "data_parent": str(args.data_parent.resolve()),
        "output_root": str(args.output_root.resolve()),
        "vae": str(args.vae.resolve()),
        "camera_keys": args.camera_keys,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "size": args.size,
        "frame_stride": args.frame_stride,
        "vae_batch_size": args.vae_batch_size,
        "roots": [{"data_root": str(root), "frames": frames} for root, frames in selected],
        "total_frames": sum(frames for _, frames in selected),
    }
    camera_tag = "-".join(args.camera_keys)
    plan_path = args.output_root / f"cache_plan_{camera_tag}_shard_{args.shard_index:02d}.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(plan_path), "roots": len(selected), "frames": plan["total_frames"]}), flush=True)
    if args.plan_only:
        return
    data_roots = [root for root, _ in selected]
    for camera_key in args.camera_keys:
        output_dirs = [args.output_root / "cache" / root.name / camera_key for root in data_roots]
        command = [
            sys.executable,
            "-m",
            "compile.cache_wan_vision",
            "--data-roots",
            *map(str, data_roots),
            "--output-dirs",
            *map(str, output_dirs),
            "--vae",
            str(args.vae),
            "--video-key",
            camera_key,
            "--size",
            str(args.size),
            "--frame-stride",
            str(args.frame_stride),
            "--vae-batch-size",
            str(args.vae_batch_size),
            "--device",
            args.device,
        ]
        print(json.dumps({"camera": camera_key, "status": "starting"}), flush=True)
        subprocess.run(command, check=True)
        print(json.dumps({"camera": camera_key, "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
