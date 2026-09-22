"""Export milestone segmentations for every root in a fused cache manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-limit", type=int, default=1)
    parser.add_argument("--frame-review-limit", type=int, default=1)
    parser.add_argument("--video-key", default="global_image")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Keep only this fraction of roots (deterministic stride sample).")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Pass-through to compile.subtasks --max-episodes.")
    parser.add_argument("--keep-npz", action="store_true",
                        help="Keep per-episode npz posteriors (deleted by default).")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    manifest = json.loads(args.dataset_manifest.read_text())
    entries = manifest.get("roots", [])[args.shard_index :: args.num_shards]
    if not 0.0 < args.sample_frac <= 1.0:
        parser.error("sample-frac must be in (0, 1]")
    if args.sample_frac < 1.0:
        step = max(1, round(1.0 / args.sample_frac))
        entries = entries[::step]
    records = []
    for number, entry in enumerate(entries, 1):
        root = Path(entry["data_root"])
        output = args.output_dir / root.name
        command = [
            sys.executable, "-m", "compile.subtasks",
            "--data-root", str(root),
            "--visual-cache-dir", str(entry["visual_cache_dir"]),
            "--checkpoint", str(args.checkpoint),
            "--output-dir", str(output),
            "--plot-limit", str(args.plot_limit),
            "--device", args.device,
        ]
        if args.max_episodes is not None:
            command.extend(["--max-episodes", str(args.max_episodes)])
        subprocess.run(command, check=True)
        rows = [json.loads(line) for line in (output / "subtasks.jsonl").read_text().splitlines() if line.strip()]
        if not args.keep_npz:
            for npz in output.glob("episode_*.npz"):
                npz.unlink()
        if args.frame_review_limit > 0:
            episode_ids = [str(row["episode_index"]) for row in rows[: args.frame_review_limit]]
            review = [
                sys.executable, "-m", "compile.visualize_phase_frames",
                "--data-root", str(root),
                "--subtasks-jsonl", str(output / "subtasks.jsonl"),
                "--episodes", *episode_ids,
                "--output-dir", str(output / "phase_frame_reviews"),
                "--video-key", args.video_key,
            ]
            subprocess.run(review, check=True)
        records.append({
            "dataset_name": root.name,
            "data_root": str(root.resolve()),
            "output_dir": str(output.resolve()),
            "episodes": len(rows),
        })
        print(json.dumps({"root": number, "of": len(entries), **records[-1]}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "roots": records,
        "episodes": sum(record["episodes"] for record in records),
    }
    (args.output_dir / f"export_manifest_shard_{args.shard_index:02d}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
