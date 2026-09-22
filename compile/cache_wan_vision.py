"""Cache frozen Wan2.2 VAE observation features for CompILE training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .vision import encode_episode_features, resolve_video_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--data-roots", type=Path, nargs="+", default=None)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--video-key", default=None)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--vae-batch-size", type=int, default=8)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sieve-root",
        type=Path,
        default=None,
        help="Directory containing third_party/wan22_vae (defaults to the VAE bundle's repository root)",
    )
    args = parser.parse_args()
    roots = args.data_roots or ([args.data_root] if args.data_root is not None else [])
    outputs = args.output_dirs or ([args.output_dir] if args.output_dir is not None else [])
    if not roots or not outputs:
        parser.error("provide --data-root/--output-dir or --data-roots/--output-dirs")
    if len(roots) != len(outputs):
        parser.error("data roots and output dirs must have equal lengths")
    if args.data_roots is not None and args.output_dir is not None:
        parser.error("use --output-dirs with --data-roots")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Wan VAE feature extraction requires CUDA")
    for output_dir in outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
    sieve_root = args.sieve_root
    if sieve_root is None:
        candidate = args.vae.resolve().parents[2]
        sieve_root = candidate if (candidate / "third_party").exists() else None
    if sieve_root is None or not (sieve_root / "third_party" / "wan22_vae").exists():
        raise FileNotFoundError(
            "Cannot locate SIEVE/third_party/wan22_vae; pass --sieve-root explicitly"
        )
    sys.path.insert(0, str(sieve_root.resolve()))
    from third_party.wan22_vae.vae2_2 import Wan2_2_VAE

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = Wan2_2_VAE(vae_pth=str(args.vae), dtype=dtype, device=str(device))
    manifest = {
        "data_roots": [str(root.resolve()) for root in roots],
        "output_dirs": [str(output.resolve()) for output in outputs],
        "vae": str(args.vae.resolve()),
        "video_key": args.video_key,
        "size": args.size,
        "frame_stride": args.frame_stride,
        "vae_batch_size": args.vae_batch_size,
        "feature_dim": 48,
        "datasets": [],
    }
    manifest_paths = [output / "manifest.json" for output in outputs]
    # Resume a long multi-root job even if a previous process was interrupted
    # before its final manifest write. Existing episode files are authoritative;
    # their metadata is reconstructed below and then updated incrementally.
    previous_by_root: dict[str, dict] = {}
    for path in manifest_paths:
        if path.exists():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(previous.get("datasets"), list):
                    for previous_dataset in previous["datasets"]:
                        data_root = previous_dataset.get("data_root")
                        if not isinstance(data_root, str):
                            continue
                        merged = previous_by_root.setdefault(
                            data_root,
                            {
                                "data_root": data_root,
                                "output_dir": previous_dataset.get("output_dir"),
                                "episodes": [],
                            },
                        )
                        records = {
                            int(record["episode_index"]): record
                            for record in merged["episodes"]
                            if "episode_index" in record
                        }
                        records.update(
                            {
                                int(record["episode_index"]): record
                                for record in previous_dataset.get("episodes", [])
                                if "episode_index" in record
                            }
                        )
                        merged["episodes"] = [records[index] for index in sorted(records)]
            except (OSError, json.JSONDecodeError):
                pass
    manifest["datasets"] = list(previous_by_root.values())

    def write_manifests() -> None:
        payload = json.dumps(manifest, indent=2) + "\n"
        for path in manifest_paths:
            path.write_text(payload, encoding="utf-8")

    started = time.perf_counter()
    for root, output_dir in zip(roots, outputs):
        episodes = resolve_video_episodes(root, args.video_key)
        if args.max_episodes is not None:
            episodes = episodes[: args.max_episodes]
        dataset_manifest = next(
            (
                item
                for item in manifest["datasets"]
                if item.get("data_root") == str(root.resolve())
            ),
            None,
        )
        if dataset_manifest is None:
            dataset_manifest = {"data_root": str(root.resolve()), "output_dir": str(output_dir.resolve()), "episodes": []}
            manifest["datasets"].append(dataset_manifest)
        existing_records = {int(item["episode_index"]): item for item in dataset_manifest["episodes"]}
        for number, episode in enumerate(episodes, 1):
            output = output_dir / f"episode_{episode.episode_index:06d}.npz"
            if output.exists():
                print(f"[{root.name} {number}/{len(episodes)}] skip {output.name}", flush=True)
                try:
                    with np.load(output) as cached:
                        encoded_count = int(len(cached["encoded_indices"]))
                except (KeyError, OSError, ValueError) as exc:
                    raise RuntimeError(f"Invalid existing visual cache: {output}") from exc
                status = "cached"
            else:
                features, encoded_indices = encode_episode_features(
                    vae,
                    episode,
                    size=args.size,
                    frame_stride=args.frame_stride,
                    batch_size=args.vae_batch_size,
                    device=device,
                )
                np.savez_compressed(output, features=features, encoded_indices=encoded_indices, frame_indices=np.arange(episode.length, dtype=np.int64))
                encoded_count = int(len(encoded_indices))
                status = "encoded"
                print(json.dumps({"dataset": root.name, "episode": episode.episode_index, "length": episode.length, "encoded_count": encoded_count}), flush=True)
            existing_records[episode.episode_index] = {"episode_index": episode.episode_index, "length": episode.length, "video": str(episode.video_path.resolve()), "cache": output.name, "encoded_count": encoded_count, "status": status}
            dataset_manifest["episodes"] = [existing_records[key] for key in sorted(existing_records)]
            manifest["elapsed_seconds"] = time.perf_counter() - started
            write_manifests()
    manifest["elapsed_seconds"] = time.perf_counter() - started
    # Keep a manifest next to the first output for single-root compatibility;
    # multi-root runs also get one manifest per output directory.
    write_manifests()


if __name__ == "__main__":
    main()
