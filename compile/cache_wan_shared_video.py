"""Cache one camera stream once and assign latent features to its episodes.

Wrist-camera LeRobot exports often pack many episodes into one MP4. The
regular episode encoder reopens that MP4 for every episode; this implementation
decodes each shared file once, extracts all requested causal windows, and
writes per-episode dense caches with a real batched VAE call.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch

from .vision import _existing_video_candidate, letterbox_frame, resolve_video_episodes


def _flush(vae: Any, pending: list[tuple[int, int, torch.Tensor]], output: dict[tuple[int, int], np.ndarray], device: torch.device) -> None:
    if not pending:
        return
    clips = torch.stack([row[2] for row in pending]).to(device=device, dtype=vae.dtype)
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=vae.dtype, enabled=device.type == "cuda"):
        encoded = vae.model.encode(clips, vae.scale).float() if hasattr(vae, "model") and hasattr(vae, "scale") else None
    if encoded is None:
        encoded = torch.stack([vae.encode([row[2].to(device=device, dtype=vae.dtype)])[0] for row in pending])
    for (episode_index, target_index, _), latent in zip(pending, encoded):
        output[(episode_index, target_index)] = latent.mean(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float32)
    pending.clear()


def cache_group(vae: Any, group: list[Any], output_root: Path, size: int, stride: int, batch_size: int, device: torch.device) -> list[dict[str, Any]]:
    video_path = group[0].video_path
    fps = group[0].fps
    # Each episode contributes local target indices; ranges are translated to
    # source timeline by its metadata timestamp.
    requests: dict[int, list[tuple[int, int]]] = defaultdict(list)
    by_episode = {episode.episode_index: episode for episode in group}
    max_timestamp = 0.0
    for episode in group:
        for local_index in sorted(set(range(0, episode.length, stride)) | {episode.length - 1}):
            source_index = int(round((episode.start_time_seconds + local_index / fps) * fps))
            requests[source_index].append((episode.episode_index, local_index))
            max_timestamp = max(max_timestamp, episode.start_time_seconds + episode.length / fps)
    pending: list[tuple[int, int, torch.Tensor]] = []
    encoded: dict[tuple[int, int], np.ndarray] = {}
    history: deque[np.ndarray] = deque(maxlen=5)
    container = av.open(str(video_path), options={"threads": "1", "hwaccel": "cuda", "hwaccel_output_format": "cuda"})
    try:
        stream = container.streams.video[0]
        for source_index, frame in enumerate(container.decode(stream)):
            if source_index > int(round(max_timestamp * fps)) + 2:
                break
            # Keep the causal history for every source frame. Only requested
            # endpoints are converted into VAE clips below.
            current_frame = frame.to_ndarray(format="rgb24")
            history.append(current_frame)
            if source_index not in requests:
                continue
            # The request map may contain multiple episode timestamps that
            # happen to share one source frame; the frame history is causal.
            window = list(history)
            while len(window) < 5:
                window.insert(0, window[0])
            clip = torch.stack([letterbox_frame(item, size) for item in window], dim=1)
            for episode_index, local_index in requests[source_index]:
                pending.append((episode_index, local_index, clip))
            if len(pending) >= batch_size:
                _flush(vae, pending, encoded, device)
    finally:
        container.close()
    _flush(vae, pending, encoded, device)
    records = []
    for episode in group:
        indices = np.asarray(sorted(set(range(0, episode.length, stride)) | {episode.length - 1}), dtype=np.int64)
        sparse = np.asarray([encoded[(episode.episode_index, int(index))] for index in indices], dtype=np.float32)
        dense = np.stack([np.interp(np.arange(episode.length), indices, sparse[:, dim]) for dim in range(48)], axis=1).astype(np.float32)
        out = output_root / f"episode_{episode.episode_index:06d}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            np.savez_compressed(out, features=dense, encoded_indices=indices, frame_indices=np.arange(episode.length, dtype=np.int64))
        records.append({"episode_index": episode.episode_index, "length": episode.length, "cache": out.name, "encoded_count": len(indices), "video": str(video_path.resolve())})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--video-key", required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--vae-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.data_roots) != len(args.output_dirs):
        parser.error("data-roots and output-dirs must have equal lengths")
    sys_root = args.vae.resolve().parents[2]
    import sys
    sys.path.insert(0, str(sys_root))
    from third_party.wan22_vae.vae2_2 import Wan2_2_VAE
    device = torch.device(args.device)
    vae = Wan2_2_VAE(vae_pth=str(args.vae), dtype=torch.bfloat16 if device.type == "cuda" else torch.float32, device=str(device))
    all_records = []
    for root, output in zip(args.data_roots, args.output_dirs):
        episodes = [
            episode
            for episode in resolve_video_episodes(root, args.video_key)
            if not (output / f"episode_{episode.episode_index:06d}.npz").is_file()
        ]
        if not episodes:
            continue
        groups: dict[Path, list[Any]] = defaultdict(list)
        for episode in episodes:
            groups[episode.video_path].append(episode)
        root_records = []
        for path, group in groups.items():
            root_records.extend(cache_group(vae, group, output, args.size, args.frame_stride, args.vae_batch_size, device))
        manifest = {"data_root": str(root.resolve()), "output_dir": str(output.resolve()), "video_key": args.video_key, "size": args.size, "frame_stride": args.frame_stride, "feature_dim": 48, "episodes": sorted(root_records, key=lambda row: row["episode_index"]), "shared_video_decode": True}
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        all_records.extend(root_records)
    print(json.dumps({"video_key": args.video_key, "episodes": len(all_records), "shared_video_decode": True}, indent=2))


if __name__ == "__main__":
    main()
