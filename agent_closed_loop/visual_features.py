"""Wan2.2 VAE feature extraction for LeRobot episodes.

The VAE is used as a frozen observation encoder only. A causal five-frame
window ending at frame ``t`` is encoded and pooled over the VAE temporal and
spatial latent axes to a compact 48-dimensional feature. The cache format is
deliberately simple (one ``episode_*.npz`` per episode) so training does not
load the multi-gigabyte VAE or decode videos repeatedly.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class VideoEpisode:
    episode_index: int
    length: int
    video_path: Path
    video_key: str = ""
    start_time_seconds: float = 0.0
    fps: float = 30.0


def _open_video(path: Path) -> Any:
    """Open AV1 video with CUDA decode when available, with safe fallback."""

    try:
        return av.open(
            str(path),
            options={
                "threads": "1",
                "hwaccel": "cuda",
                "hwaccel_output_format": "cuda",
            },
        )
    except Exception:
        return av.open(str(path), options={"threads": "1"})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _existing_video_candidate(candidates: Iterable[Path]) -> Path | None:
    """Resolve a standard video path or one uniquely suffixed recovery file."""

    candidates = list(candidates)
    exact = next((candidate for candidate in candidates if candidate.is_file()), None)
    if exact is not None:
        return exact
    recovered: list[Path] = []
    for candidate in candidates:
        recovered.extend(path for path in candidate.parent.glob(candidate.name + ".*") if path.is_file())
    unique = sorted(set(recovered))
    return unique[0] if len(unique) == 1 else None


def _episode_metadata(root: Path, video_keys: Iterable[str]) -> dict[int, dict[str, Any]]:
    """Read episode metadata using pandas-free PyArrow for a small footprint."""

    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        available = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [column for column in ("episode_index", "length") if column in available]
        for video_key in video_keys:
            prefix = f"videos/{video_key}"
            for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                column = f"{prefix}/{suffix}"
                if column in available:
                    columns.append(column)
        if "episode_index" not in columns or "length" not in columns:
            raise ValueError(f"Episode metadata {path} lacks episode_index/length")
        rows.extend(
            pq.read_table(path, columns=columns).to_pylist()
        )
    return {int(row["episode_index"]): row for row in rows}


def resolve_video_episodes(root: str | Path, video_key: str | None = None) -> list[VideoEpisode]:
    """Resolve one video file for each LeRobot episode.

    ``video_key`` is the feature name without the ``videos/`` prefix. If it is
    omitted, ``global_image`` is preferred, then
    ``observation.images.top``. The resolver also supports v2 episode-local
    video names through a glob fallback.
    """

    root = Path(root)
    info = _read_json(root / "meta" / "info.json")
    feature_names = set(info.get("features", {}))
    if video_key is None:
        video_key = next(
            (
                candidate
                for candidate in ("global_image", "observation.images.top")
                if candidate in feature_names or f"observation.images.{candidate}" in feature_names
            ),
        )
    if video_key is None:
        raise ValueError("Could not infer video key; pass --video-key explicitly")
    metadata = _episode_metadata(root, [video_key])
    if not metadata:
        # v2 stores one episode parquet and one episode video per file rather
        # than a shared meta/episodes table.
        import pyarrow.parquet as pq

        fallback: list[VideoEpisode] = []
        for data_path in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
            episode_index = int(data_path.stem.split("_")[-1])
            length = int(pq.ParquetFile(data_path).metadata.num_rows)
            chunk = data_path.parent.name.split("-")[-1]
            candidates = [
                root / "videos" / f"chunk-{int(chunk):03d}" / video_key / f"episode_{episode_index:06d}.mp4",
                root / "videos" / video_key / f"chunk-{int(chunk):03d}" / f"episode_{episode_index:06d}.mp4",
                root / "videos" / f"chunk-{int(chunk):03d}" / video_key / f"episode_{episode_index:03d}.mp4",
            ]
            video_path = _existing_video_candidate(candidates)
            if video_path is None:
                raise FileNotFoundError(
                    f"Could not resolve v2 video for episode {episode_index}; checked {candidates}"
                )
            fallback.append(
                VideoEpisode(
                    episode_index,
                    length,
                    video_path,
                    video_key=video_key,
                    fps=float(info.get("fps", 30.0)),
                )
            )
        if fallback:
            return fallback
    episodes: list[VideoEpisode] = []
    for episode_index, row in sorted(metadata.items()):
        chunk_key = f"videos/{video_key}/chunk_index"
        file_key = f"videos/{video_key}/file_index"
        # Some metadata tables expose observation.images.top while the caller
        # passed the shorter key; accept the canonical alias transparently.
        if chunk_key not in row:
            canonical = "videos/observation.images.top"
            chunk_key, file_key = f"{canonical}/chunk_index", f"{canonical}/file_index"
        chunk = row.get(chunk_key)
        file_index = row.get(file_key)
        candidates: list[Path] = []
        if chunk is not None and file_index is not None:
            candidates.extend(
                [
                    root / "videos" / video_key / f"chunk-{int(chunk):03d}" / f"file-{int(file_index):03d}.mp4",
                    root / "videos" / f"chunk-{int(chunk):03d}" / video_key / f"episode_{episode_index:06d}.mp4",
                    root / "videos" / f"chunk-{int(chunk):03d}" / video_key / f"episode_{episode_index:03d}.mp4",
                ]
            )
        candidates.extend(
            [
                root / "videos" / video_key / f"chunk-000" / f"episode_{episode_index:06d}.mp4",
                root / "videos" / video_key / f"chunk-000" / f"episode_{episode_index:03d}.mp4",
            ]
        )
        video_path = _existing_video_candidate(candidates)
        if video_path is None:
            # v2 datasets can place the video key between chunk and filename.
            matches = sorted(root.glob(f"videos/**/{video_key}/**/*{episode_index:06d}*.mp4"))
            video_path = matches[0] if matches else None
        if video_path is None:
            raise FileNotFoundError(
                f"Could not resolve video for episode {episode_index}; checked {candidates}"
            )
        start_time = float(row.get(f"videos/{video_key}/from_timestamp") or 0.0)
        episodes.append(
            VideoEpisode(
                episode_index,
                int(row["length"]),
                video_path,
                video_key=video_key,
                start_time_seconds=start_time,
                fps=float(info.get("fps", 30.0)),
            )
        )
    if not episodes:
        raise ValueError(f"No episode metadata found below {root / 'meta/episodes'}")
    return episodes


def letterbox_frame(frame: np.ndarray, size: int) -> torch.Tensor:
    """Normalize an RGB frame to Wan's square input contract."""

    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame [H,W,3], got {frame.shape}")
    height, width = frame.shape[:2]
    scale = min(size / height, size / width)
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1.0
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    pad_h, pad_w = size - new_height, size - new_width
    return F.pad(tensor, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))


def _decode_target_frames(video_path: Path, target_indices: set[int]) -> Iterable[tuple[int, np.ndarray]]:
    container = _open_video(video_path)
    try:
        for index, frame in enumerate(container.decode(video=0)):
            if index in target_indices:
                yield index, frame.to_ndarray(format="rgb24")
            if index >= max(target_indices, default=-1):
                break
    finally:
        container.close()


def encode_episode_features(
    vae: Any,
    episode: VideoEpisode,
    *,
    size: int = 256,
    frame_stride: int = 1,
    batch_size: int = 8,
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Encode causal windows and return dense ``[T,48]`` features.

    For ``frame_stride > 1``, sparse VAE encodings are linearly interpolated
    over time. This makes a controlled low-cost preview possible while keeping
    the training data interface dense and frame-aligned.
    """

    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if episode.length < 1:
        raise ValueError("episode length must be positive")
    targets = set(range(0, episode.length, frame_stride))
    targets.add(episode.length - 1)
    indices = np.asarray(sorted(targets), dtype=np.int64)
    sparse_by_index: dict[int, np.ndarray] = {}
    needed_indices = {
        index
        for target_index in indices.tolist()
        for index in range(max(0, target_index - 4), target_index + 1)
    }
    needed_frames: dict[int, np.ndarray] = {}
    next_target = iter(indices.tolist())
    target = next(next_target, None)
    pending_indices: list[int] = []
    pending_clips: list[torch.Tensor] = []

    def flush_pending() -> None:
        if not pending_clips:
            return
        with torch.inference_mode():
            clips = [clip.to(device=device, dtype=vae.dtype) for clip in pending_clips]
            if hasattr(vae, "model") and hasattr(vae, "scale"):
                # The public Wan wrapper loops over its list argument. The
                # underlying encoder is batch-safe and materially faster.
                stacked = torch.stack(clips, dim=0)
                with torch.amp.autocast(
                    device_type=torch.device(device).type,
                    dtype=vae.dtype,
                    enabled=torch.device(device).type == "cuda",
                ):
                    encoded = vae.model.encode(stacked, vae.scale).float()
                latents = list(encoded.unbind(dim=0))
            else:
                latents = vae.encode(clips)
        if latents is None or len(latents) != len(pending_indices):
            raise RuntimeError("Wan VAE returned an invalid batch result")
        for frame_index, latent in zip(pending_indices, latents):
            pooled = latent.float().mean(dim=(1, 2, 3)).detach().cpu().numpy()
            if pooled.shape != (48,):
                raise RuntimeError(f"Unexpected pooled Wan latent shape: {pooled.shape}")
            sparse_by_index[frame_index] = pooled
        pending_indices.clear()
        pending_clips.clear()
    # Decode only this episode's time range. Wrist streams commonly pack
    # several episodes into one MP4, unlike the global stream.
    container = _open_video(episode.video_path)
    try:
        stream = container.streams.video[0]
        if episode.start_time_seconds > 0 and stream.time_base is not None:
            seek_pts = int(episode.start_time_seconds / float(stream.time_base))
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        local_index = 0
        started = episode.start_time_seconds <= 0
        for frame in container.decode(stream):
            if not started:
                frame_time = float(frame.pts * stream.time_base) if frame.pts is not None else None
                if frame_time is not None and frame_time + 0.5 / episode.fps < episode.start_time_seconds:
                    continue
                started = True
            index = local_index
            local_index += 1
            if index >= episode.length:
                break
            if index in needed_indices:
                needed_frames[index] = frame.to_ndarray(format="rgb24")
            if target is None or index != target:
                if index >= episode.length - 1:
                    break
                continue
            window = [
                needed_frames[frame_index]
                for frame_index in range(max(0, index - 4), index + 1)
            ]
            while len(window) < 5:
                window.insert(0, window[0])
            pending_indices.append(index)
            pending_clips.append(torch.stack([letterbox_frame(item, size) for item in window], dim=1))
            if len(pending_clips) >= batch_size:
                flush_pending()
            target = next(next_target, None)
            if target is not None:
                oldest_needed = max(0, target - 4)
                needed_frames = {
                    frame_index: value
                    for frame_index, value in needed_frames.items()
                    if frame_index >= oldest_needed
                }
            if target is None:
                break
        flush_pending()
    finally:
        container.close()
    sparse_features = np.asarray([sparse_by_index[int(index)] for index in indices], dtype=np.float32)
    if sparse_features.shape != (len(indices), 48):
        raise RuntimeError(
            f"Unexpected Wan latent features {sparse_features.shape}; expected {(len(indices), 48)}"
        )
    dense = np.stack(
        [np.interp(np.arange(episode.length), indices, sparse_features[:, dim]) for dim in range(48)],
        axis=1,
    ).astype(np.float32)
    return dense, indices
