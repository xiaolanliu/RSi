"""Render actual episode frames together with an inferred subtask split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .vision import resolve_video_episodes


PHASE_COLORS = ["#2563eb", "#16a34a", "#d97706", "#9333ea", "#dc2626", "#0891b2"]


def _load_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record["episode_index"])] = record
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _decode_frames(video_path: Path, frame_indices: set[int]) -> dict[int, np.ndarray]:
    wanted = sorted(int(index) for index in frame_indices)
    if not wanted:
        return {}
    frames: dict[int, np.ndarray] = {}
    container = av.open(str(video_path), options={"threads": "1"})
    try:
        for index, frame in enumerate(container.decode(video=0)):
            if index in frame_indices:
                frames[index] = frame.to_ndarray(format="rgb24")
            if index >= wanted[-1]:
                break
    finally:
        container.close()
    missing = [index for index in wanted if index not in frames]
    if missing:
        raise RuntimeError(f"Could not decode frames {missing[:8]} from {video_path}")
    return frames


def _frame_points(segment: dict[str, Any]) -> list[int]:
    # ``start_frame_offset`` is the index in a temporally downsampled model
    # sequence. ``start_frame``/``end_frame_inclusive`` are source-video frame
    # indices and must be used when decoding the actual RGB episode.
    start = int(segment["start_frame"])
    end = int(segment["end_frame_inclusive"])
    if end < start:
        return []
    middle = start + (end - start) // 2
    return [start, middle, end]


def _draw_timeline(ax: Any, record: dict[str, Any]) -> None:
    length = max(
        int(record["length"]),
        max((int(segment["end_frame_inclusive"]) + 1 for segment in record["segments"]), default=1),
    )
    segments = [segment for segment in record["segments"] if int(segment["length"]) > 0]
    ax.set_xlim(0, length)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("episode frame index (0-based)")
    ax.set_title(
        f"Episode {record['episode_index']} phase timeline | total {length} frames | "
        f"{len(segments)} non-empty phases"
    )
    for segment in record["segments"]:
        start = int(segment["start_frame"])
        width = int(segment["end_frame_inclusive"]) - start + 1
        index = int(segment["segment_index"])
        if width <= 0:
            continue
        color = PHASE_COLORS[index % len(PHASE_COLORS)]
        ax.add_patch(Rectangle((start, 0.15), width, 0.7, color=color, alpha=0.82))
        label = (
            f"phase {index}\n"
            f"{start}-{start + width - 1}\n"
            f"code {segment['code_id']}"
        )
        center = start + width / 2.0
        ax.text(
            center,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=9 if width >= length * 0.12 else 7,
            color="white",
            fontweight="bold",
            clip_on=True,
        )
        if start > 0:
            ax.axvline(start, color="#111827", linewidth=1.2, alpha=0.85)
    empty = [
        int(segment["segment_index"])
        for segment in record["segments"]
        if int(segment["length"]) == 0
    ]
    if empty:
        ax.text(
            0.995,
            1.08,
            "empty slots: " + ", ".join(f"phase {index}" for index in empty),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#6b7280",
        )


def render_phase_frames(
    record: dict[str, Any],
    video_path: Path,
    output_path: Path,
) -> None:
    segments = [segment for segment in record["segments"] if int(segment["length"]) > 0]
    selected = {
        frame
        for segment in segments
        for frame in _frame_points(segment)
    }
    frames = _decode_frames(video_path, selected)
    rows = max(len(segments), 1)
    fig = plt.figure(figsize=(15, 3.2 + rows * 3.1), facecolor="white")
    grid = fig.add_gridspec(rows + 1, 3, height_ratios=[0.9] + [2.2] * rows, hspace=0.32, wspace=0.08)
    timeline = fig.add_subplot(grid[0, :])
    _draw_timeline(timeline, record)

    for row, segment in enumerate(segments, start=1):
        points = _frame_points(segment)
        color = PHASE_COLORS[int(segment["segment_index"]) % len(PHASE_COLORS)]
        for column, (point, label) in enumerate(zip(points, ("start", "middle", "end"))):
            ax = fig.add_subplot(grid[row, column])
            ax.imshow(frames[point])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(3.0)
            ax.set_title(f"{label}: frame {point}", fontsize=10, pad=7)
        segment_label = (
            f"phase {segment['segment_index']} | frames "
            f"{segment['start_frame']}–{segment['end_frame_inclusive']} "
            f"({segment['length']} frames) | latent code {segment['code_id']}"
        )
        fig.text(
            0.015,
            1.0 - (row + 0.55) / (rows + 1.18),
            segment_label,
            ha="left",
            va="center",
            fontsize=10,
            color=color,
            fontweight="bold",
        )
    fig.suptitle(
        "CompILE inferred subtasks aligned with real episode frames\n"
            f"source: {video_path.name}",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subtasks-jsonl", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episode", type=int)
    selection.add_argument("--episodes", type=int, nargs="+")
    selection.add_argument("--all", action="store_true", dest="render_all")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--video-key", default=None)
    args = parser.parse_args()
    if args.episode is not None and args.output is None:
        parser.error("--episode requires --output")
    if (args.episodes is not None or args.render_all) and args.output_dir is None:
        parser.error("--episodes/--all requires --output-dir")
    return args


def main() -> None:
    args = _parse_args()
    records = _load_records(args.subtasks_jsonl)
    video_episodes = {episode.episode_index: episode for episode in resolve_video_episodes(args.data_root, args.video_key)}
    requested = (
        sorted(records)
        if args.render_all
        else args.episodes
        if args.episodes is not None
        else [args.episode]
    )
    manifests = []
    for episode_index in requested:
        if episode_index not in records:
            raise KeyError(f"Episode {episode_index} is not present in {args.subtasks_jsonl}")
        if episode_index not in video_episodes:
            raise KeyError(f"Episode {episode_index} has no resolved video")
        output = (
            args.output
            if args.output is not None
            else args.output_dir / f"episode_{episode_index:06d}_phase_frames.png"
        )
        video_path = video_episodes[episode_index].video_path
        render_phase_frames(records[episode_index], video_path, output)
        manifests.append(
            {
                "episode_index": episode_index,
                "video": str(video_path.resolve()),
                "output": str(output.resolve()),
                "frame_policy": "start/middle/end for each non-empty inferred segment",
            }
        )
        print(json.dumps(manifests[-1]), flush=True)
    manifest = {
        "subtasks_jsonl": str(args.subtasks_jsonl.resolve()),
        "data_root": str(args.data_root.resolve()),
        "episodes": manifests,
    }
    manifest_path = (
        args.output.with_suffix(".json")
        if args.output is not None
        else args.output_dir / "phase_frame_review_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
