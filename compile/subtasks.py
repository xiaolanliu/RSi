"""Infer and visualize CompILE subtask structures.

The output is intentionally an audit artifact rather than a new training
objective. It stores monotonic discrete boundaries projected from the learned
boundary posteriors, per-segment latent code probabilities, soft masks, and
episode provenance. The accompanying plots make it possible to inspect whether
visual/state evidence, boundaries and code assignments agree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CompILEConfig
from .data import LeRobotEpisodeDataset, collate_episodes
from .model import CompILE


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def monotonic_boundary_projection(boundary_probs: torch.Tensor, length: int) -> list[int]:
    """Project independent posterior argmaxes onto nondecreasing boundaries.

    CompILE's relaxed masks are valid even when independent posterior modes
    cross. At test time the paper switches to consecutive discrete masks, so a
    monotonic projection is required for a usable segment list. This dynamic
    program maximizes the sum of log posterior probabilities while allowing
    empty segments and the terminal value ``length + 1``.
    """

    if boundary_probs.ndim != 2:
        raise ValueError("boundary_probs must have shape [segments, length + 1]")
    internal = boundary_probs.shape[0] - 1
    if internal <= 0:
        return [length + 1]
    # log_probs column 0 is boundary slot 1, i.e. b=2 because b=1 is
    # forbidden. The final candidate column is slot T, i.e. b=T+1.
    positions = torch.arange(2, length + 2, device=boundary_probs.device)
    log_probs = boundary_probs[:-1, 1 : length + 1].clamp_min(1e-8).log()
    scores = log_probs[0]
    backpointers: list[torch.Tensor] = []
    for segment_index in range(1, internal):
        best_scores, best_indices = torch.cummax(scores, dim=0)
        scores = log_probs[segment_index] + best_scores
        backpointers.append(best_indices)
    last = int(scores.argmax().item())
    selected = [last]
    for pointer in reversed(backpointers):
        last = int(pointer[last].item())
        selected.append(last)
    selected.reverse()
    return [int(positions[index].item()) for index in selected] + [length + 1]


def _segment_records(
    episode_index: int,
    frame_indices: torch.Tensor,
    boundaries: list[int],
    code_ids: list[int],
    code_confidence: list[float],
) -> list[dict[str, Any]]:
    length = int(frame_indices.numel())
    records = []
    previous_boundary = 1
    for segment_index, boundary in enumerate(boundaries):
        start = max(0, previous_boundary - 1)
        end_exclusive = min(length, max(start, boundary - 1))
        if end_exclusive > start:
            start_frame = int(frame_indices[start].item())
            end_frame = int(frame_indices[end_exclusive - 1].item())
        else:
            start_frame = end_frame = int(frame_indices[min(start, length - 1)].item())
        records.append(
            {
                "segment_index": segment_index,
                "episode_index": episode_index,
                "boundary_start_one_based": previous_boundary,
                "boundary_end_one_based": boundary,
                "start_frame": start_frame,
                "end_frame_inclusive": end_frame,
                "start_frame_offset": start,
                "end_frame_exclusive_offset": end_exclusive,
                "length": max(0, end_exclusive - start),
                "code_id": code_ids[segment_index],
                "code_confidence": code_confidence[segment_index],
            }
        )
        previous_boundary = boundary
    return records


def _plot_episode(
    output_path: Path,
    episode_index: int,
    boundary_probs: np.ndarray,
    masks: np.ndarray,
    code_probs: np.ndarray,
    boundaries: list[int],
) -> None:
    valid_length = masks.shape[1]
    time_axis = np.arange(valid_length)
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.0, 1.0]}
    )
    internal = boundary_probs[:-1, : valid_length + 1]
    axes[0].imshow(
        internal,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, valid_length, 1, max(1, internal.shape[0])],
        cmap="magma",
    )
    axes[0].set_ylabel("boundary pass")
    axes[0].set_title(f"Episode {episode_index}: learned boundary posterior")
    for boundary in boundaries[:-1]:
        axes[0].axvline(boundary - 1, color="cyan", linewidth=1.0, alpha=0.9)

    axes[1].imshow(
        masks,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, valid_length, 0, max(1, masks.shape[0])],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_ylabel("soft mask")
    for boundary in boundaries[:-1]:
        axes[1].axvline(boundary - 1, color="white", linewidth=1.0, alpha=0.9)

    # Broadcast each segment's code posterior through its soft temporal mask
    # to show the actual per-frame latent mixture.
    code_timeline = np.einsum("mt,mk->tk", masks, code_probs)
    axes[2].imshow(
        code_timeline.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, valid_length, 0, max(1, code_probs.shape[1])],
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[2].set_ylabel("latent code")
    axes[2].set_xlabel("frame offset")
    axes[2].set_xticks(np.linspace(0, valid_length, min(8, valid_length + 1), dtype=int))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_overview(output_path: Path, boundaries: list[list[int]], code_usage: np.ndarray) -> None:
    internal = np.asarray([row[:-1] for row in boundaries], dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if internal.size:
        axes[0].boxplot(
            [internal[:, column] for column in range(internal.shape[1])],
            tick_labels=[f"b{index + 1}" for index in range(internal.shape[1])],
        )
    axes[0].set_title("Projected internal boundary distribution")
    axes[0].set_ylabel("one-based boundary frame")
    used = np.flatnonzero(code_usage > 0)
    axes[1].bar(used, code_usage[used], color="#2563eb")
    axes[1].set_title("Mean posterior latent-code usage")
    axes[1].set_xlabel("code id")
    axes[1].set_ylabel("probability")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--visual-cache-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--plot-limit",
        type=int,
        default=None,
        help="Plot only the first N episodes while exporting every episode to JSON/NPZ.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = CompILEConfig(**checkpoint["config"])
    dataset = LeRobotEpisodeDataset(
        args.data_root,
        max_episodes=args.max_episodes,
        visual_cache_dir=args.visual_cache_dir,
        temporal_stride=config.temporal_stride,
        derive_actions_from_state=True,
    )
    if dataset.visual_dim != config.visual_dim:
        raise ValueError(
            f"Checkpoint expects visual_dim={config.visual_dim}, dataset cache provides {dataset.visual_dim}"
        )
    model = CompILE(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "visualizations"
    plot_dir.mkdir(exist_ok=True)
    structure_path = args.output_dir / "subtasks.jsonl"
    all_boundaries: list[list[int]] = []
    code_usage_sum = torch.zeros(config.num_codes)
    with structure_path.open("w", encoding="utf-8") as structure_file, torch.inference_mode():
        for episode_number, episode in enumerate(dataset.episodes):
            batch = collate_episodes([episode])
            output = model(_move_batch(batch, device), sample_latents=False)
            length = episode.length
            boundaries = monotonic_boundary_projection(output.boundary_probs[0], length)
            codes = output.code_probs[0].argmax(dim=-1).cpu().tolist()
            confidence = output.code_probs[0].max(dim=-1).values.cpu().tolist()
            segments = _segment_records(
                episode.episode_index,
                episode.frame_indices,
                boundaries,
                [int(code) for code in codes],
                [float(value) for value in confidence],
            )
            boundary_array = output.boundary_probs[0].cpu().numpy().astype(np.float32)
            mask_array = output.segment_masks[0].cpu().numpy().astype(np.float32)
            code_array = output.code_probs[0].cpu().numpy().astype(np.float32)
            np.savez_compressed(
                args.output_dir / f"episode_{episode.episode_index:06d}.npz",
                boundary_probs=boundary_array,
                segment_masks=mask_array,
                code_probs=code_array,
                boundaries_one_based=np.asarray(boundaries, dtype=np.int64),
            )
            if args.plot_limit is None or episode_number < args.plot_limit:
                _plot_episode(
                    plot_dir / f"episode_{episode.episode_index:06d}_timeline.png",
                    episode.episode_index,
                    boundary_array,
                    mask_array,
                    code_array,
                    boundaries,
                )
            code_usage_sum += output.code_probs[0].mean(dim=0).cpu()
            all_boundaries.append(boundaries)
            record = {
                "episode_index": episode.episode_index,
                "length": length,
                "boundary_positions_one_based": boundaries,
                "segments": segments,
                "source_root": episode.metadata.get("source_root"),
                "state_columns": episode.metadata.get("state_columns"),
                "action_columns": episode.metadata.get("action_columns"),
                "visual_cache_dir": episode.metadata.get("visual_cache_dir"),
            }
            structure_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    code_usage = (code_usage_sum / max(len(dataset), 1)).numpy()
    _plot_overview(args.output_dir / "visualizations" / "overview.png", all_boundaries, code_usage)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_root": str(args.data_root.resolve()),
        "visual_cache_dir": str(args.visual_cache_dir.resolve()) if args.visual_cache_dir else None,
        "episodes": len(dataset),
        "visual_dim": dataset.visual_dim,
        "temporal_stride": config.temporal_stride,
        "max_segments": config.max_segments,
        "num_codes": config.num_codes,
        "plot_limit": args.plot_limit,
        "code_usage": code_usage.tolist(),
        "artifacts": {
            "structure_jsonl": structure_path.name,
            "episode_arrays": "episode_*.npz",
            "visualizations": "visualizations/",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
