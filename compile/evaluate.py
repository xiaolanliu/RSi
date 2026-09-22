"""Evaluate a CompILE checkpoint on one or more LeRobot roots.

The evaluator is intentionally episode-wise: padding never contributes to a
metric, and source roots remain visible when episode indices restart at zero.
It writes both a machine-readable JSON report and a flat CSV for quick
comparison of training runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import CompILEConfig
from .data import LeRobotEpisodeDataset, collate_episodes
from .metrics import action_mae
from .model import CompILE
from .subtasks import monotonic_boundary_projection


def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _episode_metrics(
    model: CompILE,
    episode: Any,
    device: torch.device,
) -> dict[str, Any]:
    batch = _move_batch(collate_episodes([episode]), device)
    output = model(batch, sample_latents=False)
    boundaries = monotonic_boundary_projection(output.boundary_probs[0], episode.length)
    segments = []
    previous = 1
    for boundary in boundaries:
        start = max(0, previous - 1)
        end = min(episode.length, max(start, boundary - 1))
        segments.append(
            {
                "index": len(segments),
                "start_frame_offset": start,
                "end_frame_exclusive_offset": end,
                "length": max(0, end - start),
                "code_id": int(output.code_probs[0, len(segments)].argmax().item()),
                "code_confidence": float(output.code_probs[0, len(segments)].max().item()),
            }
        )
        previous = boundary
    nonempty = [segment for segment in segments if segment["length"] > 0]
    return {
        "dataset_name": episode.metadata.get("dataset_name", Path(episode.metadata["source_root"]).name),
        "source_root": episode.metadata["source_root"],
        "episode_index": int(episode.episode_index),
        "length": int(episode.length),
        "loss": float(output.loss.cpu()),
        "reconstruction_nll": float(output.reconstruction_loss.cpu()),
        "kl_z": float(output.kl_z.cpu()),
        "kl_b": float(output.kl_b.cpu()),
        "segment_balance": float(output.segment_balance.cpu()),
        "action_mae": action_mae(
            output,
            batch["actions"],
            batch["valid_mask"],
            batch.get("action_valid_mask"),
        ),
        "boundaries_one_based": boundaries,
        "phase_count": len(nonempty),
        "phase_lengths": [segment["length"] for segment in nonempty],
        "segments": segments,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--visual-cache-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.visual_cache_dirs is not None and len(args.visual_cache_dirs) != len(args.data_roots):
        raise ValueError("visual-cache-dirs must match data-roots")
    cache_dirs = args.visual_cache_dirs or [None] * len(args.data_roots)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = CompILEConfig(**checkpoint["config"])
    model = CompILE(config).to(_device(args.device))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for root, cache_dir in zip(args.data_roots, cache_dirs):
            dataset = LeRobotEpisodeDataset(
                root,
                max_episodes=args.max_episodes,
                visual_cache_dir=cache_dir,
                temporal_stride=config.temporal_stride,
                derive_actions_from_state=True,
            )
            if dataset.state_dim != config.state_dim or dataset.action_dim != config.action_dim:
                raise ValueError(f"Dimensions for {root} do not match the checkpoint")
            if dataset.visual_dim != config.visual_dim:
                raise ValueError(f"Visual dimension for {root} does not match the checkpoint")
            for episode in dataset.episodes:
                records.append(_episode_metrics(model, episode, _device(args.device)))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["dataset_name"]].append(record)

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        numeric = ["loss", "reconstruction_nll", "kl_z", "kl_b", "segment_balance", "action_mae"]
        return {
            "episodes": len(rows),
            "frames": int(sum(row["length"] for row in rows)),
            **{f"mean_{key}": float(np.mean([row[key] for row in rows])) for key in numeric},
            "phase_count_histogram": {
                str(count): sum(row["phase_count"] == count for row in rows)
                for count in sorted({row["phase_count"] for row in rows})
            },
            "mean_phase_lengths": float(
                np.mean([length for row in rows for length in row["phase_lengths"]])
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": checkpoint["config"],
        "episodes": len(records),
        "aggregate": aggregate(records),
        "by_dataset": {name: aggregate(rows) for name, rows in sorted(grouped.items())},
        "records": records,
    }
    (args.output_dir / "evaluation_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = [
        "dataset_name",
        "episode_index",
        "length",
        "loss",
        "reconstruction_nll",
        "kl_z",
        "kl_b",
        "segment_balance",
        "action_mae",
        "phase_count",
        "boundaries_one_based",
        "phase_lengths",
    ]
    with (args.output_dir / "evaluation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})
    print(json.dumps({"episodes": len(records), "aggregate": report["aggregate"], "by_dataset": report["by_dataset"]}, indent=2))


if __name__ == "__main__":
    main()
