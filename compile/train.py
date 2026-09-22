"""Command-line training entry point for the initial CompILE baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .config import CompILEConfig
from .data import LeRobotEpisodeDataset, MultiRootEpisodeDataset, collate_episodes
from .metrics import action_mae, boundary_positions, code_usage
from .model import CompILE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--data-roots", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=None,
        help="JSON with roots[].data_root and roots[].visual_cache_dir entries.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--num-codes", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--state-hidden-dim", type=int, default=64)
    parser.add_argument("--sequence-encoder", choices=["tcn", "transformer"], default="tcn")
    parser.add_argument("--fusion-mode", choices=["gate", "film", "attention"], default="gate",
                        help="How visual features fuse with state features in the recognition encoder.")
    parser.add_argument(
        "--causal-sequence-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use causal TCN convolutions / causal transformer attention so the "
             "recognition readout at frame t only sees frames <= t (matches the "
             "causal first-boundary KL prior). Disable only together with a "
             "per-segment boundary prior.",
    )
    parser.add_argument(
        "--policy-uses-fused-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Condition the policy bank on the fused recognition features "
             "(zero-initialised projection) so visual context can modulate actions.",
    )
    parser.add_argument("--visual-input-scale", type=float, default=4.0)
    parser.add_argument("--tcn-kernel-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--boundary-kl-weight", type=float, default=0.05)
    parser.add_argument("--segment-balance-weight", type=float, default=0.02)
    parser.add_argument("--segment-balance-min-ratio", type=float, default=0.5)
    parser.add_argument("--segment-balance-max-ratio", type=float, default=1.8)
    parser.add_argument("--kl-warmup-epochs", type=int, default=5)
    parser.add_argument("--poisson-rate", type=float, default=None)
    parser.add_argument(
        "--adaptive-poisson-rate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set the boundary Poisson rate per episode to length/max_segments.",
    )
    parser.add_argument("--gumbel-temperature", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--visual-cache-dir", type=Path, default=None)
    parser.add_argument("--visual-cache-dirs", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--derive-actions-from-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use state[t+1]-state[t] as the action target and mask episode termini.",
    )
    parser.add_argument("--state-columns", nargs="+", default=None)
    parser.add_argument("--action-columns", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--report-epochs",
        type=int,
        nargs="*",
        default=[100, 500, 1000],
        help="Save checkpoints and metric snapshots at these epochs.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Load model weights from this checkpoint before training (continue training).",
    )
    parser.add_argument(
        "--epoch-offset",
        type=int,
        default=0,
        help="Added to the local epoch index so report-epochs/history use global epoch numbers.",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=100,
        help="Run validation every N epochs in addition to report epochs.",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _stratified_split_indices(dataset: LeRobotEpisodeDataset | MultiRootEpisodeDataset, seed: int) -> tuple[list[int], list[int]]:
    """Split complete episodes while preserving every source root in validation."""

    if len(dataset) < 5:
        return list(range(len(dataset))), []
    groups: dict[str, list[int]] = {}
    for index, episode in enumerate(dataset.episodes):
        source_root = str(episode.metadata.get("source_root", ""))
        groups.setdefault(source_root, []).append(index)
    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for indices in groups.values():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        # Keep at least one episode from every source in training. Tiny
        # smoke subsets may contain only one episode per source.
        validation_size = min(max(1, len(shuffled) // 5), max(len(shuffled) - 1, 0))
        validation_indices.extend(shuffled[:validation_size])
        train_indices.extend(shuffled[validation_size:])
    return sorted(train_indices), sorted(validation_indices)


def _split_records(dataset: LeRobotEpisodeDataset | MultiRootEpisodeDataset, indices: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "dataset_name": episode.metadata.get("dataset_name", Path(str(episode.metadata.get("source_root", ""))).name),
            "source_root": episode.metadata.get("source_root"),
            "episode_index": episode.episode_index,
            "length": episode.length,
        }
        for index in indices
        for episode in [dataset.episodes[index]]
    ]


def _dataset_normalization_stats(
    dataset: LeRobotEpisodeDataset | MultiRootEpisodeDataset,
    indices: list[int] | None = None,
) -> dict[str, tuple[float, ...]]:
    """Compute fixed global statistics from state-derived training tensors."""

    episodes = (
        [dataset.episodes[index] for index in indices]
        if indices is not None
        else dataset.episodes
    )
    states = torch.cat([episode.states for episode in episodes], dim=0).float()
    actions = torch.cat(
        [
            episode.actions[episode.action_valid_mask.bool()]
            if episode.action_valid_mask is not None
            else episode.actions
            for episode in episodes
        ],
        dim=0,
    ).float()
    state_mean = states.mean(dim=0)
    state_std = states.std(dim=0, unbiased=False).clamp_min(1e-3)
    action_mean = actions.mean(dim=0)
    action_std = actions.std(dim=0, unbiased=False).clamp_min(1e-3)
    return {
        "state_mean": tuple(float(value) for value in state_mean),
        "state_std": tuple(float(value) for value in state_std),
        "action_mean": tuple(float(value) for value in action_mean),
        "action_std": tuple(float(value) for value in action_std),
    }


def _model_config(model: CompILE | DistributedDataParallel) -> CompILEConfig:
    return model.module.config if isinstance(model, DistributedDataParallel) else model.config


def _run_epoch(model: CompILE | DistributedDataParallel, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "reconstruction_loss": 0.0,
        "kl_z": 0.0,
        "kl_b": 0.0,
        "weighted_kl_z": 0.0,
        "weighted_kl_b": 0.0,
        "segment_balance": 0.0,
        "action_mae": 0.0,
    }
    count = 0
    num_batches = 0
    for batch in loader:
        num_batches += 1
        batch = _move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        # Validation runs under inference_mode: with the causal-attention
        # fusion the [B, 2T, 2T] attention activations dominate memory and
        # must not be retained when no backward pass will free them.
        with torch.inference_mode(not training):
            output = model(batch, sample_latents=training)
        if training:
            output.loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        batch_size = batch["states"].shape[0]
        count += batch_size
        totals["loss"] += float(output.loss.detach()) * batch_size
        totals["reconstruction_loss"] += float(output.reconstruction_loss.detach()) * batch_size
        totals["kl_z"] += float(output.kl_z.detach()) * batch_size
        totals["kl_b"] += float(output.kl_b.detach()) * batch_size
        config = _model_config(model)
        totals["weighted_kl_z"] += float(output.kl_z.detach()) * config.kl_weight * batch_size
        totals["weighted_kl_b"] += float(output.kl_b.detach()) * config.kl_weight * config.boundary_kl_weight * batch_size
        totals["segment_balance"] += float(output.segment_balance.detach()) * batch_size
        totals["action_mae"] += action_mae(
            output,
            batch["actions"],
            batch["valid_mask"],
            batch.get("action_valid_mask"),
        ) * batch_size
    if dist.is_available() and dist.is_initialized():
        keys = list(totals)
        values = torch.tensor(
            [*[totals[key] for key in keys], count, num_batches],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        totals = {key: float(values[index]) for index, key in enumerate(keys)}
        count = int(values[-2])
        num_batches = int(values[-1])
    result = {key: value / max(count, 1) for key, value in totals.items()}
    result["num_batches"] = float(num_batches)
    return result


def main() -> None:
    args = _parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        device = _device(args.device)
    _set_seed(args.seed + rank)
    manifest_cache_dirs = None
    if args.dataset_manifest is not None:
        if args.data_root is not None or args.data_roots is not None or args.visual_cache_dir is not None or args.visual_cache_dirs is not None:
            raise ValueError("dataset-manifest cannot be combined with explicit data/cache roots")
        manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
        entries = manifest.get("roots", [])
        if not entries:
            raise ValueError("dataset-manifest has no roots")
        roots = [Path(entry["data_root"]) for entry in entries]
        manifest_cache_dirs = [Path(entry["visual_cache_dir"]) for entry in entries]
    else:
        roots = args.data_roots or [args.data_root]
    if any(root is None for root in roots):
        raise ValueError("Provide --data-root, --data-roots, or --dataset-manifest")
    if args.data_roots is not None and args.visual_cache_dir is not None:
        raise ValueError("Use --visual-cache-dirs with --data-roots, not --visual-cache-dir")
    if args.data_roots is not None or args.dataset_manifest is not None:
        cache_dirs = manifest_cache_dirs or args.visual_cache_dirs
        if cache_dirs is None:
            raise ValueError("--visual-cache-dirs is required with --data-roots")
        dataset = MultiRootEpisodeDataset(
            roots,
            visual_cache_dirs=cache_dirs,
            max_episodes_per_root=args.max_episodes,
            max_length=args.max_length,
            state_columns=args.state_columns,
            action_columns=args.action_columns,
            temporal_stride=args.temporal_stride,
            derive_actions_from_state=args.derive_actions_from_state,
        )
    else:
        dataset = LeRobotEpisodeDataset(
            roots[0],
            max_episodes=args.max_episodes,
            max_length=args.max_length,
            state_columns=args.state_columns,
            action_columns=args.action_columns,
            visual_cache_dir=args.visual_cache_dir,
            temporal_stride=args.temporal_stride,
            derive_actions_from_state=args.derive_actions_from_state,
        )
    train_indices, validation_indices = _stratified_split_indices(dataset, args.seed)
    normalization_stats = _dataset_normalization_stats(dataset, train_indices)
    train_set = Subset(dataset, train_indices)
    validation_set = Subset(dataset, validation_indices) if validation_indices else None
    train_sampler = (
        DistributedSampler(train_set, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.seed)
        if distributed else None
    )
    validation_sampler = (
        DistributedSampler(validation_set, num_replicas=dist.get_world_size(), rank=rank, shuffle=False)
        if distributed and validation_set is not None else None
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_episodes,
    )
    validation_loader = (
        DataLoader(
            validation_set,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=validation_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_episodes,
        )
        if validation_set is not None
        else None
    )
    if args.poisson_rate is None:
        mean_length = sum(episode.length for episode in dataset.episodes) / max(len(dataset), 1)
        poisson_rate = max(3.0, mean_length / max(args.max_segments, 1))
    else:
        poisson_rate = args.poisson_rate
    config = CompILEConfig(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        max_segments=args.max_segments,
        num_codes=args.num_codes,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        state_hidden_dim=args.state_hidden_dim,
        visual_dim=dataset.visual_dim,
        kl_weight=args.kl_weight,
        boundary_kl_weight=args.boundary_kl_weight,
        segment_balance_weight=args.segment_balance_weight,
        segment_balance_min_ratio=args.segment_balance_min_ratio,
        segment_balance_max_ratio=args.segment_balance_max_ratio,
        gumbel_temperature=args.gumbel_temperature,
        sequence_encoder=args.sequence_encoder,
        causal_sequence_encoder=args.causal_sequence_encoder,
        policy_uses_fused_features=args.policy_uses_fused_features,
        fusion_mode=args.fusion_mode,
        visual_input_scale=args.visual_input_scale,
        tcn_kernel_size=args.tcn_kernel_size,
        poisson_rate=poisson_rate,
        adaptive_poisson_rate=args.adaptive_poisson_rate,
        temporal_stride=args.temporal_stride,
        **normalization_stats,
    )
    base_model = CompILE(config).to(device)
    if distributed:
        # The selected fusion branch leaves the other fusion modules unused in
        # forward, so their parameters never receive gradients.
        model = DistributedDataParallel(
            base_model, device_ids=[device.index], find_unused_parameters=True
        )
    else:
        model = base_model
    if args.resume_checkpoint is not None:
        resume = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
        base_model.load_state_dict(resume["model"], strict=True)
        if rank == 0:
            print(json.dumps({"resumed_from": str(args.resume_checkpoint)}))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    split = {
        "seed": args.seed,
        "strategy": "stratified_by_source_root",
        "train": _split_records(dataset, train_indices),
        "validation": _split_records(dataset, validation_indices),
    }
    if rank == 0:
        (args.output_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    history: list[dict[str, Any]] = []
    report_epochs = set(args.report_epochs)
    for local_epoch in range(1, args.epochs + 1):
        epoch = args.epoch_offset + local_epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        active_config = base_model.config
        if args.kl_warmup_epochs > 0:
            active_config.boundary_kl_weight = args.boundary_kl_weight * min(
                1.0, local_epoch / args.kl_warmup_epochs
            )
        else:
            active_config.boundary_kl_weight = args.boundary_kl_weight
        train_metrics = _run_epoch(model, train_loader, optimizer, device)
        row: dict[str, Any] = {
            "epoch": epoch,
            "boundary_kl_weight": active_config.boundary_kl_weight,
            "poisson_rate": active_config.poisson_rate,
            "adaptive_poisson_rate": active_config.adaptive_poisson_rate,
            "train": train_metrics,
        }
        if validation_loader is not None and (
            epoch % max(args.validation_interval, 1) == 0 or epoch in report_epochs
        ):
            row["validation"] = _run_epoch(model, validation_loader, None, device)
        history.append(row)
        if rank == 0:
            print(json.dumps(row, sort_keys=True))
        if rank == 0 and epoch in report_epochs:
            torch.save(
                {"model": base_model.state_dict(), "config": config.__dict__, "history": history},
                args.output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
            )
            (args.output_dir / f"epoch_{epoch:04d}_metrics.json").write_text(
                json.dumps(row, indent=2) + "\n", encoding="utf-8"
            )
        if rank == 0 and (epoch % 10 == 0 or epoch in report_epochs):
            (args.output_dir / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
    # Store the requested final objective weight in the checkpoint even when a
    # short smoke run ends before the warm-up reaches its target.
    base_model.config.boundary_kl_weight = args.boundary_kl_weight
    if rank == 0:
        torch.save({"model": base_model.state_dict(), "config": config.__dict__, "history": history}, args.output_dir / "checkpoint.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if rank == 0:
        with torch.no_grad():
            sample_batch = _move_batch(next(iter(train_loader)), device)
            sample_output = base_model(sample_batch, sample_latents=False)
            summary = {
                "device": str(device),
                "distributed_world_size": dist.get_world_size() if distributed else 1,
                "batch_size_per_rank": args.batch_size,
                "num_episodes": len(dataset),
                "train_episodes": len(train_indices),
                "validation_episodes": len(validation_indices),
                "state_dim": dataset.state_dim,
                "action_dim": dataset.action_dim,
                "data_roots": [str(root.resolve()) for root in roots],
                "temporal_stride": args.temporal_stride,
                "derive_actions_from_state": args.derive_actions_from_state,
                "adaptive_poisson_rate": args.adaptive_poisson_rate,
                "normalization_stats": normalization_stats,
                "predicted_boundaries_one_based": boundary_positions(sample_output, include_terminal=True).cpu().tolist(),
                "code_usage": code_usage(sample_output).cpu().tolist(),
            }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
