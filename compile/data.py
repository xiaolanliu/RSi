"""LeRobot-compatible trajectory loading for CompILE.

The loader supports both common layouts:

* v2: ``data/chunk-000/episode_000000.parquet``;
* v3: ``data/chunk-000/file-000.parquet`` with an ``episode_index`` column.

The standard ``observation.state``/``action`` columns or split ``state.*`` /
``action.*`` columns are accepted. Values may be stored as parquet list
columns (the standard LeRobot representation) or as scalar columns, which
makes the interface useful for small synthetic tests too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


DEFAULT_PROPRIO_STATE_COLUMNS = ("state.joints", "state.gripper_w")


def _as_matrix(values: Sequence[Any], name: str) -> np.ndarray:
    """Convert a parquet column of arrays or scalars into a float matrix."""

    if len(values) == 0:
        raise ValueError(f"{name} is empty")
    first = values[0]
    if np.isscalar(first):
        array = np.asarray(values, dtype=np.float32)[:, None]
    else:
        array = np.asarray([np.asarray(v, dtype=np.float32) for v in values])
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D sequence, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _resolve_vector_columns(
    frame_table: pd.DataFrame,
    requested: str | Sequence[str] | None,
    *,
    prefix: str,
    preferred_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Read a standard vector or concatenate split fields in canonical order."""

    if requested is not None:
        columns = [requested] if isinstance(requested, str) else list(requested)
        missing = [column for column in columns if column not in frame_table.columns]
        if missing:
            raise ValueError(f"Requested {prefix} columns are missing: {missing}")
    elif prefix in frame_table.columns:
        columns = [prefix]
    else:
        discovered = {
            column for column in frame_table.columns if column.startswith(f"{prefix}.")
        }
        ordered = [column for column in (preferred_order or ()) if column in discovered]
        columns = ordered + sorted(discovered.difference(ordered))
        if not columns:
            raise ValueError(
                f"Could not find {prefix!r} or split {prefix}.* columns in parquet table"
            )
    matrices = [_as_matrix(frame_table[column].tolist(), column) for column in columns]
    if len({matrix.shape[0] for matrix in matrices}) != 1:
        raise ValueError(f"Split {prefix} columns have inconsistent row counts")
    return np.concatenate(matrices, axis=1), columns


@dataclass(frozen=True)
class Episode:
    """One complete state-action trajectory."""

    episode_index: int
    states: torch.Tensor
    actions: torch.Tensor
    frame_indices: torch.Tensor
    metadata: Dict[str, Any]
    visual_features: torch.Tensor | None = None
    action_valid_mask: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return int(self.states.shape[0])


def discover_data_files(root: Path) -> List[Path]:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found below {root / 'data'}")
    return files


def _read_info(root: Path) -> Dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LeRobot info.json: {path}") from exc


def _iter_episode_frames(root: Path, *, require_action: bool = True) -> Iterable[tuple[int, pd.DataFrame]]:
    """Yield episode-indexed frames while preserving v2/v3 compatibility."""

    for path in discover_data_files(root):
        frame_table = pd.read_parquet(path)
        if not (
            "observation.state" in frame_table.columns
            or any(column.startswith("state.") for column in frame_table.columns)
        ):
            raise ValueError(f"{path} has no observation.state or state.* columns")
        if require_action and not (
            "action" in frame_table.columns
            or any(column.startswith("action.") for column in frame_table.columns)
        ):
            raise ValueError(f"{path} has no action or action.* columns")

        if "episode_index" in frame_table.columns:
            episode_values = frame_table["episode_index"].to_numpy()
            for episode_index in sorted(np.unique(episode_values).tolist()):
                subset = frame_table.loc[episode_values == episode_index].copy()
                yield int(episode_index), subset
        else:
            # v2 episode files encode the index in the filename.
            stem = path.stem
            if not stem.startswith("episode_"):
                raise ValueError(
                    f"Cannot infer episode_index from {path}; add episode_index column"
                )
            yield int(stem.split("_")[-1]), frame_table


def load_episodes(
    root: str | Path,
    *,
    max_episodes: int | None = None,
    max_length: int | None = None,
    state_columns: str | Sequence[str] | None = None,
    action_columns: str | Sequence[str] | None = None,
    visual_cache_dir: str | Path | None = None,
    temporal_stride: int = 1,
    derive_actions_from_state: bool = False,
) -> List[Episode]:
    """Load complete trajectories from a LeRobot dataset root."""

    root = Path(root)
    info = _read_info(root)
    visual_cache = Path(visual_cache_dir) if visual_cache_dir is not None else None
    if temporal_stride < 1:
        raise ValueError("temporal_stride must be positive")
    episodes: List[Episode] = []
    seen: set[int] = set()
    for episode_index, frame_table in _iter_episode_frames(
        root, require_action=not derive_actions_from_state
    ):
        if episode_index in seen:
            raise ValueError(f"Episode {episode_index} appears in multiple files")
        seen.add(episode_index)
        sort_column = "frame_index" if "frame_index" in frame_table else "index"
        if sort_column in frame_table:
            frame_table = frame_table.sort_values(sort_column, kind="stable")
        state_prefix = (
            "observation.state"
            if "observation.state" in frame_table.columns
            else "state"
        )
        effective_state_columns = state_columns
        if effective_state_columns is None and "state.joints" in frame_table.columns and "state.gripper_w" in frame_table.columns:
            effective_state_columns = DEFAULT_PROPRIO_STATE_COLUMNS
        states, resolved_state_columns = _resolve_vector_columns(
            frame_table,
            effective_state_columns,
            prefix=state_prefix,
            preferred_order=[
                column
                for column in info.get("features", {})
                if column.startswith(f"{state_prefix}.")
            ],
        )
        actions = None
        resolved_action_columns: list[str] = []
        if not derive_actions_from_state:
            actions, resolved_action_columns = _resolve_vector_columns(
                frame_table,
                action_columns,
                prefix="action",
                preferred_order=[
                    column
                    for column in info.get("features", {})
                    if column.startswith("action.")
                ],
            )
            if states.shape[0] != actions.shape[0]:
                raise ValueError(f"Episode {episode_index} has mismatched state/action lengths")
        if temporal_stride > 1:
            states = states[::temporal_stride]
            if actions is not None:
                actions = actions[::temporal_stride]
        if derive_actions_from_state:
            # The dataset records state at every frame. Treat the action at t
            # as the transition to t+1, with no fabricated transition after
            # the episode terminator.
            actions = np.zeros_like(states, dtype=np.float32)
            actions[:-1] = states[1:] - states[:-1]
            action_valid = np.ones(states.shape[0], dtype=bool)
            action_valid[-1] = False
        else:
            action_valid = np.ones(states.shape[0], dtype=bool)
        if max_length is not None:
            if max_length < 1:
                raise ValueError("max_length must be positive")
            states, actions = states[:max_length], actions[:max_length]
            action_valid = action_valid[:max_length]
        if states.shape[0] == 0:
            continue
        if "frame_index" in frame_table:
            frame_indices = frame_table["frame_index"].to_numpy(dtype=np.int64)[::temporal_stride][: len(states)]
        else:
            frame_indices = np.arange(len(states), dtype=np.int64) * temporal_stride
        visual_features = None
        if visual_cache is not None:
            candidates = [
                visual_cache / f"episode_{episode_index:06d}.npz",
                visual_cache / f"episode_{episode_index:03d}.npz",
                visual_cache / f"episode_{episode_index}.npz",
            ]
            cache_path = next((path for path in candidates if path.exists()), None)
            if cache_path is None:
                raise FileNotFoundError(
                    f"No visual cache for episode {episode_index}; checked {candidates}"
                )
            with np.load(cache_path) as cache:
                if "features" not in cache:
                    raise ValueError(f"Visual cache {cache_path} has no 'features' array")
                visual_array = np.asarray(cache["features"], dtype=np.float32)
            sampled_visual = visual_array[::temporal_stride]
            if sampled_visual.ndim != 2 or sampled_visual.shape[0] < len(states):
                raise ValueError(
                    f"Visual cache {cache_path} shape {visual_array.shape} cannot cover {len(states)} sampled frames"
                )
            visual_features = torch.from_numpy(sampled_visual[: len(states)])
            if not np.isfinite(sampled_visual[: len(states)]).all():
                raise ValueError(f"Visual cache {cache_path} contains NaN or infinite values")
        episodes.append(
            Episode(
                episode_index=episode_index,
                states=torch.from_numpy(states),
                actions=torch.from_numpy(actions),
                frame_indices=torch.from_numpy(frame_indices),
                metadata={
                    "source_root": str(root),
                    "info": info,
                    "state_columns": resolved_state_columns,
                    "action_columns": resolved_action_columns,
                    "action_source": "state_transition_delta" if derive_actions_from_state else "dataset_action_columns",
                    "visual_cache_dir": str(visual_cache) if visual_cache is not None else None,
                    "temporal_stride": temporal_stride,
                },
                visual_features=visual_features,
                action_valid_mask=torch.from_numpy(action_valid),
            )
        )
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    if not episodes:
        raise ValueError(f"No non-empty episodes found below {root}")
    return sorted(episodes, key=lambda episode: episode.episode_index)


class LeRobotEpisodeDataset(Dataset[Episode]):
    """A map-style dataset returning complete trajectories."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_episodes: int | None = None,
        max_length: int | None = None,
        state_columns: str | Sequence[str] | None = None,
        action_columns: str | Sequence[str] | None = None,
        visual_cache_dir: str | Path | None = None,
        temporal_stride: int = 1,
        derive_actions_from_state: bool = False,
    ):
        self.root = Path(root)
        self.episodes = load_episodes(
            self.root,
            max_episodes=max_episodes,
            max_length=max_length,
            state_columns=state_columns,
            action_columns=action_columns,
            visual_cache_dir=visual_cache_dir,
            temporal_stride=temporal_stride,
            derive_actions_from_state=derive_actions_from_state,
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> Episode:
        return self.episodes[index]

    @property
    def state_dim(self) -> int:
        return int(self.episodes[0].states.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.episodes[0].actions.shape[-1])

    @property
    def visual_dim(self) -> int:
        dims = {int(episode.visual_features.shape[-1]) for episode in self.episodes if episode.visual_features is not None}
        if not dims:
            return 0
        if len(dims) != 1 or any(episode.visual_features is None for episode in self.episodes):
            raise ValueError("All episodes must have visual features with the same dimension")
        return next(iter(dims))


class MultiRootEpisodeDataset(Dataset[Episode]):
    """In-memory concatenation of complete episodes from multiple roots.

    Each root keeps its own visual-cache directory. This is important because
    episode indices restart at zero for every LeRobot dataset.
    """

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        visual_cache_dirs: Sequence[str | Path | None] | None = None,
        max_episodes_per_root: int | None = None,
        max_length: int | None = None,
        temporal_stride: int = 1,
        state_columns: str | Sequence[str] | None = None,
        action_columns: str | Sequence[str] | None = None,
        derive_actions_from_state: bool = False,
    ):
        if not roots:
            raise ValueError("roots must contain at least one dataset root")
        if visual_cache_dirs is None:
            visual_cache_dirs = [None] * len(roots)
        if len(visual_cache_dirs) != len(roots):
            raise ValueError("visual_cache_dirs must have the same length as roots")
        self.roots = [Path(root) for root in roots]
        self.episodes: list[Episode] = []
        for root, cache_dir in zip(self.roots, visual_cache_dirs):
            loaded = load_episodes(
                root,
                max_episodes=max_episodes_per_root,
                max_length=max_length,
                state_columns=state_columns,
                action_columns=action_columns,
                visual_cache_dir=cache_dir,
                temporal_stride=temporal_stride,
                derive_actions_from_state=derive_actions_from_state,
            )
            for episode in loaded:
                metadata = dict(episode.metadata)
                metadata["dataset_name"] = root.name
                self.episodes.append(
                    Episode(
                        episode_index=episode.episode_index,
                        states=episode.states,
                        actions=episode.actions,
                        frame_indices=episode.frame_indices,
                        metadata=metadata,
                        visual_features=episode.visual_features,
                        action_valid_mask=episode.action_valid_mask,
                    )
                )
        if not self.episodes:
            raise ValueError("No episodes loaded from the provided roots")
        state_dims = {episode.states.shape[-1] for episode in self.episodes}
        action_dims = {episode.actions.shape[-1] for episode in self.episodes}
        if len(state_dims) != 1 or len(action_dims) != 1:
            raise ValueError("All roots must have consistent state/action dimensions")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> Episode:
        return self.episodes[index]

    @property
    def state_dim(self) -> int:
        return int(self.episodes[0].states.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.episodes[0].actions.shape[-1])

    @property
    def visual_dim(self) -> int:
        dims = {
            int(episode.visual_features.shape[-1])
            for episode in self.episodes
            if episode.visual_features is not None
        }
        if not dims:
            return 0
        if len(dims) != 1 or any(episode.visual_features is None for episode in self.episodes):
            raise ValueError("All episodes must have visual features with the same dimension")
        return next(iter(dims))


def collate_episodes(batch: Sequence[Episode]) -> Dict[str, torch.Tensor | List[int]]:
    """Pad variable-length episodes and return a boolean valid-frame mask."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    states = pad_sequence([episode.states for episode in batch], batch_first=True)
    actions = pad_sequence([episode.actions for episode in batch], batch_first=True)
    frame_indices = pad_sequence(
        [episode.frame_indices for episode in batch], batch_first=True, padding_value=-1
    )
    lengths = torch.tensor([episode.length for episode in batch], dtype=torch.long)
    steps = torch.arange(states.shape[1]).unsqueeze(0)
    valid_mask = steps < lengths.unsqueeze(1)
    result: Dict[str, torch.Tensor | List[int]] = {
        "states": states,
        "actions": actions,
        "frame_indices": frame_indices,
        "valid_mask": valid_mask,
        "action_valid_mask": pad_sequence(
            [
                episode.action_valid_mask
                if getattr(episode, "action_valid_mask", None) is not None
                else torch.ones(episode.length, dtype=torch.bool)
                for episode in batch
            ],
            batch_first=True,
            padding_value=False,
        ),
        "lengths": lengths,
        "episode_indices": [episode.episode_index for episode in batch],
    }
    visual_presence = [getattr(episode, "visual_features", None) is not None for episode in batch]
    if any(visual_presence):
        if not all(visual_presence):
            raise ValueError("A batch cannot mix episodes with and without visual features")
        result["visual_features"] = pad_sequence(
            [episode.visual_features for episode in batch if getattr(episode, "visual_features", None) is not None],
            batch_first=True,
        )
    return result
