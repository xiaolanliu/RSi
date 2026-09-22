"""A PyTorch reference implementation of the CompILE model.

The package intentionally keeps the original paper's decomposition small and
explicit: a recurrent recognition model predicts soft boundaries and segment
codes, while per-code policies reconstruct the demonstrated actions.
"""

from .config import CompILEConfig
from .data import Episode, LeRobotEpisodeDataset, MultiRootEpisodeDataset, collate_episodes, load_episodes
from .model import CompILE, CompILEOutput, compute_segment_masks

__all__ = [
    "CompILE",
    "CompILEConfig",
    "CompILEOutput",
    "Episode",
    "LeRobotEpisodeDataset",
    "MultiRootEpisodeDataset",
    "collate_episodes",
    "compute_segment_masks",
    "load_episodes",
]
