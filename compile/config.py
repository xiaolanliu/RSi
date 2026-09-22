"""Configuration objects for the initial CompILE implementation."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class CompILEConfig:
    """Model and objective settings.

    ``max_segments`` is the maximum number of segments represented by a
    trajectory. The final boundary is fixed to ``T + 1`` as in the paper;
    therefore only ``max_segments - 1`` boundary distributions are learned.
    """

    state_dim: int
    action_dim: int
    max_segments: int = 3
    num_codes: int = 10
    hidden_dim: int = 256
    embedding_dim: int = 128
    state_hidden_dim: int = 128
    action_mode: Literal["continuous", "categorical"] = "continuous"
    visual_dim: int = 0
    # How the visual branch is fused with the state branch in the recognition
    # encoder. ``gate`` is the legacy additive form (state_enc + gate*visual_enc).
    # ``film`` conditions the state features on a per-timestep (gamma, beta)
    # produced from the visual features (causal, feature-wise modulation).
    # ``attention`` treats the visual token as a causal cross-attention
    # key/value that the state query attends to at the same timestep.
    fusion_mode: Literal["gate", "film", "attention"] = "gate"
    sequence_encoder: Literal["tcn", "transformer"] = "tcn"
    # Causal main sequence encoder: the TCN uses left-padded (causal)
    # convolutions and the Transformer branch adds a causal attention mask,
    # so the boundary/code readout at frame t only sees frames <= t. This
    # matches the causal first-boundary KL prior (Appendix A.5) and prevents
    # the recognition net from becoming a lookahead change-point detector.
    # Set to False only if you also replace the boundary KL with a proper
    # per-segment prior (bidirectional inference network).
    causal_sequence_encoder: bool = True
    # Feed the fused recognition features (state + visual, after the main
    # sequence encoder) to the policy bank in addition to the raw proprio
    # state, so observations can directly modulate the action reconstruction.
    policy_uses_fused_features: bool = True
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_dropout: float = 0.0
    # Kept for checkpoint/API compatibility. Visual fusion now uses a learned
    # modality gate rather than multiplying latents by a fixed scalar.
    visual_input_scale: float = 1.0
    tcn_kernel_size: int = 5
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8)
    # For categorical actions, ``action_dim`` is the number of action
    # channels in the input (the initial implementation supports one scalar
    # categorical action), while this field is the number of classes.
    action_classes: int | None = None
    gumbel_temperature: float = 1.0
    poisson_rate: float = 3.0
    adaptive_poisson_rate: bool = False
    kl_weight: float = 1.0
    boundary_kl_weight: float = 0.01
    # Weight for the bounded SmoothL1 occupancy regularizer. Unlike the old
    # equal-occupancy target, it only penalizes phases outside a feasible
    # length band around the episode average.
    segment_balance_weight: float = 0.02
    segment_balance_min_ratio: float = 0.5
    segment_balance_max_ratio: float = 1.8
    action_log_std_init: float = -1.0
    min_sequence_length: int = 2
    temporal_stride: int = 1
    state_mean: tuple[float, ...] | None = None
    state_std: tuple[float, ...] | None = None
    action_mean: tuple[float, ...] | None = None
    action_std: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if self.fusion_mode not in ("gate", "film", "attention"):
            raise ValueError("fusion_mode must be one of gate/film/attention")
        if self.visual_dim < 0:
            raise ValueError("visual_dim must be non-negative")
        if self.visual_input_scale < 0:
            raise ValueError("visual_input_scale must be non-negative")
        for name, values, expected_dim in (
            ("state_mean", self.state_mean, self.state_dim),
            ("state_std", self.state_std, self.state_dim),
            ("action_mean", self.action_mean, self.action_dim),
            ("action_std", self.action_std, self.action_dim),
        ):
            if values is not None and len(values) != expected_dim:
                raise ValueError(f"{name} must have length {expected_dim}")
        if self.state_std is not None and any(value <= 0 for value in self.state_std):
            raise ValueError("state_std values must be positive")
        if self.action_std is not None and any(value <= 0 for value in self.action_std):
            raise ValueError("action_std values must be positive")
        if self.tcn_kernel_size < 3 or self.tcn_kernel_size % 2 == 0:
            raise ValueError("tcn_kernel_size must be odd and at least 3")
        if not self.tcn_dilations or any(dilation < 1 for dilation in self.tcn_dilations):
            raise ValueError("tcn_dilations must contain positive values")
        if self.temporal_stride < 1:
            raise ValueError("temporal_stride must be positive")
        if self.max_segments < 1:
            raise ValueError("max_segments must be at least one")
        if self.num_codes < 1:
            raise ValueError("num_codes must be at least one")
        if self.gumbel_temperature <= 0:
            raise ValueError("gumbel_temperature must be positive")
        if self.poisson_rate <= 0:
            raise ValueError("poisson_rate must be positive")
        if self.transformer_layers < 1 or self.transformer_heads < 1:
            raise ValueError("transformer_layers and transformer_heads must be positive")
        if self.hidden_dim % self.transformer_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if not 0.0 <= self.transformer_dropout < 1.0:
            raise ValueError("transformer_dropout must be in [0, 1)")
        if self.kl_weight < 0 or self.boundary_kl_weight < 0:
            raise ValueError("KL weights must be non-negative")
        if self.segment_balance_weight < 0:
            raise ValueError("segment_balance_weight must be non-negative")
        if not 0.0 <= self.segment_balance_min_ratio <= 1.0:
            raise ValueError("segment_balance_min_ratio must be in [0, 1]")
        if self.segment_balance_max_ratio < 1.0:
            raise ValueError("segment_balance_max_ratio must be at least 1")
        if self.action_mode == "categorical":
            if self.action_dim != 1:
                raise ValueError("categorical action_mode currently supports action_dim=1")
            if self.action_classes is None or self.action_classes < 2:
                raise ValueError("categorical action_mode requires action_classes >= 2")
