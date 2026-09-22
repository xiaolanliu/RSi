"""Core CompILE recognition, soft segmentation and policy decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import CompILEConfig


def compute_segment_masks(boundary_probs: Tensor, valid_mask: Tensor) -> Tensor:
    """Compute the paper's differentiable segment masks.

    Args:
        boundary_probs: ``[batch, segments, max_length + 1]``. Index ``j``
            represents the one-based boundary value ``b = j + 1``. The last
            segment boundary is expected to be a one-hot distribution at
            ``T + 1`` (the caller handles per-example padding).
        valid_mask: ``[batch, max_length]``.

    Returns:
        Soft masks ``P(t in C_i)`` with shape ``[batch, segments, max_length]``.
    """

    batch, segments, positions = boundary_probs.shape
    max_length = valid_mask.shape[1]
    if positions != max_length + 1:
        raise ValueError("boundary_probs must have max_length + 1 positions")
    # For frame t=1..T, Eq. (8) uses P(b <= t), which corresponds to CDF
    # indices 0..T-1. The last index represents b=T+1.
    cdf = boundary_probs.cumsum(dim=-1)[..., :max_length]
    previous_cdf = torch.ones(batch, 1, max_length, device=boundary_probs.device, dtype=boundary_probs.dtype)
    if segments > 1:
        previous_cdf = torch.cat(
            [previous_cdf, boundary_probs[:, :-1].cumsum(dim=-1)[..., :max_length]], dim=1
        )
    # Product over all earlier boundaries, including b_0 = 1.
    previous_product = torch.cumprod(previous_cdf, dim=1)
    masks = (1.0 - cdf) * previous_product
    return masks * valid_mask[:, None, :].to(masks.dtype)


@dataclass
class CompILEOutput:
    loss: Tensor
    reconstruction_loss: Tensor
    kl_z: Tensor
    kl_b: Tensor
    # Kept in the output schema for compatibility with older checkpoints and
    # CSV readers. The former SmoothL1 occupancy regularizer is disabled in
    # the current objective, so this is always a detached zero scalar.
    segment_balance: Tensor
    boundary_probs: Tensor
    code_probs: Tensor
    boundary_samples: Tensor
    code_samples: Tensor
    segment_masks: Tensor
    action_prediction: Tensor
    boundary_positions: Tensor


class _StateEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, states: Tensor) -> Tensor:
        return self.net(states)


class _FiLMFusion(nn.Module):
    """Causal FiLM conditioning of state features on visual features.

    Instead of adding the two modalities, the encoded visual feature at
    timestep ``t`` predicts a per-feature scale ``gamma`` and shift ``beta``
    that modulate the encoded state feature at the *same* timestep. This is
    causal (no future information) and multiplicative/additive on the state
    pathway, so the visual stream *conditions* the state stream rather than
    being summed with it.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.to_gamma = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.to_beta = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Start as an identity modulation: gamma=1, beta=0.
        nn.init.zeros_(self.to_gamma[1].weight)
        nn.init.zeros_(self.to_gamma[1].bias)
        nn.init.zeros_(self.to_beta[1].weight)
        nn.init.zeros_(self.to_beta[1].bias)

    def forward(self, state_features: Tensor, visual_features: Tensor) -> Tensor:
        gamma = 1.0 + self.to_gamma(visual_features)
        beta = self.to_beta(visual_features)
        return state_features * gamma + beta


class _CausalCrossAttentionFusion(nn.Module):
    """Deep causal two-stream attention fusion of state and visual features.

    The encoded state and visual tokens are stacked into a single two-stream
    sequence ``[S_1..S_T, V_1..V_T]`` (each with its own sinusoidal position
    encoding) and processed by a small pre-norm Transformer encoder under a
    structured causal mask:

    - ``S_t`` attends to ``{S_<=t}`` and ``{V_<=t}``: the state representation
      may read past states and past *and current* observations.
    - ``V_t`` attends only to ``{V_<=t}``: the visual stream evolves causally
      on its own and never reads state tokens, so information cannot flow
      backwards in time through the visual branch.

    The fused representation is taken from the state positions and combined
    with the original state features through a zero-initialised gated
    residual, so the module starts as an identity mapping and gradually learns
    how much visual context to inject. This replaces the previous shallow
    single-query cross-attention (one KV per timestep), whose attention
    weights degenerated to a constant and which was effectively a projection.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.position_encoding = _SinusoidalPositionEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Zero-init the output projection so the branch starts as identity.
        nn.init.zeros_(self.out_proj[1].weight)
        nn.init.zeros_(self.out_proj[1].bias)
        # Per-channel learnable gate (sigmoid) so every encoder channel
        # receives gradient even when the gates start near zero; a scalar
        # tanh-gate would collapse dL/d(encoder) onto one shared scalar.
        self.gate = nn.Parameter(torch.full((hidden_dim,), -2.0))

    def _causal_mask(self, length: int, device: torch.device) -> Tensor:
        # Rows are queries, columns are keys for the stacked sequence
        # [state tokens | visual tokens]. -inf blocks attention.
        mask = torch.full((2 * length, 2 * length), float("-inf"), device=device)
        causal = torch.tril(torch.ones(length, length, device=device, dtype=torch.bool))
        # State queries: keys = past states and past+current visual tokens.
        mask[:length, :length][causal] = 0.0
        mask[:length, length:][causal] = 0.0
        # Visual queries: keys = past visual tokens only (never states).
        mask[length:, length:][causal] = 0.0
        return mask

    def forward(self, state_features: Tensor, visual_features: Tensor) -> Tensor:
        length = state_features.shape[1]
        state_tokens = self.position_encoding(state_features)
        visual_tokens = self.position_encoding(visual_features)
        tokens = torch.cat([state_tokens, visual_tokens], dim=1)
        mask = self._causal_mask(length, state_features.device)
        encoded = self.encoder(tokens, mask=mask)
        fused = encoded[:, :length]
        return state_features + torch.sigmoid(self.gate) * self.out_proj(fused)


class _SinusoidalPositionEncoding(nn.Module):
    """Length-agnostic positional encoding for variable-length episodes.

    The encoding is *subtracted* rather than added. Zeroed (information-
    barrier) frames in the recognition loop must remain zero after
    position encoding; adding the encoding would smear content across the
    barrier through any non-causal operator.
    """

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, inputs: Tensor) -> Tensor:
        length = inputs.shape[1]
        positions = torch.arange(length, device=inputs.device, dtype=inputs.dtype).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.dimension, 2, device=inputs.device, dtype=inputs.dtype)
            * (-math.log(10000.0) / self.dimension)
        )
        encoding = torch.zeros(length, self.dimension, device=inputs.device, dtype=inputs.dtype)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        return inputs - encoding.unsqueeze(0)


class _TemporalConvBlock(nn.Module):
    """Residual same-length dilated temporal convolution block.

    With ``causal=True`` each convolution is left-padded by the full
    receptive-field offset and the trailing outputs are trimmed, so the
    output at frame ``t`` only depends on frames ``<= t`` (receptive field
    ``2 * dilation * (kernel_size - 1)`` frames into the past for the two
    stacked convolutions). In causal mode the normalisation is a per-frame
    LayerNorm over channels: any ``GroupNorm`` on Conv1d output pools
    statistics over the temporal axis and would re-introduce a future leak
    even with causal convolutions. With ``causal=False`` the legacy
    symmetric ("same") padding + GroupNorm form is kept for checkpoint
    compatibility, and it does leak future frames.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, causal: bool = True):
        super().__init__()
        self.causal = causal
        if causal:
            # Each of the two stacked convolutions is left-padded by its own
            # receptive-field offset, so both stay same-length and causal.
            self.left_pad = dilation * (kernel_size - 1)
            padding = 0
            # LayerNorm over channels, applied per frame: no statistics are
            # shared across time, which is required for causality.
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)
        else:
            self.left_pad = 0
            padding = dilation * (kernel_size - 1) // 2
            self.norm1 = nn.GroupNorm(1, channels)
            self.norm2 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)

    def _normalize(self, norm: nn.Module, hidden: Tensor) -> Tensor:
        if self.causal:
            return norm(hidden.transpose(1, 2)).transpose(1, 2)
        return norm(hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        hidden = inputs
        if self.causal:
            hidden = F.pad(hidden, (self.left_pad, 0))
        hidden = self._normalize(self.norm1, self.conv1(hidden))
        hidden = self.dropout(F.gelu(hidden))
        if self.causal:
            hidden = F.pad(hidden, (self.left_pad, 0))
        hidden = self._normalize(self.norm2, self.conv2(hidden))
        return F.gelu(residual + hidden)


class _PolicyBank(nn.Module):
    """One independent low-level policy head per latent code.

    Besides the raw proprio state, the heads can be conditioned on the fused
    recognition features (state + visual after the main sequence encoder).
    The conditioning enters through a zero-initialised projection, so the
    policies start as state-only and gradually learn how much observational
    context to use for the action reconstruction.
    """

    def __init__(self, config: CompILEConfig):
        super().__init__()
        self.config = config
        self.state_encoder = _StateEncoder(config.state_dim, config.state_hidden_dim, config.embedding_dim)
        self.uses_fused = config.policy_uses_fused_features
        if self.uses_fused:
            self.fused_projection = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.embedding_dim),
            )
            nn.init.zeros_(self.fused_projection[1].weight)
            nn.init.zeros_(self.fused_projection[1].bias)
        else:
            self.fused_projection = None
        self.policy_heads = nn.ModuleList()
        output_dim = config.action_dim if config.action_mode == "continuous" else int(config.action_classes)
        for _ in range(config.num_codes):
            self.policy_heads.append(
                nn.Sequential(
                    nn.Linear(config.embedding_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, output_dim),
                )
            )
        if config.action_mode == "continuous":
            self.log_std = nn.Parameter(
                torch.full((config.num_codes, config.action_dim), config.action_log_std_init)
            )

    def forward(self, states: Tensor, fused_features: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        encoded = self.state_encoder(states)
        if self.uses_fused:
            if fused_features is None:
                raise ValueError("policy_uses_fused_features is enabled but no fused features were provided")
            encoded = encoded + self.fused_projection(fused_features)
        means_or_logits = torch.stack([head(encoded) for head in self.policy_heads], dim=2)
        if self.config.action_mode == "continuous":
            return means_or_logits, self.log_std
        return means_or_logits, None


class CompILE(nn.Module):
    """CompILE with continuous or categorical action reconstruction.

    ``forward`` accepts padded batches from :func:`collate_episodes`. The
    model predicts ``max_segments - 1`` learned boundaries and appends a fixed
    terminal boundary at the end of each sequence, which guarantees that soft
    segment masks cover every valid frame.
    """

    def __init__(self, config: CompILEConfig):
        super().__init__()
        self.config = config
        state_mean = torch.tensor(config.state_mean or [0.0] * config.state_dim, dtype=torch.float32)
        state_std = torch.tensor(config.state_std or [1.0] * config.state_dim, dtype=torch.float32)
        action_mean = torch.tensor(config.action_mean or [0.0] * config.action_dim, dtype=torch.float32)
        action_std = torch.tensor(config.action_std or [1.0] * config.action_dim, dtype=torch.float32)
        self.register_buffer("state_mean", state_mean, persistent=True)
        self.register_buffer("state_std", state_std.clamp_min(1e-6), persistent=True)
        self.register_buffer("action_mean", action_mean, persistent=True)
        self.register_buffer("action_std", action_std.clamp_min(1e-6), persistent=True)
        self.state_input_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        if config.visual_dim > 0:
            self.visual_input_encoder = nn.Sequential(
                nn.LayerNorm(config.visual_dim),
                nn.Linear(config.visual_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
            )
            # Legacy additive gate (fusion_mode == "gate").
            self.visual_gate = nn.Sequential(
                nn.LayerNorm(config.visual_dim),
                nn.Linear(config.visual_dim, config.hidden_dim),
                nn.Sigmoid(),
            )
            # Start with a vision-favouring gate while allowing the network to
            # reduce visual influence when it is uninformative.
            nn.init.constant_(self.visual_gate[1].bias, 1.0)
            # Causal fusion alternatives. Only the selected one is used in
            # ``_recognize``; the others remain unused (and absent from older
            # checkpoints, which is handled by strict=False loading if needed).
            self.film_fusion = _FiLMFusion(config.hidden_dim)
            self.attention_fusion = _CausalCrossAttentionFusion(
                config.hidden_dim,
                num_heads=max(1, min(4, config.transformer_heads)),
                num_layers=2,
                dropout=config.transformer_dropout,
            )
        else:
            self.visual_input_encoder = None
            self.visual_gate = None
            self.film_fusion = None
            self.attention_fusion = None
        self.position_encoding = _SinusoidalPositionEncoding(config.hidden_dim)
        if config.sequence_encoder == "tcn":
            self.sequence_encoder = nn.Sequential(
                *[
                    _TemporalConvBlock(
                        config.hidden_dim,
                        config.tcn_kernel_size,
                        dilation,
                        config.transformer_dropout,
                        causal=config.causal_sequence_encoder,
                    )
                    for dilation in config.tcn_dilations
                ],
            )
            self.sequence_norm = nn.LayerNorm(config.hidden_dim)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.transformer_heads,
                dim_feedforward=4 * config.hidden_dim,
                dropout=config.transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.transformer_layers,
                norm=nn.LayerNorm(config.hidden_dim),
            )
            self.sequence_norm = nn.Identity()
        self.boundary_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.code_head = nn.Linear(config.hidden_dim, config.num_codes)
        self.policy_bank = _PolicyBank(config)

    @staticmethod
    def _masked_logits(logits: Tensor, valid_mask: Tensor) -> Tensor:
        """Mask b=1 and per-example padding positions without NaNs."""

        logits = logits.clone()
        logits[..., 0] = -1e9  # b=1 would create an empty first segment.
        positions = torch.arange(logits.shape[-1], device=logits.device)[None, :]
        # Boundary positions after the last valid frame are legal only at L+1.
        lengths = valid_mask.sum(dim=-1).long()
        allowed = positions <= lengths[:, None]
        logits = logits.masked_fill(~allowed, -1e9)
        return logits

    def _state_normalize(self, values: Tensor, valid_mask: Tensor) -> Tensor:
        mean = self.state_mean.to(device=values.device, dtype=values.dtype)
        std = self.state_std.to(device=values.device, dtype=values.dtype)
        normalized = (values - mean) / std
        return normalized * valid_mask.unsqueeze(-1).to(normalized.dtype)

    def _action_normalize(self, values: Tensor) -> Tensor:
        mean = self.action_mean.to(device=values.device, dtype=values.dtype)
        std = self.action_std.to(device=values.device, dtype=values.dtype)
        return (values - mean) / std

    def _action_denormalize(self, values: Tensor) -> Tensor:
        mean = self.action_mean.to(device=values.device, dtype=values.dtype)
        std = self.action_std.to(device=values.device, dtype=values.dtype)
        return values * std + mean

    def _recognize(
        self,
        states: Tensor,
        visual_features: Tensor | None,
        valid_mask: Tensor,
        *,
        sample_latents: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # The last return value is the fused per-timestep input sequence. It
        # conditions the policy bank: unlike the in-loop encoded features it
        # never passes under the ``previous_product`` information barrier, so
        # its frame-t content only depends on frames <= t (for the causal
        # fusion modes), independent of the segment posteriors.
        batch, max_length, _ = states.shape
        segments = self.config.max_segments
        if self.config.visual_dim > 0:
            if visual_features is None:
                raise ValueError("visual_dim is configured but batch has no visual_features")
            if visual_features.shape[:2] != states.shape[:2] or visual_features.shape[-1] != self.config.visual_dim:
                raise ValueError(
                    f"visual_features must have shape [B,T,{self.config.visual_dim}], got {tuple(visual_features.shape)}"
                )
            visual_encoded = self.visual_input_encoder(visual_features)
            state_encoded = self.state_input_encoder(states)
            if self.config.fusion_mode == "film":
                # FiLM: visual features causally condition the state features.
                inputs = self.film_fusion(state_encoded, visual_encoded)
            elif self.config.fusion_mode == "attention":
                # Deep causal two-stream attention fusion.
                inputs = self.attention_fusion(state_encoded, visual_encoded)
            else:  # "gate" legacy additive fusion
                visual_gate = self.visual_gate(visual_features)
                inputs = state_encoded + visual_gate * visual_encoded
        else:
            if visual_features is not None:
                raise ValueError("batch contains visual_features but config.visual_dim is zero")
            inputs = self.state_input_encoder(states)
        boundary_posteriors = []
        boundary_samples = []
        code_posteriors = []
        code_samples = []
        previous_product = torch.ones(batch, max_length, device=states.device, dtype=states.dtype)

        for segment_index in range(segments):
            # Modernized recognition backbone: a compact pre-norm Transformer
            # replaces the paper's LSTM. Multiplying the token sequence by the
            # cumulative previous-segment mask gives the same soft-prefix
            # information barrier while preserving differentiability.
            masked_inputs = inputs * previous_product.unsqueeze(-1)
            encoded = self.position_encoding(masked_inputs)
            # TCN is channel-first and has linear O(T) memory. Transformer is
            # retained as an explicit compatibility option for old experiments.
            if self.config.sequence_encoder == "tcn":
                encoded = self.sequence_encoder(encoded.transpose(1, 2)).transpose(1, 2)
                encoded = self.sequence_norm(encoded)
            else:
                encoded = encoded * previous_product.unsqueeze(-1)
                attn_mask = None
                if self.config.causal_sequence_encoder:
                    attn_mask = torch.triu(
                        torch.ones(max_length, max_length, device=states.device, dtype=torch.bool),
                        diagonal=1,
                    )
                encoded = self.sequence_encoder(
                    encoded, mask=attn_mask, src_key_padding_mask=~valid_mask
                )
            hidden_sequence = encoded * previous_product.unsqueeze(-1)
            boundary_logits = self.boundary_head(hidden_sequence).squeeze(-1)
            # Recurrent output t predicts boundary b=t. Append the legal
            # terminal position b=T+1, which has no recurrent output.
            boundary_logits = F.pad(boundary_logits, (0, 1), value=0.0)
            boundary_logits = self._masked_logits(boundary_logits, valid_mask)
            if segment_index == segments - 1:
                # b_M = T + 1 is fixed in the generative model.
                terminal = torch.full_like(boundary_logits, -1e9)
                lengths = valid_mask.sum(dim=-1).long()
                terminal.scatter_(1, lengths[:, None], 0.0)
                boundary_logits = terminal
            posterior = F.softmax(boundary_logits, dim=-1)
            relaxed = (
                F.gumbel_softmax(
                    boundary_logits,
                    tau=self.config.gumbel_temperature,
                    hard=False,
                    dim=-1,
                )
                if sample_latents and segment_index < segments - 1
                else posterior
            )
            boundary_posteriors.append(posterior)
            boundary_samples.append(relaxed)

            # Weighted readout at the last frame of the softly selected segment.
            # Eq. (10): q(b_i=t+1) is used to softly read the code head at t.
            # Since index j represents b=j+1, these are indices 1..T.
            readout_weights = relaxed[:, 1 : max_length + 1]
            code_logits_per_frame = self.code_head(hidden_sequence)
            weighted_logits = torch.sum(readout_weights.unsqueeze(-1) * code_logits_per_frame, dim=1)
            code_posterior = F.softmax(weighted_logits, dim=-1)
            code_sample = (
                F.gumbel_softmax(
                    weighted_logits,
                    tau=self.config.gumbel_temperature,
                    hard=False,
                    dim=-1,
                )
                if sample_latents
                else code_posterior
            )
            code_posteriors.append(code_posterior)
            code_samples.append(code_sample)

            if segment_index < segments - 1:
                cdf = relaxed.cumsum(dim=-1)[:, :max_length]
                previous_product = previous_product * cdf

        return (
            torch.stack(boundary_posteriors, dim=1),
            torch.stack(code_posteriors, dim=1),
            torch.stack(boundary_samples, dim=1),
            torch.stack(code_samples, dim=1),
            inputs,
        )

    def _reconstruction_loss(
        self,
        states: Tensor,
        actions: Tensor,
        code_probs: Tensor,
        masks: Tensor,
        fused_features: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        policy_outputs, log_std = self.policy_bank(states, fused_features)
        if self.config.action_mode == "continuous":
            # Expected negative log likelihood under the relaxed code posterior.
            target = actions[:, None, :, None, :]
            means = policy_outputs[:, None, :, :, :]
            std = log_std.exp()[None, None, None, :, :]
            nll = 0.5 * (((target - means) / std) ** 2 + 2.0 * log_std[None, None, None, :, :] + math.log(2.0 * math.pi))
            nll = nll.mean(dim=-1)  # [B, 1, T, K]
            nll = nll.squeeze(1).permute(0, 2, 1)  # [B, K, T]
            expected_nll = torch.einsum("bmk,bkt,bmt->", code_probs, nll, masks)
            denominator = masks.sum().clamp_min(1.0)
            selected = torch.einsum("bmk,btka->bmta", code_probs, policy_outputs)
            action_prediction = (selected * masks.unsqueeze(-1)).sum(dim=1)
            return expected_nll / denominator, action_prediction

        if actions.shape[-1] != 1:
            raise ValueError("categorical action_mode expects one class id per frame")
        logits = policy_outputs[:, None, :, :, :]  # [B, 1, T, K, C]
        log_prob = F.log_softmax(logits, dim=-1)
        target_ids = actions.long()[:, None, :, None, :].expand(
            -1, 1, -1, self.config.num_codes, -1
        )
        gathered = log_prob.gather(-1, target_ids).squeeze(-1).squeeze(1).permute(0, 2, 1)
        expected_nll = -torch.einsum("bmk,bkt,bmt->", code_probs, gathered, masks)
        denominator = masks.sum().clamp_min(1.0)
        predicted = policy_outputs.argmax(dim=-1).float().unsqueeze(-1)
        selected = torch.einsum("bmk,btka->bmta", code_probs, predicted)
        action_prediction = (selected * masks.unsqueeze(-1)).sum(dim=1)
        return expected_nll / denominator, action_prediction

    def _kl_terms(self, boundary_probs: Tensor, code_probs: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        code_prior = torch.full_like(code_probs, 1.0 / self.config.num_codes)
        kl_z = (code_probs * (code_probs.clamp_min(1e-8).log() - code_prior.log())).sum(dim=-1).mean()

        # Appendix A.5: in the relaxed setting evaluate the first-boundary KL
        # and multiply by M under the shared-segment assumption. The first
        # segment length is b_1 - b_0 = b_1 - 1, with lengths 1..T.
        if self.config.max_segments == 1:
            return kl_z, torch.zeros((), device=boundary_probs.device, dtype=boundary_probs.dtype)
        positions_count = boundary_probs.shape[-1]
        position_ids = torch.arange(positions_count, device=boundary_probs.device)
        lengths = position_ids.to(boundary_probs.dtype)
        sequence_lengths = valid_mask.sum(dim=-1).long()
        if self.config.adaptive_poisson_rate:
            rates = (sequence_lengths.to(boundary_probs.dtype) / self.config.max_segments).clamp_min(3.0)
            raw_log_prior = (
                lengths[None, :] * rates[:, None].log()
                - rates[:, None]
                - torch.lgamma(lengths[None, :] + 1.0)
            )
        else:
            raw_log_prior = (
                lengths * math.log(self.config.poisson_rate)
                - self.config.poisson_rate
                - torch.lgamma(lengths + 1.0)
            )[None, :]
        allowed = (position_ids >= 1)[None, :] & (position_ids <= sequence_lengths[:, None])
        normalizer = torch.logsumexp(raw_log_prior.masked_fill(~allowed, -1e9), dim=-1, keepdim=True)
        prior_log = raw_log_prior - normalizer
        prior_log = prior_log.masked_fill(~allowed, 0.0)
        first = boundary_probs[:, 0]
        kl_b = (first * (first.clamp_min(1e-8).log() - prior_log)).masked_fill(~allowed, 0.0).sum(dim=-1).mean()
        kl_b = kl_b * self.config.max_segments
        return kl_z, kl_b

    def forward(self, batch: Dict[str, Tensor], *, sample_latents: bool = True) -> CompILEOutput:
        raw_states = batch["states"]
        raw_actions = batch["actions"]
        valid_mask = batch["valid_mask"].bool()
        states = self._state_normalize(raw_states, valid_mask)
        actions = self._action_normalize(raw_actions)
        boundary_probs, code_probs, boundary_samples, code_samples, fused_inputs = self._recognize(
            states,
            batch.get("visual_features"),
            valid_mask,
            sample_latents=sample_latents,
        )
        # Eq. (8) is evaluated on the relaxed boundary sample during training;
        # ``sample_latents=False`` makes the same path deterministic by using
        # the categorical posterior in place of the Gumbel sample.
        masks = compute_segment_masks(boundary_samples, valid_mask)
        action_valid_mask = batch.get("action_valid_mask", valid_mask).bool()
        reconstruction_loss, normalized_action_prediction = self._reconstruction_loss(
            states,
            actions,
            code_samples,
            masks * action_valid_mask[:, None, :],
            fused_inputs,
        )
        action_prediction = self._action_denormalize(normalized_action_prediction)
        kl_z, kl_b = self._kl_terms(boundary_probs, code_probs, valid_mask)
        # Use SmoothL1 only on occupancy violations. The feasible band keeps
        # every phase from collapsing or one phase from absorbing the episode,
        # while leaving boundaries inside the band to action/visual evidence.
        segment_mass = masks.sum(dim=-1)
        target_mass = valid_mask.sum(dim=-1, keepdim=True).to(masks.dtype) / self.config.max_segments
        normalized_mass = segment_mass / target_mass.clamp_min(1.0)
        lower_violation = F.relu(self.config.segment_balance_min_ratio - normalized_mass)
        upper_violation = F.relu(normalized_mass - self.config.segment_balance_max_ratio)
        zero = torch.zeros_like(normalized_mass)
        segment_balance = 0.5 * (
            F.smooth_l1_loss(lower_violation, zero, beta=0.25, reduction="mean")
            + F.smooth_l1_loss(upper_violation, zero, beta=0.25, reduction="mean")
        )
        loss = reconstruction_loss + self.config.kl_weight * (
            kl_z + self.config.boundary_kl_weight * kl_b
        ) + self.config.segment_balance_weight * segment_balance
        boundary_positions = boundary_probs.argmax(dim=-1)
        return CompILEOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_z=kl_z,
            kl_b=kl_b,
            segment_balance=segment_balance,
            boundary_probs=boundary_probs,
            code_probs=code_probs,
            boundary_samples=boundary_samples,
            code_samples=code_samples,
            segment_masks=masks,
            action_prediction=action_prediction,
            boundary_positions=boundary_positions,
        )
