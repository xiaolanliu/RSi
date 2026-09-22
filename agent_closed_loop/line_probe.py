"""Causal-frame phase probe with an explicitly post-hoc LINe scoring branch.

CompILE's five slots are ordinal, latent temporal phases, not supervised semantic
classes. A trainer may distil its ID-only offline masks into this per-frame probe;
the probe itself never consumes a trajectory's future masks at inference.

Reference: Ahn et al., LINe, CVPR 2023, https://arxiv.org/abs/2303.13995.
We use the paper's absolute Taylor contribution E[|h_j W_cj|] (Eq. 4).
The authors' released precompute code instead retains signed h*gradient for AP.
Weight pruning below retains the signed contribution-times-weight ranking, over
the entire classifier matrix separately for each routing class, as in LINe.
Our exact top-k masks resolve ties deterministically; the released implementation
uses a strict percentile comparison and can retain fewer units at tied values.

The raw classification branch stays unchanged by fitting LINe. The LINe energy
is an OOD ranking score, not a calibrated probability or recoverability estimate.
The authors' code returns positive logsumexp as ID confidence; this module uses
its negative so that larger energy consistently means more anomalous.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _largest_mask(values: Tensor, keep: float) -> Tensor:
    """Keep ceil(n*keep) entries, breaking equal ranks by flattened index."""
    if not 0 < keep <= 1:
        raise ValueError("keep fraction must be in (0, 1]")
    flat = values.reshape(-1)
    count = max(1, math.ceil(flat.numel() * keep))
    chosen = torch.argsort(flat, descending=True, stable=True)[:count]
    result = torch.zeros_like(flat, dtype=torch.bool)
    result[chosen] = True
    return result.reshape_as(values)


class PhaseProbe(nn.Module):
    """A trained ReLU phase classifier plus frozen LINe masks and statistics.

    Train ``forward`` using normal demonstration phase pseudo-labels; then call
    ``fit_line`` on training episodes only. Refit after modifying learned weights.
    Inputs must be causal features already normalized by training-only statistics.
    No temporal operation in this module introduces access to future frames.

    Defaults keep 10% of neurons/weights, matching LINe's CIFAR-10 setting.
    They are configurable, not universal optima: CIFAR-100 keeps 90% of neurons
    and 10% of weights; ImageNet keeps 90% of both. Activation clipping is a
    configurable training-ID quantile here, rather than copying an image model's
    absolute 1.0/0.8 clipping values to a differently scaled robot probe.
    """

    def __init__(self, input_dim: int = 126, hidden_dim: int = 64, classes: int = 5):
        super().__init__()
        if min(input_dim, hidden_dim, classes) < 1:
            raise ValueError("input_dim, hidden_dim and classes must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.classes = classes
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, classes)
        self.register_buffer("line_importance", torch.zeros(classes, hidden_dim))
        self.register_buffer("line_activation_mask", torch.zeros(classes, hidden_dim, dtype=torch.bool))
        self.register_buffer("line_weight_mask", torch.zeros(classes, classes, hidden_dim, dtype=torch.bool))
        self.register_buffer("line_masked_weight", torch.zeros(classes, classes, hidden_dim))
        self.register_buffer("line_bias", torch.zeros(classes))
        self.register_buffer("line_clip", torch.tensor(float("inf")))
        self.register_buffer("line_class_counts", torch.zeros(classes, dtype=torch.long))
        self.register_buffer("line_reference_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("line_fit_settings", torch.zeros(4))
        self.register_buffer("line_fitted", torch.tensor(False))

    def features(self, x: Tensor) -> Tensor:
        return F.relu(self.hidden(x))

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))

    @torch.no_grad()
    def fit_line(
        self,
        x: Tensor,
        labels: Tensor,
        correct_only: bool = True,
        clip_quantile: float = .99,
        neuron_keep: float = .1,
        weight_keep: float = .1,
        *,
        fit_split: str = "train",
        batch_size: int = 8192,
    ) -> dict:
        """Estimate LINe buffers exclusively from supplied ID training examples.

        ``labels`` are explicit integer labels or hard ID teacher phases. If
        ``correct_only``, contributions use only correctly classified examples.
        Empty classes use the pooled selected-ID feature mean times |W_c|,
        recorded by zero ``line_class_counts``; no validation/OOD fallback is
        used. If none of the examples pass selection, fitting raises an error.

        The caller owns episode-level split isolation. ``fit_split`` additionally
        guards accidental calibration/validation calls; it cannot inspect the
        provenance of an otherwise unlabeled tensor.
        """
        if fit_split != "train":
            raise ValueError("LINe importance must be fitted on ID train episodes only")
        if x.ndim != 2 or x.shape[1] != self.input_dim or x.shape[0] == 0:
            raise ValueError(f"x must be a nonempty [N,{self.input_dim}] tensor")
        if labels.ndim != 1 or len(labels) != len(x):
            raise ValueError("labels must have shape [N]")
        if labels.is_floating_point() or labels.dtype == torch.bool:
            raise ValueError("labels must be integer class indices")
        if ((labels < 0) | (labels >= self.classes)).any():
            raise ValueError("label outside probe class range")
        if not 0 < clip_quantile <= 1:
            raise ValueError("clip_quantile must be in (0, 1]")
        if not 0 < neuron_keep <= 1 or not 0 < weight_keep <= 1:
            raise ValueError("keep fractions must be in (0, 1]")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        device = self.classifier.weight.device
        dtype = self.classifier.weight.dtype
        sums = torch.zeros(self.classes, self.hidden_dim, device=device, dtype=torch.float64)
        counts = torch.zeros(self.classes, device=device, dtype=torch.long)
        activations = []
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size].to(device=device, dtype=dtype)
            target = labels[start : start + batch_size].to(device=device, dtype=torch.long)
            if not torch.isfinite(batch).all():
                raise ValueError("ID reference inputs must be finite")
            h = self.features(batch)
            if not torch.isfinite(h).all():
                raise ValueError("probe activation is not finite")
            # Copy to CPU to bound GPU allocation; quantiles are estimated from
            # all provided training activations, independently of correctness.
            activations.append(h.float().cpu())
            selected = self.classifier(h).argmax(-1) == target if correct_only else torch.ones_like(target, dtype=torch.bool)
            sums.index_add_(0, target[selected], h[selected].to(torch.float64))
            counts.add_(torch.bincount(target[selected], minlength=self.classes))
        if int(counts.sum()) == 0:
            raise ValueError("no correctly predicted training reference examples")
        global_mean = sums.sum(0) / counts.sum()
        means = sums / counts.clamp_min(1).unsqueeze(-1)
        means[counts == 0] = global_mean
        # ReLU h>=0 makes E|h_j*W_cj| = E[h_j]*|W_cj| exactly.
        importance = means.to(dtype) * self.classifier.weight.detach().abs()
        # numpy avoids torch.quantile's 2**24 element limit. There is no random
        # sampling of reference activations, including for large ID datasets.
        import numpy as np

        clip = float(np.quantile(torch.cat(activations).numpy(), clip_quantile))
        # A zero-activation reference remains finite and deterministic. Clipping
        # to zero correctly yields a bias-only detector, not artificial evidence.
        masks_a = torch.stack([_largest_mask(row, neuron_keep) for row in importance])
        masks_w = torch.stack([
            _largest_mask(self.classifier.weight.detach() * row[None, :], weight_keep)
            for row in importance
        ])
        self.line_importance.copy_(importance)
        self.line_activation_mask.copy_(masks_a)
        self.line_weight_mask.copy_(masks_w)
        self.line_masked_weight.copy_(self.classifier.weight.detach()[None, :, :] * masks_w)
        self.line_bias.copy_(self.classifier.bias.detach())
        self.line_clip.fill_(clip)
        self.line_class_counts.copy_(counts)
        self.line_reference_count.fill_(len(x))
        self.line_fit_settings.copy_(torch.tensor([clip_quantile, neuron_keep, weight_keep, float(correct_only)], device=device))
        self.line_fitted.fill_(True)
        return {
            "fit_split": fit_split,
            "reference_examples": len(x),
            "selected_examples": int(counts.sum()),
            "class_counts": counts.cpu().tolist(),
            "fallback_classes": torch.nonzero(counts == 0).flatten().cpu().tolist(),
            "clip_value": clip,
            "clip_quantile": clip_quantile,
            "neuron_keep": neuron_keep,
            "weight_keep": weight_keep,
            "correct_only": bool(correct_only),
            "importance": "paper_eq4_mean_absolute_activation_times_logit_gradient",
            "score_orientation": "negative_logsumexp_larger_is_more_ood",
            "routing": "argmax_clipped_unpruned_logits_as_official_code",
        }

    def line_forward(self, x: Tensor) -> dict[str, Tensor]:
        """Score any leading shape [..., D], preserving it in all outputs.

        AP and WP route by the preliminary *clipped* unpruned prediction, as
        the official implementation does. ``predicted_class`` is the unchanged
        raw classification branch; ``routing_class`` makes this distinction
        visible when clipping changes the preliminary class.
        """
        if not self.line_fitted.item():
            raise RuntimeError("fit_line must be called before LINe inference")
        h = self.features(x)
        raw_logits = self.classifier(h)
        clipped = h.clamp(max=self.line_clip)
        routing_class = self.classifier(clipped).argmax(-1)
        leading = x.shape[:-1]
        flat_h = clipped.reshape(-1, self.hidden_dim)
        flat_route = routing_class.reshape(-1)
        masked_h = flat_h * self.line_activation_mask[flat_route]
        weight = self.line_masked_weight[flat_route]
        line_logits = (weight * masked_h[:, None, :]).sum(-1) + self.line_bias
        line_logits = line_logits.reshape(*leading, self.classes)
        return {
            "raw_logits": raw_logits,
            "line_logits": line_logits,
            "energy": -torch.logsumexp(line_logits, dim=-1),
            "predicted_class": raw_logits.argmax(-1),
            "routing_class": routing_class,
        }
