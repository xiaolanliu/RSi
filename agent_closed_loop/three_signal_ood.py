"""Three observable OOD signals; no learned latent bypass or uncertainty head."""
import numpy as np
import torch
from torch import nn


SIGNAL_NAMES = ['projection_distance', 'repeat_without_progress', 'line_energy']


class ProjectionMemory(nn.Module):
    """RMS nearest-neighbour distance in a normal-only PCA projection.

    The reference bank is frozen. Never insert the current episode: doing so
    would let an unfamiliar stationary state become its own normal reference.
    Training statistics exclude the query episode's entire reference group.
    """
    def __init__(self, center, projection, reference, reference_episode):
        super().__init__()
        for name, value in [('center', center), ('projection', projection),
                            ('reference', reference), ('reference_episode', reference_episode)]:
            self.register_buffer(name, torch.as_tensor(value).clone())

    def forward(self, standardized_latent, exclude_episode=None):
        q = (standardized_latent - self.center) @ self.projection
        # Direct differences for online inference avoid subtracting large,
        # nearly equal squared norms. Batch fitting uses the same expression.
        d = torch.cdist(q.reshape(-1, q.shape[-1]), self.reference,
                        compute_mode='donot_use_mm_for_euclid_dist')
        if exclude_episode is not None:
            groups = torch.as_tensor(exclude_episode, device=q.device).reshape(-1)
            d = d.masked_fill(groups[:, None] == self.reference_episode[None], float('inf'))
        return (d.min(-1).values / q.shape[-1]**.5).reshape(q.shape[:-1])


class ThreeSignalRisk(nn.Module):
    """Normal empirical tail ranks; a monotone maximum with no latent bypass.

    The optional median/scale arguments only load the archived first prototype.
    Current bundles provide sorted normal-training reference values instead.
    """
    def __init__(self, reference=None, median=None, scale=None, weights=None):
        super().__init__()
        weights=torch.ones(3) if weights is None else torch.as_tensor(weights,dtype=torch.float32).clone()
        if weights.shape!=(3,) or not torch.isfinite(weights).all() or not (weights>0).all():
            raise ValueError('Expected three finite positive criterion weights')
        self.register_buffer('weights',weights)
        if reference is not None:
            reference=torch.as_tensor(reference).clone()
            if reference.ndim!=2 or reference.shape[0]!=3 or reference.shape[1]<2:
                raise ValueError('Expected three sorted normal reference distributions')
            if not torch.isfinite(reference).all() or (reference[:,1:]<reference[:,:-1]).any():
                raise ValueError('Reference must be finite and sorted')
            self.register_buffer('reference',reference)
            return
        self.register_buffer('median', torch.as_tensor(median).clone())
        self.register_buffer('scale', torch.as_tensor(scale).clone())
        if self.median.shape != (3,) or self.scale.shape != (3,) or not (self.scale > 0).all():
            raise ValueError('Expected exactly three finite positive scales')
        if not torch.isfinite(self.median).all() or not torch.isfinite(self.scale).all():
            raise ValueError('Nonfinite signal normalization')

    def components(self, evidence):
        if evidence.shape[-1] != 3:
            raise ValueError('Only distance, repetition and LINe are accepted')
        if hasattr(self,'reference'):
            values=evidence.reshape(-1,3).T.contiguous()
            rank=torch.searchsorted(self.reference,values,right=False).T.reshape(evidence.shape)
            # (1 + number of reference values >= query)/(N + 1). Conservative
            # ties, bounded finite tails even for values above all references.
            p=(self.reference.shape[1]+1-rank.float())/(self.reference.shape[1]+1)
            return -p.log10()
        return ((evidence - self.median) / self.scale).clamp_min(0)

    def percentiles(self, evidence):
        return 1-torch.pow(10.,-self.components(evidence))

    def forward(self, evidence):
        return (self.components(evidence)*self.weights).max(-1).values


def calibration_tail(score, reference):
    """Normal calibration episode-maximum tail rank, not a failure probability."""
    reference=np.asarray(reference)
    rank=np.searchsorted(reference,score,side='left')
    return (len(reference)+1-rank)/(len(reference)+1)


def sustained_trace(raw, preprocessing):
    """Causal rolling mean followed by a minimum over the confirmation window."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError('Expected nonempty finite nonnegative scores')
    width, dwell = preprocessing['smooth_frames'], preprocessing['persist_frames']
    t = np.arange(len(raw)); start = np.maximum(0, t + 1 - width)
    sums = np.r_[0., np.cumsum(raw)]
    mean = (sums[t + 1] - sums[start]) / (t + 1 - start)
    score = np.zeros_like(mean)
    for i in range(dwell - 1, len(raw)):
        score[i] = mean[i-dwell+1:i+1].min()
    score[:preprocessing['minimum_history']] = 0
    return mean.astype('float32'), score.astype('float32')
