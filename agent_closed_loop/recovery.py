"""Causal temporal intervention candidate detector, supervised at episode level.

No output certifies irrecoverability. LINe supplies novelty evidence; motion
history supplies repetition/progress evidence. Three positive trajectories are
weak positive bags, never all-positive frame labels.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .recovery_features import causal_behavior_features


class _CausalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.padding = 4 * dilation
        self.conv = nn.Conv1d(width, width, 5, dilation=dilation)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv(F.pad(x.transpose(1,2), (self.padding,0))).transpose(1,2)
        return x + self.dropout(F.gelu(self.norm(h)))


class TemporalRecoveryHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=.1):
        super().__init__()
        self.input_dim = input_dim; self.hidden_dim = hidden_dim; self.dropout = dropout
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[_CausalBlock(hidden_dim, d, dropout) for d in (1,4,16)])
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError('Temporal head input must be [B,T,input_dim]')
        return self.output(self.blocks(F.gelu(self.input(x)))).squeeze(-1)


def detector_features(arrays, phase_probs, line_energy, raw_energy, scales):
    """All inputs at t are available by t; no teacher masks are accessed."""
    states = arrays['states'].numpy(); actions = arrays['actions'].numpy()
    visuals = arrays['features'][:, -48:].numpy()
    behavior, names = causal_behavior_features(states, actions, visuals,
        state_scale=scales['state_scale'], action_scale=scales['action_scale'],
        visual_scale=scales['visual_scale'], fps=arrays['metadata']['fps'])
    probs = np.asarray(phase_probs, dtype=np.float32)
    entropy = -(probs * np.log(np.clip(probs,1e-10,1))).sum(1) / math.log(probs.shape[1])
    columns = [np.asarray(line_energy), np.asarray(raw_energy), arrays['baseline_distance'].numpy(),
               probs.max(1), entropy, *probs.T, *behavior.T]
    feature_names = ['line_energy','raw_energy','normal_support_distance','phase_confidence','phase_entropy']
    feature_names += [f'phase_P{i+1}' for i in range(5)] + names
    expected = probs @ np.arange(1,6,dtype=np.float32)
    time = np.arange(len(probs))
    for seconds in (1,2,4):
        lag = int(round(seconds * arrays['metadata']['fps']))
        previous = np.maximum(0,time-lag)
        change = (expected-expected[previous]) / 4
        columns += [change, np.maximum(0,-change), (np.argmax(probs,1)==np.argmax(probs[previous],1)).astype(np.float32)]
        feature_names += [f'phase_progress_{seconds}s',f'phase_backtrack_{seconds}s',f'phase_unchanged_{seconds}s']
    features = np.stack(columns,axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('Detector features contain nonfinite values')
    repeat = np.max(behavior[:,[i for i,n in enumerate(names) if n.startswith('action_repeat_similarity_')]],axis=1)
    loop = np.max(behavior[:,[i for i,n in enumerate(names) if n.startswith('repeat_without_progress_')]],axis=1)
    progress = .5 * (behavior[:,names.index('state_path_efficiency_1s')] + behavior[:,names.index('visual_path_efficiency_1s')])
    return features, feature_names, dict(repeat_score=repeat,loop_score=loop,progress_score=progress)


def intervention_trace(probabilities, smooth_frames=30, persist_frames=15):
    """Trailing mean then trailing minimum: causal 1 s smoothing, .5 s dwell."""
    p = np.asarray(probabilities,dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or not 0 <= p.min() <= p.max() <= 1:
        raise ValueError('Expected nonempty finite probabilities in [0,1]')
    if smooth_frames < 1 or persist_frames < 1:
        raise ValueError('Smoothing and persistence lengths must be positive')
    t = np.arange(len(p)); prefix = np.r_[0.,np.cumsum(p)]
    start = np.maximum(0,t+1-smooth_frames)
    smooth = (prefix[t+1]-prefix[start])/(t+1-start)
    sustained = np.zeros_like(smooth)
    for i in range(persist_frames-1,len(p)):
        sustained[i] = smooth[i-persist_frames+1:i+1].min()
    return sustained.astype(np.float32)


def episode_metrics(positive_scores, negative_scores):
    positive = np.asarray(positive_scores,dtype=np.float64)
    negative = np.asarray(negative_scores,dtype=np.float64)
    if not len(positive) or not len(negative):
        raise ValueError('Episode AUROC requires both positive and negative episodes')
    delta = positive[:,None]-negative[None,:]
    return dict(auroc=float(((delta>0)+.5*(delta==0)).mean()),
                positive_episodes=len(positive),negative_episodes=len(negative))


def weak_episode_loss(logits, valid_mask, is_positive, top_fraction=.1):
    """Normal bags supervise all valid frames; positive bags only top-k MIL."""
    losses=[]; bag_logits=[]
    for row,valid,positive in zip(logits,valid_mask,is_positive):
        values=row[valid]
        if not len(values):
            raise ValueError('Empty episode in weak supervision')
        k=max(1,math.ceil(top_fraction*len(values)))
        pooled=values.topk(k).values.mean(); bag_logits.append(pooled)
        if bool(positive):
            loss=F.softplus(-pooled) + .01*values.sigmoid().mean()
        else:
            # all normal frames + hard negative top-k; per-episode weighting
            loss=.5*F.softplus(values).mean()+.5*F.softplus(pooled)
        if len(values)>1:
            loss=loss+.05*(values.sigmoid()[1:]-values.sigmoid()[:-1]).square().mean()
        losses.append(loss)
    return torch.stack(losses).mean(), torch.stack(bag_logits)
