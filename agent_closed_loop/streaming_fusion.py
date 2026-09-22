"""Shared configurable-width visual/motion encoding with exact causal caches."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FusionConfig:
    dim: int = 128
    heads: int = 4
    window: int = 120
    dilations: tuple = (1, 2, 4, 8)
    kernel: int = 3
    phase_dropout: float = .1

    def __post_init__(self):
        self.dilations = tuple(self.dilations)
        if self.dim % self.heads or self.window < 1 or self.kernel != 3:
            raise ValueError('Invalid width, window or kernel')

    @property
    def warmup(self):
        return self.window - 1 + (self.kernel - 1) * sum(self.dilations)


class CausalBlock(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(dim, dim, 3, dilation=dilation)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, valid):
        y = self.conv(F.pad(x.transpose(1, 2), (2*self.dilation, 0))).transpose(1, 2)
        return (x + F.gelu(self.norm(y))) * valid[..., None]

    def step(self, x, cache):
        if cache is None:
            cache = x.new_zeros(x.shape[0], x.shape[-1], 2*self.dilation)
        history = torch.cat((cache, x[..., None]), dim=-1)
        y = self.conv(history).squeeze(-1)
        return x + F.gelu(self.norm(y)), history[:, :, 1:]


class SharedFusion(nn.Module):
    """48-D raw visual features enter K/V projections directly; no raw concat."""
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        d = config.dim
        self.state = nn.Sequential(nn.Linear(14, d), nn.GELU(), nn.LayerNorm(d))
        self.motion = nn.Sequential(nn.Linear(14, d), nn.GELU(), nn.LayerNorm(d))
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(48, d)
        self.v = nn.Linear(48, d)
        self.out = nn.Linear(d, d)
        self.fusion_gate = nn.Parameter(torch.full((d,), -1.))
        self.visual_gate = nn.Parameter(torch.zeros(d))
        self.relative_bias = nn.Parameter(torch.zeros(config.heads, config.window))
        self.blocks = nn.ModuleList([CausalBlock(d, dilation) for dilation in config.dilations])
        self.latent = nn.Linear(d, d)
        self.phase = nn.Linear(d, 5)
        self.visual_decoder = nn.Linear(d, 48)
        self.state_decoder = nn.Linear(d, 14)

    def _heads(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.config.heads, -1).transpose(1, 2)

    def forward(self, states, deltas, visual, valid=None, motion_keep=None):
        if valid is None:
            valid = torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
        u = self.state(states) + self.motion(deltas)
        if motion_keep is not None:
            u = u * motion_keep[:, None, None]
        q, k, v = self._heads(self.q(u)), self._heads(self.k(visual)), self._heads(self.v(visual))
        t = states.shape[1]
        pos = torch.arange(t, device=states.device)
        lag = pos[:, None] - pos[None, :]
        allowed = (lag >= 0) & (lag < self.config.window)
        bias = self.relative_bias[:, lag.clamp(0, self.config.window-1)].to(q.dtype)
        mask = bias[None].expand(states.shape[0], -1, -1, -1).masked_fill(
            ~(allowed[None, None] & valid[:, None, None, :]), float('-inf'))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        y = y.transpose(1, 2).reshape_as(u)
        current_visual = v.transpose(1, 2).reshape_as(u)
        x = (u + self.fusion_gate.sigmoid()*self.out(y) + self.visual_gate.sigmoid()*current_visual) * valid[..., None]
        for block in self.blocks:
            x = block(x, valid)
        z = F.relu(self.latent(x)) * valid[..., None]
        return z

    def readout(self, z, *, sample_dropout=False):
        return self.phase(F.dropout(z, self.config.phase_dropout, training=sample_dropout))

    @torch.inference_mode()
    def step(self, states, deltas, visual, cache=None):
        """Normalized [B,D] inputs. A fresh episode starts with cache=None."""
        cache = {} if cache is None else cache
        u = self.state(states) + self.motion(deltas)
        q = self._heads(self.q(u[:, None]))
        k = self._heads(self.k(visual[:, None]))
        v = self._heads(self.v(visual[:, None]))
        current_v = v[:, :, 0].reshape_as(u)
        if 'k' in cache:
            k = torch.cat((cache['k'], k), dim=2)[:, :, -self.config.window:]
            v = torch.cat((cache['v'], v), dim=2)[:, :, -self.config.window:]
        lag = torch.arange(k.shape[2]-1, -1, -1, device=k.device)
        bias = self.relative_bias[:, lag].to(q.dtype)[None, :, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.)
        y = y[:, :, 0].reshape_as(u)
        x = u + self.fusion_gate.sigmoid()*self.out(y) + self.visual_gate.sigmoid()*current_v
        histories = []
        old = cache.get('blocks', [None]*len(self.blocks))
        for block, history in zip(self.blocks, old):
            x, history = block.step(x, history)
            histories.append(history)
        z = F.relu(self.latent(x))
        return z, dict(k=k, v=v, blocks=histories)

    def serializable_config(self):
        return asdict(self.config)


def readout_disagreement(z, classifier, frames, episode_seed=0, samples=4, dropout=.1):
    """Deterministic frame/sample/channel hashing; prefix and batch independent."""
    if samples < 1 or not 0 < dropout < 1:
        raise ValueError('Invalid readout sampling configuration')
    channels = torch.arange(z.shape[-1], device=z.device, dtype=torch.int64)
    frames = torch.as_tensor(frames, device=z.device, dtype=torch.int64)
    probs = []
    for sample in range(samples):
        bits = (frames[..., None]*73856093 + channels*19349663 + (sample+1)*83492791 + episode_seed) & 0x7fffffff
        bits = ((bits ^ (bits >> 13))*1274126177) & 0x7fffffff
        keep = (bits.double() / 2147483648. >= dropout).to(z.dtype)
        probs.append(classifier(z*keep/(1-dropout)).float().softmax(-1))
    p = torch.stack(probs)
    entropy = lambda x: -(x*x.clamp_min(1e-9).log()).sum(-1)
    return (entropy(p.mean(0))-entropy(p).mean(0)).clamp_min(0)
