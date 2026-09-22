"""Causal boundary events from fused-latent changes, with fresh stage evidence.

Length is a weak TRAINING regularizer only. Runtime has no time-based advance.
The soft candidate can recede; the separately confirmed stage cannot regress.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ChangeConfig:
    dimension: int = 128
    hidden: int = 64
    windows: tuple = (8,32,128)
    nominal_frames: int = 200
    duration_weight: float = .01
    duration_lower_ratio: float = .5
    duration_upper_ratio: float = 2.5
    boundary_sigma: float = 20.
    confirmation_threshold: float = .85
    confirmation_frames: int = 15
    evidence_frames: int = 5
    release_threshold: float = .35
    rearm_frames: int = 5
    change_floor: float = .03
    anchor_floor: float = .15

    def __post_init__(self):
        self.windows=tuple(self.windows)
        if self.dimension<1 or self.hidden<1 or self.nominal_frames<1:raise ValueError('Positive dimensions/duration required')
        if len(self.windows)!=3 or tuple(sorted(self.windows))!=self.windows or self.windows[0]<1:raise ValueError('Three increasing causal windows required')
        if not 0<self.duration_lower_ratio<self.duration_upper_ratio or self.duration_weight<0:raise ValueError('Invalid weak duration band')
        if not 0<self.confirmation_threshold<1 or min(self.evidence_frames,self.confirmation_frames)<1:raise ValueError('Invalid confirmation')
        if not 0<=self.release_threshold<self.confirmation_threshold or self.rearm_frames<1:raise ValueError('Invalid evidence rearming')


class ChangeBlock(nn.Module):
    def __init__(self,dimension,dilation):
        super().__init__();self.padding=4*dilation
        self.conv=nn.Conv1d(dimension,dimension,5,dilation=dilation);self.norm=nn.LayerNorm(dimension)

    def forward(self,x,old=None):
        if old is None:old=x.new_zeros(x.shape[0],x.shape[2],self.padding)
        window=torch.cat((old,x.transpose(1,2)),-1)
        y=x+F.gelu(self.norm(self.conv(window).transpose(1,2)))
        return y,window[:,:,-self.padding:]


class LatentChangeHead(nn.Module):
    """Shared latent plus causal multiscale contrasts -> four boundary events.

    No episode length, elapsed fraction, phase teacher or duration feature is
    input to the network. All four events are trained, but runtime accepts
    only fresh evidence for the immediate successor of the committed stage.
    """
    def __init__(self,config):
        super().__init__();self.config=config;d,h=config.dimension,config.hidden
        self.current=nn.Linear(d,h)
        self.contrasts=nn.ModuleList([nn.Linear(d,h,bias=False) for _ in range(3)])
        self.change_size=nn.Linear(3,h,bias=False)
        self.norm=nn.LayerNorm(h);self.blocks=nn.ModuleList([ChangeBlock(h,k) for k in (1,4,16)])
        self.output=nn.Linear(h,4);nn.init.constant_(self.output.bias,-3.)

    def forward(self,z,cache=None):
        if z.ndim!=3 or z.shape[1]<1 or z.shape[2]!=self.config.dimension:raise ValueError('Expected nonempty [B,T,D] latent')
        cache={} if cache is None else cache;history=self.config.windows[-1]-1
        old=cache.get('latent',z[:,:1].expand(-1,history,-1))
        window=torch.cat((old,z),1);means=[]
        for width in self.config.windows:
            # Replicate only the first actually observed frame at reset.
            means.append(F.avg_pool1d(window.transpose(1,2),width,stride=1)[:,:,-z.shape[1]:].transpose(1,2))
        contrasts=[z-means[0],means[0]-means[1],means[1]-means[2]]
        sizes=torch.stack([v.float().square().mean(-1).clamp_min(1e-12).sqrt() for v in contrasts],-1)
        x=self.current(z)+sum(layer(v) for layer,v in zip(self.contrasts,contrasts))+self.change_size(sizes.to(z.dtype))
        x=F.gelu(self.norm(x));buffers=[]
        for block,buf in zip(self.blocks,cache.get('convs',[None]*3)):
            x,buf=block(x,buf);buffers.append(buf)
        logits=self.output(x).float()
        return dict(logits=logits,boundary_event_probs=logits.sigmoid(),change_sizes=sizes,
            recent_latent=means[0].float()),dict(latent=window[:,-history:],convs=buffers)

    @torch.inference_mode()
    def step(self,z,cache=None):
        out,cache=self.forward(z[:,None],cache)
        return {k:v[:,0] for k,v in out.items()},cache


def teacher_boundary_targets(phase,config):
    """Offline labels only: soft occupancy -> 0.5 crossings -> event windows."""
    cdf=(1-phase.cumsum(-1)[:,:4]).clamp(0,1).cummax(0).values
    centers=(cdf>=.5).float().argmax(0);exists=cdf[-1]>=.5
    t=torch.arange(len(phase),device=phase.device,dtype=torch.float32)
    target=torch.exp(-.5*((t[:,None]-centers[None])/config.boundary_sigma).square())
    target=target*exists
    return target,centers,exists


def duration_regularizer(logits,valid,config,observed=None):
    """Broad zero-penalty band around 200 frames; NOT exact 200-frame MSE.

    Expected event positions normalize over training trajectory for a LOSS,
    never for online inference. The final segment is right-censored: no upper
    penalty at recording end because recording length is not task completion.
    """
    weights=logits.sigmoid()*valid[:,:,None]
    t=torch.arange(logits.shape[1],device=logits.device,dtype=logits.dtype)
    boundaries=(weights*t[None,:,None]).sum(1)/weights.sum(1).clamp_min(1e-6)
    durations=torch.diff(torch.cat((boundaries.new_zeros(boundaries.shape[0],1),boundaries),1),dim=1)
    ratio=durations/config.nominal_frames
    low=F.relu(config.duration_lower_ratio-ratio);high=F.relu(ratio-config.duration_upper_ratio)
    penalty=.5*(F.smooth_l1_loss(low,torch.zeros_like(low),beta=.25,reduction='none')+
        F.smooth_l1_loss(high,torch.zeros_like(high),beta=.25,reduction='none'))
    if observed is None:observed=torch.ones_like(penalty,dtype=torch.bool)
    return (penalty*observed).sum()/observed.sum().clamp_min(1),durations


class FreshStageMemory:
    """At most one adjacent advance, with current evidence and a fresh anchor.

    Later-stage scores are not banked. OOD clears confirmation evidence, but
    observations still flow. No boundary can be accepted solely by age.
    """
    def __init__(self,config):self.config=config;self.reset()

    def reset(self):
        self.phase=1;self.count=0;self.age=0;self.frame=-1;self.anchor=None;self.armed=False;self.release_count=0
        self.recent=deque(maxlen=self.config.evidence_frames)

    def step(self,event_probs,change_sizes,recent_latent,alarm=False):
        probs=np.asarray(event_probs,dtype=np.float32);z=np.asarray(recent_latent,dtype=np.float32)
        if probs.shape!=(4,) or not np.isfinite(probs).all() or (probs<0).any() or (probs>1).any():raise ValueError('Invalid boundary probabilities')
        if z.shape!=(self.config.dimension,) or not np.isfinite(z).all():raise ValueError('Invalid recent latent')
        self.frame+=1;self.age+=1
        if self.anchor is None:self.anchor=z.copy()
        distance=float(np.sqrt(np.mean((z-self.anchor)**2)))
        change=float(max(np.asarray(change_sizes)[1:]))
        support=change>=self.config.change_floor and distance>=self.config.anchor_floor
        raw=float(probs[self.phase-1]) if self.phase<5 else 0.
        # A permanently high future-boundary score is not a new event. The
        # newly entered stage must first observe low evidence, then a rise.
        if not self.armed:
            self.release_count=self.release_count+1 if raw<self.config.release_threshold else 0
            if self.release_count>=self.config.rearm_frames:self.armed=True
        candidate=raw if self.phase<5 and support else 0.
        self.recent.append(candidate if self.armed else 0.)
        sustained=min(self.recent) if len(self.recent)==self.config.evidence_frames else 0.
        if alarm:
            self.count=0;self.recent.clear()
        elif self.phase<5:
            self.count=self.count+1 if sustained>=self.config.confirmation_threshold else 0
        advanced=False;previous=self.phase
        if self.count>=self.config.confirmation_frames:
            self.phase+=1;self.count=0;self.age=0;self.anchor=z.copy();self.recent.clear();advanced=True
            self.armed=False;self.release_count=0
        q=np.zeros(5,dtype=np.float32)
        if self.phase==5 or advanced:q[self.phase-1]=1.
        else:q[self.phase-1]=1.-candidate;q[self.phase]=candidate
        return dict(phase_probs=q,boundary_cdf=(1-q.cumsum())[:4],confirmed_phase=self.phase,
            accepted_phase=0 if alarm else self.phase,boundary_event_probs=probs.copy(),
            boundary_evidence=candidate,sustained_boundary_evidence=sustained,confirmation_count=self.count,
            latent_change=change,anchor_distance=distance,stage_age=self.age,
            transition_armed=self.armed,
            advanced=advanced,previous_phase=previous)
