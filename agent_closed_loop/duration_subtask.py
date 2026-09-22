"""V7: causal ordered segmentation with an explicit default-duration prior.

CompILE Eq.4 motivates a Poisson length preference; Appendix A.5 explicitly
allows regularizing soft segment occupancy. This is a causal distillation
adaptation, not a reimplementation of the paper's offline VAE.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .change_subtask import ChangeConfig, LatentChangeHead


@dataclass
class DurationConfig:
    dimension: int = 128
    hidden: int = 64
    nominal_frames: int = 200
    duration_weight: float = .05
    prior_strength: float = 2.
    prior_clip: float = 4.
    evidence_frames: int = 5
    confirmation_threshold: float = .5
    change_floor: float = .03

    def __post_init__(self):
        if min(self.dimension, self.hidden, self.nominal_frames, self.evidence_frames) < 1:
            raise ValueError('Positive dimensions and windows required')
        if self.duration_weight < 0 or self.prior_strength < 0 or self.prior_clip <= 0:
            raise ValueError('Invalid duration prior')
        if not .5<=self.confirmation_threshold<1:
            raise ValueError('Confirmation cutoff must be in [0.5,1)')


def duration_penalty(lengths, nominal_frames=200):
    """KL(Poisson(d) || Poisson(L))/L; unique minimum at d=L, no dead band."""
    ratio=(lengths/nominal_frames).clamp_min(1e-6)
    return ratio*ratio.log()-ratio+1


def ordered_duration(logits, support, config, cache=None, pause=None):
    """Same causal recurrence in full-sequence training and streaming inference.

    A boundary is eligible only after its predecessor crossed the cutoff on a
    PREVIOUS frame. Future boundaries cannot bank confidence. The prior uses
    observed stage age, never episode length. log(age/L) is the negative log
    ratio of consecutive Poisson masses. It is bounded and evidence-gated.
    CDFs only increase: the confirmed quantile and expected phase never regress.
    OOD pauses both the posterior and age, while clearing persistence evidence.
    """
    b,t,k=logits.shape
    if k!=4:raise ValueError('Four boundaries required')
    cache={} if cache is None else cache
    previous=cache.get('cdf',logits.new_zeros(b,4))
    ages=cache.get('ages',logits.new_zeros(b,4))
    if pause is None:pause=torch.zeros((b,t),dtype=torch.bool,device=logits.device)
    allowed=(~pause).to(logits.dtype)
    columns=[];new_ages=[];buffers=[];candidates=[];biases=[]
    width=config.evidence_frames
    old_buffers=cache.get('evidence',[None]*4)
    for j in range(4):
        upstream=logits.new_ones(b,t) if j==0 else torch.cat((previous[:,j-1:j],columns[j-1][:,:-1]),1)
        eligible=(upstream>=config.confirmation_threshold).to(logits.dtype)*allowed
        age=ages[:,j:j+1]+eligible.cumsum(1)
        bias=(config.prior_strength*(age.clamp_min(1)/config.nominal_frames).log()).clamp(-config.prior_clip,config.prior_clip)
        raw=(logits[:,:,j]+bias).sigmoid()
        # Neither age nor a weak static score can independently advance a stage.
        raw=raw*eligible*support*(logits[:,:,j].sigmoid()>=.5)
        old=old_buffers[j]
        if old is None:old=raw.new_zeros(b,width-1)
        window=torch.cat((old,raw),1)
        sustained=window.unfold(1,width,1).amin(-1)
        value=torch.minimum(sustained,upstream)
        c=torch.cummax(torch.cat((previous[:,j:j+1],value),1),1).values[:,1:]
        columns.append(c);new_ages.append(age[:,-1]);buffers.append(window[:,-(width-1):] if width>1 else window[:,:0])
        candidates.append(sustained);biases.append(bias)
    cdf=torch.stack(columns,-1)
    q=torch.cat((1-cdf[:,:,:1],cdf[:,:,:-1]-cdf[:,:,1:],cdf[:,:,-1:]),-1)
    state=dict(cdf=cdf[:,-1],ages=torch.stack(new_ages,-1),evidence=buffers)
    return dict(phase_probs=q,boundary_cdf=cdf,boundary_evidence=torch.stack(candidates,-1),
                duration_log_bias=torch.stack(biases,-1),confirmed_phase=1+(cdf>=config.confirmation_threshold).sum(-1)),state


class DurationOrderedHead(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        self.features=LatentChangeHead(ChangeConfig(dimension=config.dimension,hidden=config.hidden))

    def forward(self,z,cache=None,pause=None):
        cache={} if cache is None else cache
        features,feature_cache=self.features(z,cache.get('features'))
        support=(features['change_sizes'][...,1:].amax(-1)>=self.config.change_floor).to(z.dtype)
        out,memory=ordered_duration(features['logits'],support,self.config,cache.get('memory'),pause)
        out.update(logits=features['logits'],observation_boundary_probs=features['boundary_event_probs'],
                   change_sizes=features['change_sizes'])
        return out,dict(features=feature_cache,memory=memory)

    @torch.inference_mode()
    def step(self,z,cache=None,pause=None):
        if pause is not None:pause=torch.as_tensor(pause,device=z.device,dtype=torch.bool).reshape(z.shape[0],1)
        out,cache=self.forward(z[:,None],cache,pause)
        return {k:v[:,0] for k,v in out.items()},cache


def duration_objective(out,teacher,valid,config):
    q=out['phase_probs'].clamp_min(1e-6)
    target=(1-teacher.cumsum(-1)[...,:4]).clamp(0,1)
    kl=(teacher*(teacher.clamp_min(1e-8).log()-q.log())).sum(-1)
    bce=F.binary_cross_entropy_with_logits(out['logits'],target,reduction='none').mean(-1)
    cdf=(out['boundary_cdf']-target).abs().mean(-1)
    fit=(((kl+.5*bce+.5*cdf)*valid).sum(1)/valid.sum(1).clamp_min(1)).mean()
    mass=(out['phase_probs']*valid[:,:,None]).sum(1)[:,:4]
    observed=(target*valid[:,:,None]).amax(1)>=.5
    duration=(duration_penalty(mass,config.nominal_frames)*observed).sum()/observed.sum().clamp_min(1)
    # Raw recognition may fluctuate; discourage backward crossing evidence.
    regress=F.relu(out['observation_boundary_probs'][:,:-1]-out['observation_boundary_probs'][:,1:]).square().mean(-1)
    pairs=valid[:,:-1]&valid[:,1:]
    regression=(regress*pairs).sum()/pairs.sum().clamp_min(1)
    loss=fit+config.duration_weight*duration+.1*regression
    return loss,dict(fit=float(fit.detach()),duration_poisson_kl=float(duration.detach()),
                    weighted_duration=float((config.duration_weight*duration).detach()),regression=float(regression.detach()))
