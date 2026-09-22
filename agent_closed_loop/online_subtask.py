"""Causal learned boundary evidence, ordered soft memory and committed stage.

This is NOT argmax smoothing of a frame classifier. Four independently learned
boundary-crossing confidences are trained through the memory recurrence. Each
update transports mass at most one stage. A constant weak observation cannot
accumulate hazard forever. No episode length, teacher or future input is used.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from .recovery import TemporalRecoveryHead
from .uncertainty_ood import SingleRiskHead


@dataclass
class OrderedConfig:
    dimension: int = 128
    hidden: int = 64
    evidence_frames: int = 5
    confirmation_threshold: float = .8
    confirmation_frames: int = 15


def ordered_memory(evidence, previous=None):
    """[B,T,4] evidence -> boundary CDF, phase probabilities, final memory.

    C[t,k] = max(C[t-1,k], min(e[t,k], C[t-1,k-1])), C[t,-1]=1.
    Thus delta C[t,k] <= q[t-1,k]: no mass can skip a stage in one step.
    Vectorized over time with four prefix scans; optional state spans chunks.
    """
    if evidence.ndim != 3 or evidence.shape[-1] != 4:
        raise ValueError('Expected boundary evidence [B,T,4]')
    if previous is None:previous=evidence.new_zeros(evidence.shape[0],4)
    columns=[]
    for k in range(4):
        upstream=torch.ones_like(evidence[:,:,k]) if k==0 else torch.cat((previous[:,k-1:k],columns[k-1][:,:-1]),1)
        candidate=torch.minimum(evidence[:,:,k],upstream)
        column=torch.cummax(torch.cat((previous[:,k:k+1],candidate),1),1).values[:,1:]
        columns.append(column)
    c=torch.stack(columns,-1)
    q=torch.cat((1-c[:,:,:1],c[:,:,:-1]-c[:,:,1:],c[:,:,-1:]),-1)
    return c,q,c[:,-1]


class OrderedSubtaskHead(nn.Module):
    def __init__(self,config: OrderedConfig):
        super().__init__();self.config=config
        self.network=TemporalRecoveryHead(config.dimension,config.hidden,dropout=0.)
        self.network.output=nn.Linear(config.hidden,4)
        nn.init.constant_(self.network.output.bias,-3.)

    def forward(self,z,memory=None):
        # Cache contains differentiable conv histories AND accumulated CDF.
        memory={} if memory is None else memory
        x=F.gelu(self.network.input(z));histories=[]
        for block,old in zip(self.network.blocks,memory.get('convs',[None]*3)):
            xt=x.transpose(1,2)
            if old is None:old=x.new_zeros(x.shape[0],x.shape[-1],block.padding)
            window=torch.cat((old,xt),-1)
            y=block.conv(window).transpose(1,2)
            x=x+F.gelu(block.norm(y));histories.append(window[:,:,-block.padding:])
        logits=self.network.output(x).float();raw=logits.sigmoid()
        width=self.config.evidence_frames
        old=memory.get('evidence',raw.new_zeros(raw.shape[0],width-1,4))
        window=torch.cat((old,raw),1)
        sustained=window.unfold(1,width,1).amin(-1)
        c,q,last=ordered_memory(sustained,memory.get('cdf'))
        return dict(phase_probs=q,boundary_cdf=c,boundary_evidence=raw,logits=logits),dict(
            convs=histories,evidence=window[:,-(width-1):] if width>1 else window[:,:0],cdf=last)

    @torch.inference_mode()
    def step(self,z,memory=None):
        out,cache=self.forward(z[:,None],memory)
        return {k:v[:,0] for k,v in out.items()},cache


class StageConfirmation:
    """A task starts at P1. Rejection pauses confirmation, not observation.

    confirmed_phase is historical progress; accepted_phase=0 on alarm is a
    rejection, never a regression to a different subtask. Reset explicitly on
    a new task or externally authorized re-localization.
    """
    def __init__(self,threshold=.8,frames=15):
        if not 0<threshold<1 or frames<1:raise ValueError('Invalid confirmation rule')
        self.threshold=threshold;self.frames=frames;self.reset()

    def reset(self):self.phase=1;self.count=0

    def step(self,cdf,alarm=False):
        if alarm:self.count=0
        elif self.phase<5:
            self.count=self.count+1 if float(cdf[self.phase-1])>=self.threshold else 0
            if self.count>=self.frames:self.phase+=1;self.count=0
        return self.phase,0 if alarm else self.phase


class OnlineRiskHead(nn.Module):
    """One risk output; learned phase context is added to the shared latent.

    Instantaneous LINe/dropout/loop evidence remains available independently
    of ordered confidence. The model never treats ordered confidence as proof
    that the robot is in domain or recoverable.
    """
    def __init__(self,dimension,hidden=32):
        super().__init__();self.dimension=dimension;self.hidden=hidden
        self.phase_context=nn.Linear(9,dimension,bias=False)
        self.risk=SingleRiskHead(dimension,hidden)

    def forward(self,z,evidence,phase):
        return self.risk(z+self.phase_context(phase),evidence)

    @torch.inference_mode()
    def step(self,z,evidence,phase,cache=None):
        return self.risk.step(z+self.phase_context(phase),evidence,cache)


def phase_context(out):return torch.cat((out['phase_probs'],out['boundary_cdf']),-1)


def detached(memory):
    return {k:[x.detach() for x in v] if isinstance(v,list) else v.detach() for k,v in memory.items()}
