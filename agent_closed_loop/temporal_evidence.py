"""Bounded history evidence; observed motion, not raw policy command chunks."""
import numpy as np
from .recovery_features import _rolling_sum

EVIDENCE_NAMES=['line_energy','readout_disagreement','repeat_without_progress','active_repeat','progress','period_scaled','history_available']


def stable_cycles(s,v):
    """Three-block repetition; shortest period among numerically tied scores.

    Global vs bounded rolling sums differ at floating roundoff. argmax of
    perfect repeated cycles can otherwise jump from 1s to 4s arbitrarily.
    A fixed 1e-6 tie tolerance is negligible relative to measured evidence,
    and has the interpretable preference for the shortest matching period.
    """
    ds=np.concatenate((np.zeros_like(s[:1]),np.diff(s,axis=0)))
    n=len(s);t=np.arange(n);energy=np.mean(ds*ds,axis=1);periods=(15,20,30,45,60,90,120);scores=[]
    dv=np.concatenate((np.zeros_like(v[:1]),np.diff(v,axis=0)))
    speed_v=np.sqrt(np.mean(dv*dv,axis=1))
    for lag in periods:
        dot=np.zeros(n);dot[lag:]=np.mean(ds[lag:]*ds[:-lag],axis=1)
        d=_rolling_sum(dot,lag);e=_rolling_sum(energy,lag);previous=e[np.maximum(0,t-lag)]
        match=np.clip(2*d/np.maximum(e+previous,1e-10),0,1);match[t<2*lag]=0
        consistent=np.minimum(match,match[np.maximum(0,t-lag)]);consistent[t<3*lag]=0
        speed=np.sqrt(np.maximum(e,0)/lag)*30;repeat=consistent*speed/(speed+.05)
        net=np.sqrt(np.mean((v-v[np.maximum(0,t-3*lag)])**2,axis=1))
        path=_rolling_sum(speed_v,3*lag);efficiency=np.clip(net/np.maximum(path,1e-10),0,1)
        scores.append(np.stack((repeat,repeat*(1-efficiency),efficiency),axis=1))
    cube=np.stack(scores,axis=1);loop=cube[:,:,1]
    which=(loop>=loop.max(1,keepdims=True)-1e-6).argmax(1)
    return np.column_stack((cube[np.arange(n),which],np.asarray(periods)[which]/120)).astype('float32')


def time_evidence(states,latent,latent_scale,visual_importance):
    # Fixed normal-training neuron weights throughout every comparison window.
    z=np.asarray(latent,dtype='float64')/np.maximum(np.asarray(latent_scale),.05)
    weight=np.asarray(visual_importance,dtype='float64')
    weight=weight/max(weight.mean(),1e-8)
    z=z*np.sqrt(weight)
    s=np.asarray(states,dtype='float64')
    cycles=stable_cycles(s,z)
    # Zero period if even the shortest three blocks have not yet completed.
    available=(np.arange(len(s))>=45).astype('float32')
    return np.column_stack((cycles[:,1],cycles[:,0],cycles[:,2],cycles[:,3]*available,available)).astype('float32')


class HistoryEvidence:
    def __init__(self,latent_scale,visual_importance):
        self.scale=latent_scale;self.importance=visual_importance;self.reset()

    def reset(self):self.states=[];self.latents=[]

    def step(self,state,latent):
        self.states.append(np.asarray(state));self.latents.append(np.asarray(latent))
        self.states=self.states[-361:];self.latents=self.latents[-361:]
        return time_evidence(np.asarray(self.states),np.asarray(self.latents),self.scale,self.importance)[-1]
