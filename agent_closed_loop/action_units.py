"""Causal per-arm motion units, anchored by observed gripper transitions.

No commanded actions are fabricated from observed state. This module measures
executed kinematic repetition; action-conditioned response needs separate logs.
"""
from collections import deque
from dataclasses import dataclass,asdict
import numpy as np

PACKED_TO_ARMS=np.array([0,1,2,3,4,5,12,6,7,8,9,10,11,13])
ARMS_TO_PACKED=np.argsort(PACKED_TO_ARMS)


def repetition_strength(periodic,units,reference):
    """Two forms of the SAME repetition criterion, on a common tail scale.

    Gripper-anchored units cover changing tempo and asynchronous arms. Legacy
    motion recurrence covers motion with no gripper transitions. No new top-
    level criterion is introduced; the combined signal is calibrated again.
    """
    values=[]
    for name,x in [('periodic',periodic),('units',units)]:
        ref=np.asarray(reference[name])
        # Streaming arithmetic is float64, persisted training/cached evidence
        # is float32. Rank at the reference precision so conservative ties
        # agree between scalar online queries and batched offline queries.
        x=np.asarray(x,dtype=ref.dtype)
        rank=np.searchsorted(ref,x,side='left')
        values.append(-np.log10((len(ref)+1-rank)/(len(ref)+1)))
    return np.maximum(*values).astype('float32')


def packed_state(values, layout='joints12_grippers2'):
    values=np.asarray(values,dtype=np.float32)
    if values.shape[-1]!=14 or not np.isfinite(values).all():raise ValueError('Expected finite 14-D vector')
    if layout=='joints12_grippers2':return values.copy()
    if layout=='left7_right7':return values[...,ARMS_TO_PACKED].copy()
    raise ValueError('Declare joints12_grippers2 or left7_right7 explicitly')


@dataclass
class UnitConfig:
    close: tuple
    open: tuple
    smooth_frames: int=3
    min_frames: int=6
    max_frames: int=300
    history_frames: int=900
    history_units: int=16
    points: int=24
    recency_frames: float | None=None

    def __post_init__(self):
        self.close=tuple(float(x) for x in self.close);self.open=tuple(float(x) for x in self.open)
        if len(self.close)!=2 or len(self.open)!=2 or np.any(np.array(self.open)<=self.close):raise ValueError('Invalid gripper ranges')
        if min(self.smooth_frames,self.min_frames,self.history_units,self.points)<1 or self.max_frames<self.min_frames:
            raise ValueError('Invalid unit windows')
        if self.recency_frames is not None:
            self.recency_frames=float(self.recency_frames)
            if not np.isfinite(self.recency_frames) or self.recency_frames<=0:
                raise ValueError('Expected positive finite repetition memory')


def resample_path(path,points):
    """Arc-length phase removes execution-speed differences, preserving order."""
    path=np.asarray(path,dtype=np.float64)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    keep=np.r_[True,np.diff(arc)>1e-10]
    if arc[-1]<1e-8:return None
    return np.column_stack([np.interp(np.linspace(0,arc[-1],points),arc[keep],path[keep,j]) for j in range(path.shape[1])])


class ActionUnitHistory:
    def __init__(self,config):
        self.config=config if isinstance(config,UnitConfig) else UnitConfig(**config)
        if len(self.config.close)!=2 or np.any(np.array(self.config.open)<=self.config.close):raise ValueError('Invalid gripper hysteresis')
        self.reset()

    def reset(self):
        c=self.config;self.frame=0;self.observations=deque(maxlen=c.smooth_frames);self.paths=[[],[]]
        self.armed=[False,False];self.starts=[None,None];self.units=[deque(maxlen=c.history_units),deque(maxlen=c.history_units)]
        self.scores=np.zeros(2);self.last_end=np.full(2,-1);self.last_length=np.ones(2);self.events=[]

    def step(self,state,visual):
        c=self.config;s=np.asarray(state,dtype=np.float64);v=np.asarray(visual,dtype=np.float64)
        if s.shape!=(14,) or v.shape!=(48,) or not np.isfinite(s).all() or not np.isfinite(v).all():raise ValueError('Expected normalized state14 and visual48')
        self.observations.append(s);sm=np.mean(self.observations,axis=0);self.events=[]
        for arm,indices in enumerate((PACKED_TO_ARMS[:7],PACKED_TO_ARMS[7:])):
            pose=sm[indices];grip=pose[-1]
            if self.starts[arm] is not None:
                self.paths[arm].append(pose)
                if self.frame-self.starts[arm]>c.max_frames:self.starts[arm]=None;self.paths[arm]=[]
            if grip>=c.open[arm]:self.armed[arm]=True
            if self.armed[arm] and grip<=c.close[arm]:
                self.armed[arm]=False;start=self.starts[arm]
                if start is not None and self.frame-start>=c.min_frames:
                    curve=resample_path(self.paths[arm],c.points)
                    if curve is not None:
                        memory=self.units[arm]
                        while memory and self.frame-memory[0]['end']>c.history_frames:memory.popleft()
                        score=0.;best=0.;best_end=None
                        for old in memory:
                            # Absolute pose and direction/order are retained;
                            # the gripper has its own training normalization.
                            distance=np.sqrt(np.mean((curve-old['curve'])**2))
                            match=float(np.exp(-2*distance))
                            # Compare effects at the same gripper phase and
                            # similar body trajectory, reducing robot-motion confounding.
                            effect=float(np.sqrt(np.mean((v-old['visual'])**2)))
                            contribution=match*np.exp(-.5*effect**2)
                            if c.recency_frames is not None:
                                # Dense retries carry more evidence than similar
                                # actions separated by normal task progression.
                                contribution*=np.exp(-(self.frame-old['end'])/c.recency_frames)
                            score+=contribution
                            if contribution>best:best=contribution;best_end=old['end']
                        self.scores[arm]=score;self.last_end[arm]=self.frame;self.last_length[arm]=self.frame-start
                        event=dict(arm=arm,start=start,end=self.frame,frames=self.frame-start,score=score,
                            previous_units=len(memory),best_match=best,best_previous_end=best_end)
                        self.events.append(event);memory.append(dict(curve=curve,visual=v.copy(),end=self.frame))
                self.starts[arm]=self.frame;self.paths[arm]=[pose]
        # Keep evidence between units; clear stale evidence when motion stops.
        elapsed=np.maximum(0,self.frame-self.last_end-2*self.last_length)
        held=self.scores*np.exp(-elapsed/np.maximum(self.last_length,1))
        if c.recency_frames is not None:
            # An old completed unit must not keep supplying fresh evidence.
            held=self.scores*np.exp(-np.maximum(0,self.frame-self.last_end)/c.recency_frames)
        self.frame+=1
        return float(held.max())


def unit_trace(states,visual,config):
    history=ActionUnitHistory(config);score=np.empty(len(states),dtype='float32');events=[]
    for t,(s,v) in enumerate(zip(states,visual)):
        score[t]=history.step(s,v);events.extend(history.events)
    return score,events
