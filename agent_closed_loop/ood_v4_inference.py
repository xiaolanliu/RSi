"""Causal stateful V4 inference from state14 + externally encoded visual48.

The VAE remains external. reset() is required at every episode boundary.
No phase teacher, future observations, recorded action alias, or timestamps
encoding episode completion are inputs. frame_index is only for dropout RNG.
"""
from .paths import artifact_path
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .fit_ood_v4 import load,inference_precision
from .prepare_ood_v4 import sha
from .streaming_fusion import FusionConfig, SharedFusion, readout_disagreement
from .temporal_evidence import HistoryEvidence
from .uncertainty_ood import FeatureLINe, SingleRiskHead


class OODV4Stream:
    def __init__(self,checkpoint,device='cpu',*,episode_seed=0):
        inference_precision()
        ck=load(Path(checkpoint));self.device=torch.device(device)
        if sha(artifact_path(ck['backbone']))!=ck['backbone_sha256']:raise ValueError('Backbone hash mismatch')
        if sha(artifact_path(ck['normalization']))!=ck['normalization_sha256']:raise ValueError('Normalization hash mismatch')
        backbone=load(ck['backbone']);self.norm=load(ck['normalization'])
        self.encoder=SharedFusion(FusionConfig(**backbone['config'])).to(device).eval();self.encoder.load_state_dict(backbone['model'])
        self.line=FeatureLINe(ck['dimension']).to(device).eval();self.line.load_state_dict(self.norm['line'])
        self.head=SingleRiskHead(ck['dimension'],hidden=ck['hidden']).to(device).eval();self.head.load_state_dict(ck['model'])
        self.threshold=ck['threshold'];self.preprocessing=ck['preprocessing'];self.checkpoint=ck
        self.input_mean=np.asarray(self.norm['input_normalization']['mean'])
        self.input_scale=np.asarray(self.norm['input_normalization']['scale'])
        self.zmean=self.norm['z_mean'].to(device);self.zscale=self.norm['z_scale'].to(device)
        self.emean=self.norm['evidence_mean'].to(device);self.escale=self.norm['evidence_scale'].to(device)
        self.history=HistoryEvidence(self.norm['z_scale'].numpy(),self.norm['visual_importance'].numpy())
        self.reset(episode_seed=episode_seed)

    def reset(self,*,episode_seed=0):
        self.seed=int(episode_seed);self.frame=0;self.previous=None;self.encoder_cache=None;self.head_cache=None
        self.history.reset();self.raw_history=deque(maxlen=self.preprocessing['smooth_frames'])
        self.smooth_history=deque(maxlen=self.preprocessing['persist_frames'])

    def normalize(self,state,visual):
        state=np.asarray(state,dtype=np.float32);visual=np.asarray(visual,dtype=np.float32)
        if state.shape!=(14,) or visual.shape!=(48,):raise ValueError('Expected state14 and visual48')
        if not np.isfinite(state).all() or not np.isfinite(visual).all():raise ValueError('Nonfinite observation')
        delta=np.zeros_like(state) if self.previous is None else state-self.previous
        x=np.concatenate((state,delta,visual))
        self.previous=state.copy()
        return np.clip((x-self.input_mean)/self.input_scale,-15,15).astype('float32')

    @torch.inference_mode()
    def step(self,state,visual,*,debug=False):
        return self.step_normalized(self.normalize(state,visual),debug=debug)

    @torch.inference_mode()
    def step_normalized(self,values,*,debug=False):
        """Prepared-input API for parity/latency tests; supply all 76 columns.

        Do not mix with step(raw) within an episode: raw delta history is owned
        by normalize(), whereas this entry point accepts precomputed deltas.
        """
        values=np.asarray(values,dtype=np.float32)
        if values.shape!=(76,) or not np.isfinite(values).all():raise ValueError('Expected finite normalized 76-vector')
        x=torch.tensor(values,device=self.device)[None]
        z,self.encoder_cache=self.encoder.step(x[:,:14],x[:,14:28],x[:,28:76],self.encoder_cache)
        energy,logits=self.line(z);probs=logits.softmax(-1)
        uncertainty=readout_disagreement(z,self.encoder.phase,torch.tensor([self.frame],device=self.device),self.seed,
                                        samples=self.norm['dropout_samples'],dropout=self.norm['phase_dropout'])
        temporal=self.history.step(values[:14],z[0].cpu().numpy())
        evidence=torch.cat((energy[:,None],uncertainty[:,None],torch.tensor(temporal,device=self.device)[None]),-1)
        zn=((z-self.zmean)/self.zscale).clamp(-15,15);en=((evidence-self.emean)/self.escale).clamp(-15,15)
        zero=self.checkpoint['results'].get('zeroed_evidence',[])
        if zero:en[:,zero]=0
        logit,self.head_cache=self.head.step(zn,en,self.head_cache);raw=float(logit.sigmoid()[0])
        self.raw_history.append(raw);self.smooth_history.append(float(np.mean(self.raw_history)))
        risk=min(self.smooth_history) if len(self.smooth_history)==self.preprocessing['persist_frames'] else 0.
        if self.frame<self.preprocessing['minimum_history']:risk=0.
        # Offline intervention_trace emits float32; match threshold precision.
        risk=float(np.float32(risk));alarm=risk>self.threshold
        out=dict(frame_index=self.frame,frame_risk_score=raw,risk_score=risk,threshold=self.threshold,alarm=alarm,
                 phase_probs=probs[0].cpu().numpy(),accepted_phase=0 if alarm else int(probs[0].argmax())+1)
        if debug:out['debug']=dict(latent=z[0].cpu().numpy(),evidence=evidence[0].cpu().numpy())
        self.frame+=1;return out
