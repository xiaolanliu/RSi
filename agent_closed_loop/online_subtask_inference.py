"""Streaming subtask + OOD: only current state and causal visual features.

Frozen shared encoder -> ordered learned boundary memory -> one risk head.
OOD pauses CONFIRMATION, not feature collection or the candidate soft memory.
Every new task must reset; a task is assumed to start at P1. Camera/VAE remains
external and must itself be causal. No prefix is treated as a finished episode.
"""
from .paths import artifact_path
from pathlib import Path
import json
import numpy as np
import torch
from .ood_v4_inference import OODV4Stream
from .fit_ood_v4 import load,sha
from .streaming_fusion import readout_disagreement
from .online_subtask import OrderedSubtaskHead,OrderedConfig,OnlineRiskHead,StageConfirmation,phase_context


class OnlineSubtaskMonitor(OODV4Stream):
    def __init__(self,checkpoint,device='cpu',*,episode_seed=0,fps=30):
        if fps!=30:raise ValueError('Expected consecutive 30 FPS observations')
        ck=load(checkpoint);source=artifact_path(ck['source'])
        if sha(source/'manifest.json')!=ck['source_manifest_sha256']:raise ValueError('Shared cache provenance changed')
        if sha(source/'normalization.pt')!=ck['normalization_sha256']:raise ValueError('Normalization changed')
        if sha(artifact_path(ck['phase_checkpoint']))!=ck['phase_sha256']:raise ValueError('Ordered subtask checkpoint changed')
        # Reuse hash-verified observation normalization, encoder, LINe, and
        # bounded cycle cache. The old risk head is immediately replaced.
        super().__init__(source/'fold_all/checkpoint.pt',device,episode_seed=episode_seed)
        phase=load(ck['phase_checkpoint'])
        self.ordered=OrderedSubtaskHead(OrderedConfig(**phase['config'])).to(device).eval()
        self.ordered.load_state_dict(phase['model'])
        self.head=OnlineRiskHead(ck['dimension'],ck['hidden']).to(device).eval();self.head.load_state_dict(ck['model'])
        self.checkpoint=ck;self.threshold=ck['threshold'];self.preprocessing=ck['preprocessing']
        self.reset(episode_seed=episode_seed)

    def reset(self,*,episode_seed=0):
        super().reset(episode_seed=episode_seed)
        self.phase_memory=None
        if hasattr(self,'ordered'):
            cfg=self.ordered.config;self.confirmation=StageConfirmation(cfg.confirmation_threshold,cfg.confirmation_frames)

    @torch.inference_mode()
    def step_normalized(self,values,*,debug=False):
        values=np.asarray(values,dtype=np.float32)
        if values.shape!=(76,) or not np.isfinite(values).all():raise ValueError('Expected finite normalized 76-vector')
        x=torch.tensor(values,device=self.device)[None]
        z,self.encoder_cache=self.encoder.step(x[:,:14],x[:,14:28],x[:,28:76],self.encoder_cache)
        energy,instant_logits=self.line(z)
        uncertainty=readout_disagreement(z,self.encoder.phase,torch.tensor([self.frame],device=self.device),self.seed,
            samples=self.norm['dropout_samples'],dropout=self.norm['phase_dropout'])
        temporal=self.history.step(values[:14],z[0].cpu().numpy())
        evidence=torch.cat((energy[:,None],uncertainty[:,None],torch.tensor(temporal,device=self.device)[None]),-1)
        zn=((z-self.zmean)/self.zscale).clamp(-15,15);en=((evidence-self.emean)/self.escale).clamp(-15,15)
        phase,self.phase_memory=self.ordered.step(zn,self.phase_memory)
        logit,self.head_cache=self.head.step(zn,en,phase_context(phase),self.head_cache)
        raw=float(logit.sigmoid()[0]);self.raw_history.append(raw);self.smooth_history.append(float(np.mean(self.raw_history)))
        risk=min(self.smooth_history) if len(self.smooth_history)==self.preprocessing['persist_frames'] else 0.
        if self.frame<self.preprocessing['minimum_history']:risk=0.
        risk=float(np.float32(risk));alarm=risk>self.threshold;cdf=phase['boundary_cdf'][0].cpu().numpy()
        confirmed,accepted=self.confirmation.step(cdf,alarm)
        out=dict(frame_index=self.frame,phase_probs=phase['phase_probs'][0].cpu().numpy(),boundary_cdf=cdf,
            confirmed_phase=confirmed,accepted_phase=accepted,frame_risk_score=raw,risk_score=risk,threshold=self.threshold,alarm=alarm)
        if debug:out['debug']=dict(latent=z[0].cpu().numpy(),evidence=evidence[0].cpu().numpy(),
            boundary_evidence=phase['boundary_evidence'][0].cpu().numpy(),instant_phase_probs=instant_logits.softmax(-1)[0].cpu().numpy())
        self.frame+=1;return out
