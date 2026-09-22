"""Portable streaming inference; weights contain every learned dependency."""
from collections import deque
import numpy as np
import torch
from .fit_ood_v4 import load,inference_precision
from .streaming_fusion import SharedFusion,FusionConfig,readout_disagreement
from .temporal_evidence import HistoryEvidence
from .uncertainty_ood import FeatureLINe,SingleRiskHead
from .online_subtask import OrderedConfig,OrderedSubtaskHead,OnlineRiskHead,StageConfirmation,phase_context
from .change_subtask import ChangeConfig,LatentChangeHead,FreshStageMemory
from .duration_subtask import DurationConfig,DurationOrderedHead
from .three_signal_ood import ProjectionMemory,ThreeSignalRisk,SIGNAL_NAMES,calibration_tail
from .action_units import ActionUnitHistory,packed_state,repetition_strength


class OnlineMonitor:
    def __init__(self,bundle,device='cpu',*,fps=30,episode_seed=0):
        if fps!=30:raise ValueError('Current temporal windows require consecutive 30 FPS observations')
        inference_precision();self.bundle=load(bundle);b=self.bundle
        if b['format']!='agent_closed_loop.bundle.v1':raise ValueError('Unsupported model bundle')
        self.device=torch.device(device);self.norm=b['normalization'];self.threshold=b['threshold'];self.preprocessing=b['preprocessing']
        self.encoder=SharedFusion(FusionConfig(**b['encoder_config'])).to(device).eval();self.encoder.load_state_dict(b['encoder'])
        d=self.encoder.config.dim
        self.line=FeatureLINe(d,route_tolerance=b.get('line_route_tolerance',0.)).to(device).eval();self.line.load_state_dict(self.norm['line'])
        self.kind=b['stage_kind']
        self.risk_kind=b.get('risk_kind','legacy')
        self.three_signals=self.risk_kind in ('three_signal_v8','three_signal_v9','three_signal_v10','three_signal_v11')
        self.signal_names=b.get('signal_names',SIGNAL_NAMES)
        if self.kind=='ordered_v5':
            self.config=OrderedConfig(**b['stage_config']);self.stage=OrderedSubtaskHead(self.config)
            self.head=OnlineRiskHead(d,b['risk_hidden'])
        elif self.kind=='latent_change_v6':
            self.config=ChangeConfig(**b['stage_config']);self.stage=LatentChangeHead(self.config)
            self.head=SingleRiskHead(d,b['risk_hidden'])
        elif self.kind=='ordered_duration_v7':
            self.config=DurationConfig(**b['stage_config']);self.stage=DurationOrderedHead(self.config)
            if self.risk_kind=='legacy':self.head=SingleRiskHead(d,b['risk_hidden'])
        else:raise ValueError('Unsupported stage semantics')
        self.stage.load_state_dict(b['stage']);self.stage.to(device).eval()
        if self.three_signals:
            if self.kind!='ordered_duration_v7':raise ValueError('Three-signal bundle requires the V7 stage head')
            self.projection=ProjectionMemory(**b['projection_memory']).to(device).eval()
            self.head=ThreeSignalRisk(**b['signal_normalization']).to(device).eval()
            self.calibration_reference=b.get('calibration_reference')
            if self.risk_kind in ('three_signal_v9','three_signal_v10','three_signal_v11'):
                self.unit_history=ActionUnitHistory(b['unit_config'])
                self.repetition_reference={k:v.numpy() for k,v in b['repetition_reference'].items()}
        elif self.risk_kind=='legacy':
            self.head.load_state_dict(b['risk']);self.head.to(device).eval()
        else:raise ValueError('Unsupported risk semantics')
        self.input_mean=np.asarray(self.norm['input_normalization']['mean']);self.input_scale=np.asarray(self.norm['input_normalization']['scale'])
        self.zmean=self.norm['z_mean'].to(device);self.zscale=self.norm['z_scale'].to(device)
        if self.risk_kind=='legacy':
            self.emean=self.norm['evidence_mean'].to(device);self.escale=self.norm['evidence_scale'].to(device)
        self.history=HistoryEvidence(self.norm['z_scale'].numpy(),self.norm['visual_importance'].numpy());self.reset(episode_seed=episode_seed)

    def reset(self,*,episode_seed=0):
        self.seed=int(episode_seed);self.frame=0;self.previous=None;self.encoder_cache=None;self.stage_cache=None;self.risk_cache=None
        self.history.reset();self.raw_history=deque(maxlen=self.preprocessing['smooth_frames']);self.smooth_history=deque(maxlen=self.preprocessing['persist_frames'])
        if hasattr(self,'unit_history'):self.unit_history.reset()
        self.memory=(None if self.kind=='ordered_duration_v7' else FreshStageMemory(self.config) if self.kind=='latent_change_v6' else StageConfirmation(self.config.confirmation_threshold,self.config.confirmation_frames))

    def normalize(self,state,visual):
        state=np.asarray(state,dtype=np.float32);visual=np.asarray(visual,dtype=np.float32)
        if state.shape!=(14,) or visual.shape!=(48,) or not np.isfinite(state).all() or not np.isfinite(visual).all():raise ValueError('Expected finite state14 and causal visual48')
        delta=np.zeros_like(state) if self.previous is None else state-self.previous;self.previous=state.copy()
        return np.clip((np.concatenate((state,delta,visual))-self.input_mean)/self.input_scale,-15,15).astype(np.float32)

    @torch.inference_mode()
    def step(self,state,visual,*,debug=False,state_layout='joints12_grippers2'):
        """One current observation. Motion is the observed backward state delta."""
        return self.step_normalized(self.normalize(packed_state(state,state_layout),visual),debug=debug)

    @torch.inference_mode()
    def step_normalized(self,values,*,debug=False):
        """Prepared 76-D input; do not mix with raw step within an episode."""
        values=np.asarray(values,dtype=np.float32)
        if values.shape!=(76,) or not np.isfinite(values).all():raise ValueError('Expected normalized state14/delta14/visual48')
        x=torch.tensor(values,device=self.device)[None]
        z,self.encoder_cache=self.encoder.step(x[:,:14],x[:,14:28],x[:,28:76],self.encoder_cache)
        energy,instant_logits=self.line(z)
        zn=((z-self.zmean)/self.zscale).clamp(-15,15)
        if self.three_signals:
            distance=self.projection(zn)
            loop=self.history.step(values[:14],z[0].cpu().numpy())[0]
            if self.risk_kind in ('three_signal_v9','three_signal_v10','three_signal_v11'):
                unit=self.unit_history.step(values[:14],values[28:76])
                loop=float(repetition_strength(loop,unit,self.repetition_reference))
            evidence=torch.stack((distance,energy.new_tensor([loop]),energy),-1)
            raw=float(self.head(evidence)[0])
        else:
            uncertainty=readout_disagreement(z,self.encoder.phase,torch.tensor([self.frame],device=self.device),self.seed,
                samples=self.norm['dropout_samples'],dropout=self.norm['phase_dropout'])
            temporal=self.history.step(values[:14],z[0].cpu().numpy())
            evidence=torch.cat((energy[:,None],uncertainty[:,None],torch.tensor(temporal,device=self.device)[None]),-1)
            en=((evidence-self.emean)/self.escale).clamp(-15,15)
            if self.kind!='ordered_duration_v7':stage,self.stage_cache=self.stage.step(zn,self.stage_cache)
            if self.kind=='ordered_v5':logit,self.risk_cache=self.head.step(zn,en,phase_context(stage),self.risk_cache)
            else:logit,self.risk_cache=self.head.step(zn,en,self.risk_cache)
            raw=float(logit.sigmoid()[0])
        self.raw_history.append(raw);self.smooth_history.append(float(np.mean(self.raw_history)))
        score=min(self.smooth_history) if len(self.smooth_history)==self.preprocessing['persist_frames'] else 0.
        if self.frame<self.preprocessing['minimum_history']:score=0.
        score=float(np.float32(score));alarm=score>self.threshold
        if self.three_signals and self.calibration_reference is not None:
            tail=float(calibration_tail(score,self.calibration_reference.numpy()))
            alarm=tail<=self.bundle['alarm_tail_fraction']
        if self.kind=='ordered_v5':
            cdf=stage['boundary_cdf'][0].cpu().numpy();confirmed,accepted=self.memory.step(cdf,alarm)
            result=dict(phase_probs=stage['phase_probs'][0].cpu().numpy(),boundary_cdf=cdf,confirmed_phase=confirmed,accepted_phase=accepted)
        elif self.kind=='latent_change_v6':
            result=self.memory.step(stage['boundary_event_probs'][0].cpu().numpy(),stage['change_sizes'][0].cpu().numpy(),
                stage['recent_latent'][0].cpu().numpy(),alarm)
        else:
            stage,self.stage_cache=self.stage.step(zn,self.stage_cache,pause=[alarm])
            confirmed=int(stage['confirmed_phase'][0])
            result={k:stage[k][0].cpu().numpy() for k in ('phase_probs','boundary_cdf','boundary_evidence','duration_log_bias','observation_boundary_probs','change_sizes')}
            result.update(confirmed_phase=confirmed,accepted_phase=0 if alarm else confirmed)
        result.update(frame_index=self.frame,frame_risk_score=raw,risk_score=score,threshold=self.threshold,
                      risk_threshold_ratio=score/max(self.threshold,1e-30),alarm=alarm)
        if self.three_signals:
            result['ood_signals']=dict(zip(self.signal_names,evidence[0].cpu().tolist()))
            if self.calibration_reference is not None:
                result['ood_percentiles']=dict(zip(self.signal_names,self.head.percentiles(evidence)[0].cpu().tolist()))
                result['risk_percentile']=1-tail
        if debug:result['debug']=dict(latent=z[0].cpu().numpy(),evidence=evidence[0].cpu().numpy(),instant_phase_probs=instant_logits.softmax(-1)[0].cpu().numpy())
        self.frame+=1;return result
