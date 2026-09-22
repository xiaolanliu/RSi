"""Batch-1 warm streaming latency, with cached VAE features (VAE excluded).

Total timing uses the actual deployment step. Stage timings synchronize at
boundaries and therefore include measurement overhead; do not sum percentiles.
"""
from .paths import artifact_path
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .fit_ood_v4 import save_json,episode_seed,load
from .online_subtask_inference import OnlineSubtaskMonitor
from .online_subtask import phase_context
from .fit_ood_v4 import sha
from .streaming_fusion import readout_disagreement


def stats(values):
    return dict(p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),mean=float(np.mean(values)))


@torch.inference_mode()
def detailed_step(stream,state,visual):
    """Instrument the same equations as SharedFusion.step; validate separately."""
    stages={};device=stream.device
    def clock():
        if device.type=='cuda':torch.cuda.synchronize(device)
        return time.perf_counter()
    start=clock()
    def mark(name):
        nonlocal start
        now=clock();stages[name]=(now-start)*1000;start=now
    values=stream.normalize(state,visual);mark('input_normalization')
    x=torch.tensor(values,device=device)[None];mark('input_transfer')
    enc=stream.encoder;cache={} if stream.encoder_cache is None else stream.encoder_cache
    u=enc.state(x[:,:14])+enc.motion(x[:,14:28]);q=enc._heads(enc.q(u[:,None]));k=enc._heads(enc.k(x[:,None,28:76]));v=enc._heads(enc.v(x[:,None,28:76]))
    current_v=v[:,:,0].reshape_as(u);mark('state_motion_and_qkv')
    if 'k' in cache:
        k=torch.cat((cache['k'],k),2)[:,:,-enc.config.window:];v=torch.cat((cache['v'],v),2)[:,:,-enc.config.window:]
    lag=torch.arange(k.shape[2]-1,-1,-1,device=device);bias=enc.relative_bias[:,lag].to(q.dtype)[None,:,None]
    y=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.)[:,:,0].reshape_as(u)
    x=u+enc.fusion_gate.sigmoid()*enc.out(y)+enc.visual_gate.sigmoid()*current_v;mark('causal_attention_cache_and_fusion')
    histories=[]
    for block,buf in zip(enc.blocks,cache.get('blocks',[None]*len(enc.blocks))):
        x,buf=block.step(x,buf);histories.append(buf)
    z=F.relu(enc.latent(x));stream.encoder_cache=dict(k=k,v=v,blocks=histories);mark('causal_tcn_and_latent')
    energy,logits=stream.line(z);mark('line_instantaneous_readout')
    u=readout_disagreement(z,enc.phase,torch.tensor([stream.frame],device=device),stream.seed,
                          samples=stream.norm['dropout_samples'],dropout=stream.norm['phase_dropout']);mark('dropout_readout_4')
    h=stream.history.step(values[:14],z[0].cpu().numpy());mark('latent_to_cpu_and_cycle_history')
    e=torch.cat((energy[:,None],u[:,None],torch.tensor(h,device=device)[None]),-1)
    zn=((z-stream.zmean)/stream.zscale).clamp(-15,15);en=((e-stream.emean)/stream.escale).clamp(-15,15)
    phase,stream.phase_memory=stream.ordered.step(zn,stream.phase_memory);mark('ordered_boundary_and_soft_memory')
    logit,stream.head_cache=stream.head.step(zn,en,phase_context(phase),stream.head_cache);raw=float(logit.sigmoid()[0]);mark('single_risk_head')
    stream.raw_history.append(raw);stream.smooth_history.append(float(np.mean(stream.raw_history)))
    risk=min(stream.smooth_history) if len(stream.smooth_history)==stream.preprocessing['persist_frames'] else 0.
    if stream.frame<stream.preprocessing['minimum_history']:risk=0.
    risk=float(np.float32(risk));alarm=risk>stream.threshold
    cdf=phase['boundary_cdf'][0].cpu().numpy();confirmed,accepted=stream.confirmation.step(cdf,alarm)
    result=dict(frame_risk_score=raw,risk_score=risk,alarm=alarm,phase_probs=phase['phase_probs'][0].cpu().numpy(),confirmed_phase=confirmed,accepted_phase=accepted)
    stream.frame+=1;mark('smoothing_and_output');return result,stages


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--device',default='cuda:0');p.add_argument('--samples',type=int,default=400);a=p.parse_args()
    torch.set_num_threads(2);m=json.loads((a.folder/'manifest.json').read_text())
    r=next(r for r in m['records'] if r['split']=='ood' and r['episode_index']==2)
    source=json.loads((artifact_path(m['source'])/'manifest.json').read_text())
    data_manifest=json.loads((artifact_path(source['data'])/'manifest.json').read_text())
    arrays=load(artifact_path(data_manifest['source'])/r['file']);states=arrays['states'].numpy();visuals=arrays['features'][:,-48:].numpy()
    warmup=400;n=min(a.samples,len(states)-warmup);checkpoint=a.folder/'fold_all/checkpoint.pt'
    s=OnlineSubtaskMonitor(checkpoint,a.device,episode_seed=episode_seed(r));full=[];outputs=[]
    for i,(state,visual) in enumerate(zip(states[:warmup+n],visuals[:warmup+n])):
        if s.device.type=='cuda':torch.cuda.synchronize(s.device)
        start=time.perf_counter();o=s.step(state,visual)
        if s.device.type=='cuda':torch.cuda.synchronize(s.device)
        if i>=warmup:full.append((time.perf_counter()-start)*1000);outputs.append(o)
    s.reset(episode_seed=episode_seed(r));stage_times={};max_difference=0
    for i,(state,visual) in enumerate(zip(states[:warmup+n],visuals[:warmup+n])):
        o,stage=detailed_step(s,state,visual)
        if i>=warmup:
            ref=outputs[i-warmup];max_difference=max(max_difference,abs(o['risk_score']-ref['risk_score']))
            if abs(o['frame_risk_score']-ref['frame_risk_score'])>1e-5 or o['alarm']!=ref['alarm']:raise AssertionError('Instrumented pipeline differs')
            np.testing.assert_allclose(o['phase_probs'],ref['phase_probs'],atol=1e-6)
            assert o['confirmed_phase']==ref['confirmed_phase'] and o['accepted_phase']==ref['accepted_phase']
            for k,v in stage.items():stage_times.setdefault(k,[]).append(v)
    result=dict(checkpoint_sha256=sha(checkpoint),phase_sha256=m['signature']['phase_sha256'],device=a.device,dimension=m['dimension'],backbone_epoch=m['backbone_epoch'],samples=n,warmup_frames=warmup,
        scope='raw state14 + cached visual48 -> normalization, observed backward difference, phase and OOD output; NO VAE, image decode, robot IO',
        total_ms=stats(full),stages_ms={k:stats(v) for k,v in stage_times.items()},instrumented_max_risk_difference=max_difference,
        note='stage boundaries synchronize; percentiles are not additive; total measured without stage instrumentation',
        hardware=torch.cuda.get_device_name(s.device) if s.device.type=='cuda' else 'CPU',threads=torch.get_num_threads())
    save_json(result,a.folder/'latency.json');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
