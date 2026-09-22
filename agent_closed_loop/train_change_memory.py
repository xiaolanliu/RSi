"""Sustained cross-boundary evidence with causal contrasts and weak duration.

Unlike a brief event target, cumulative evidence remains available if the
confirmation window starts late. Runtime still requires fresh stage evidence;
training-only ordered soft occupancy supplies the bounded duration objective.
"""
import argparse,json,math,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from .change_subtask import ChangeConfig,LatentChangeHead
from .online_subtask import ordered_memory
from .fit_ood_v4 import load,save_json,sha,section,inference_precision
from .paths import artifact_path
from .train_online_subtask import batches
from .train_ood_v4 import atomic_save


def memory_objective(out,teacher,valid,config):
    logits=out['logits'];raw=logits.sigmoid();width=config.evidence_frames
    sustained=F.pad(raw.transpose(1,2),(width-1,0)).transpose(1,2).unfold(1,width,1).amin(-1)
    cdf,q,_=ordered_memory(sustained)
    target=(1-teacher.cumsum(-1)[...,:4]).clamp(0,1)
    phase=(teacher*(teacher.clamp_min(1e-8).log()-q.clamp_min(1e-6).log())).sum(-1)
    evidence=F.binary_cross_entropy_with_logits(logits,target,reduction='none').mean(-1)
    n=valid.sum(1).clamp_min(1)
    fit=(((phase+.5*evidence)*valid).sum(1)/n).mean()
    mass=(q*valid[:,:,None]).sum(1)[:,:4];ratio=mass/config.nominal_frames
    low=F.relu(config.duration_lower_ratio-ratio);high=F.relu(ratio-config.duration_upper_ratio)
    violations=.5*(F.smooth_l1_loss(low,torch.zeros_like(low),beta=.25,reduction='none')+
        F.smooth_l1_loss(high,torch.zeros_like(high),beta=.25,reduction='none'))
    # No unobserved/terminal boundary is invented at the end of a recording.
    observed=(target*valid[:,:,None]).amax(1)>=.5
    duration=(violations*observed).sum()/observed.sum().clamp_min(1)
    loss=fit+config.duration_weight*duration
    return loss,dict(phase_kl=float(((phase*valid).sum()/valid.sum()).detach()),
        boundary_bce=float(((evidence*valid).sum()/valid.sum()).detach()),
        duration_smooth_l1=float(duration.detach()),weighted_duration=float((config.duration_weight*duration).detach()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('runs/ood_v4_2000_20260920/dim128/ood'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--duration-weight',type=float,default=.01);p.add_argument('--nominal-frames',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if (a.output/'history.jsonl').exists():raise FileExistsError('Use a fresh training output')
    torch.set_num_threads(2);inference_precision();torch.manual_seed(41)
    m=json.loads((a.source/'manifest.json').read_text());norm=load(a.source/'normalization.pt')
    if sha(artifact_path(m['checkpoint']))!=m['signature']['checkpoint_sha256'] or sha(a.source/'normalization.pt')!=m['normalization_sha256']:raise ValueError('Shared encoder/normalization changed')
    records=m['records'];ids=[i for i,r in enumerate(records) if r['split']=='train'];assert len(ids)==2246
    config=ChangeConfig(dimension=m['dimension'],duration_weight=a.duration_weight,nominal_frames=a.nominal_frames)
    zmap=np.load(a.source/'latent.npy',mmap_mode='r');frames=np.load(artifact_path(m['data'])/'frames.npy',mmap_mode='r')
    values={};teachers={};missing=0
    for i in ids:
        r=records[i];z=torch.tensor(np.array(section(zmap,r)),device=a.device)
        values[i]=((z-norm['z_mean'].to(a.device))/norm['z_scale'].to(a.device)).clamp(-15,15)
        y=torch.tensor(np.array(frames[r['source_offset']:r['source_offset']+r['length'],76:]),device=a.device);teachers[i]=y
        missing+=int(((1-y.cumsum(-1)[:,:4]).amax(0)<.5).sum())
    a.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='latent_change_memory_v6',seed=41,epochs=a.epochs,smoke=False,training_episodes=len(ids),
        frames_per_epoch=sum(records[i]['length'] for i in ids),config=asdict(config),unobserved_teacher_boundaries=missing,
        source=str(a.source.resolve()),source_manifest_sha256=sha(a.source/'manifest.json'),
        backbone_sha256=m['signature']['checkpoint_sha256'],normalization_sha256=m['normalization_sha256'],
        input='causal fused latent and 8/32/128-frame contrasts only; no duration/teacher/absolute time input',
        target='normal-only soft phase occupancy and cumulative crossing evidence',
        loss='ordered phase KL + .5 crossing BCE + .01 bounded duration SmoothL1 on completed nonterminal stages',
        selection='fixed epoch60; confirmation threshold chosen using normal calibration only',
        duration='200 default, no penalty within [100,500]; no runtime scheduled transition',
        code={n:sha(Path(__file__).parent/n) for n in ('change_subtask.py','train_change_memory.py','online_subtask.py')})
    save_json(protocol,a.output/'protocol.json');model=LatentChangeHead(config).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);started=time.monotonic()
    for epoch in range(1,a.epochs+1):
        model.train();metrics=[];seen=set();covered=0
        for g in optimizer.param_groups:g['lr']=.001*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,a.epochs-1))))
        for batch in batches(ids,records,a.batch_size,np.random.default_rng(41+epoch)):
            z=pad_sequence([values[i] for i in batch],batch_first=True);y=pad_sequence([teachers[i] for i in batch],batch_first=True)
            lengths=torch.tensor([records[i]['length'] for i in batch],device=a.device);valid=torch.arange(z.shape[1],device=a.device)[None]<lengths[:,None]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out,_=model(z)
            loss,parts=memory_objective(out,y,valid,config)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2,error_if_nonfinite=True);optimizer.step()
            metrics.append(dict(loss=float(loss.detach()),**parts));seen.update(batch);covered+=sum(records[i]['length'] for i in batch)
        assert seen==set(ids) and covered==protocol['frames_per_epoch']
        row=dict(epoch=epoch,frames=covered,seconds=time.monotonic()-started,**{k:float(np.mean([r[k] for r in metrics])) for k in metrics[0]})
        with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True);save_json(dict(state='complete' if epoch==a.epochs else 'training',**row),a.output/'status.json')
        if epoch==a.epochs or epoch%10==0:atomic_save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=asdict(config),epoch=epoch,
            protocol=protocol,optimizer=optimizer.state_dict()),a.output/f'checkpoint_epoch_{epoch:04d}.pt')
    print('CHANGE_MEMORY_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
