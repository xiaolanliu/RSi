"""Full normal-trajectory V7 training; identical train/runtime ordered updates."""
import argparse,json,math,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .duration_subtask import DurationConfig,DurationOrderedHead,duration_objective
from .fit_ood_v4 import load,sha,save_json,section,inference_precision
from .paths import artifact_path
from .train_online_subtask import batches
from .train_ood_v4 import atomic_save


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('runs/ood_v4_2000_20260920/dim128/ood'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--duration-weight',type=float,default=.05);p.add_argument('--prior-strength',type=float,default=2.)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if (a.output/'history.jsonl').exists():raise FileExistsError('Use a fresh training directory')
    torch.set_num_threads(2);inference_precision();torch.manual_seed(41)
    m=json.loads((a.source/'manifest.json').read_text());norm=load(a.source/'normalization.pt')
    if sha(artifact_path(m['checkpoint']))!=m['signature']['checkpoint_sha256'] or sha(a.source/'normalization.pt')!=m['normalization_sha256']:raise ValueError('Stale latent provenance')
    records=m['records'];ids=[i for i,r in enumerate(records) if r['split']=='train'];assert len(ids)==2246
    config=DurationConfig(dimension=m['dimension'],duration_weight=a.duration_weight,prior_strength=a.prior_strength)
    zm=np.load(a.source/'latent.npy',mmap_mode='r');frames=np.load(artifact_path(m['data'])/'frames.npy',mmap_mode='r')
    values={};teachers={}
    for i in ids:
        r=records[i];z=torch.tensor(np.array(section(zm,r)),device=a.device)
        values[i]=((z-norm['z_mean'].to(a.device))/norm['z_scale'].to(a.device)).clamp(-15,15)
        teachers[i]=torch.tensor(np.array(frames[r['source_offset']:r['source_offset']+r['length'],76:]),device=a.device)
    a.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='ordered_duration_v7',seed=41,epochs=a.epochs,smoke=False,training_episodes=len(ids),
        frames_per_epoch=sum(records[i]['length'] for i in ids),config=asdict(config),source=str(a.source.resolve()),
        source_manifest_sha256=sha(a.source/'manifest.json'),backbone_sha256=m['signature']['checkpoint_sha256'],
        normalization_sha256=m['normalization_sha256'],
        input='frozen causal latent; observed stage age only; no total length, future frames or teacher inputs',
        recurrence='same full-sequence and streaming update; no teacher forcing or low-then-high rearm',
        loss='phase KL + .5 crossing BCE + .5 CDF L1 + weighted centered Poisson duration KL + .1 raw-evidence regression',
        selection='fixed final epoch, model variant chosen on normal calibration only before new test evaluation',
        code={n:sha(Path(__file__).parent/n) for n in ('duration_subtask.py','train_duration_subtask.py')})
    save_json(protocol,a.output/'protocol.json');model=DurationOrderedHead(config).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.03);started=time.monotonic()
    for epoch in range(1,a.epochs+1):
        model.train();metrics=[];seen=set();covered=0
        for g in optimizer.param_groups:g['lr']=.0007*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,a.epochs-1))))
        for batch in batches(ids,records,a.batch_size,np.random.default_rng(41+epoch)):
            z=pad_sequence([values[i] for i in batch],batch_first=True);y=pad_sequence([teachers[i] for i in batch],batch_first=True)
            lengths=torch.tensor([records[i]['length'] for i in batch],device=a.device);valid=torch.arange(z.shape[1],device=a.device)[None]<lengths[:,None]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out,_=model(z)
            loss,parts=duration_objective(out,y,valid,config)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2,error_if_nonfinite=True);optimizer.step()
            metrics.append(dict(loss=float(loss.detach()),**parts));seen.update(batch);covered+=sum(records[i]['length'] for i in batch)
        assert seen==set(ids) and covered==protocol['frames_per_epoch']
        row=dict(epoch=epoch,frames=covered,seconds=time.monotonic()-started,**{k:float(np.mean([r[k] for r in metrics])) for k in metrics[0]})
        with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True);save_json(dict(state='complete' if epoch==a.epochs else 'training',**row),a.output/'status.json')
        if epoch==a.epochs or epoch%10==0:atomic_save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=asdict(config),epoch=epoch,
            protocol=protocol),a.output/f'checkpoint_epoch_{epoch:04d}.pt')

if __name__=='__main__':main()
