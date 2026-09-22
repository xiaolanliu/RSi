"""Train causal visual/motion change events with a weak 200-frame duration band."""
import argparse,json,math,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from .change_subtask import ChangeConfig,LatentChangeHead,teacher_boundary_targets,duration_regularizer
from .fit_ood_v4 import load,save_json,sha,section,inference_precision
from .paths import artifact_path
from .train_online_subtask import batches
from .train_ood_v4 import atomic_save


def objective(out,target,valid,config):
    logits=out['logits'];length=valid.sum(1).clamp_min(1)
    element=F.binary_cross_entropy_with_logits(logits,target,reduction='none',pos_weight=logits.new_tensor(6.))
    event=((element.mean(-1)*valid).sum(1)/length).mean()
    # A location loss uses complete training labels, never a future input.
    predicted=logits.sigmoid()*valid[:,:,None]
    predicted=predicted/predicted.sum(1,keepdim=True).clamp_min(1e-8)
    reference=target*valid[:,:,None];reference=reference/reference.sum(1,keepdim=True).clamp_min(1e-8)
    location=(reference*(reference.clamp_min(1e-8).log()-predicted.clamp_min(1e-8).log())).sum(1).mean()
    duration,expected=duration_regularizer(logits,valid,config,observed=target.amax(1)>0)
    loss=event+.25*location+config.duration_weight*duration
    return loss,dict(event_bce=float(event.detach()),location_kl=float(location.detach()),
        duration_smooth_l1=float(duration.detach()),weighted_duration=float(config.duration_weight*duration.detach()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('runs/ood_v4_2000_20260920/dim128/ood'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--duration-weight',type=float,default=.01);p.add_argument('--nominal-frames',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);inference_precision();torch.manual_seed(41)
    if (a.output/'history.jsonl').exists():raise FileExistsError('Use a fresh output; training is never silently overwritten')
    m=json.loads((a.source/'manifest.json').read_text());norm=load(a.source/'normalization.pt')
    if sha(artifact_path(m['checkpoint']))!=m['signature']['checkpoint_sha256']:raise ValueError('Shared encoder changed')
    if sha(a.source/'normalization.pt')!=m['normalization_sha256']:raise ValueError('Normalization changed')
    records=m['records'];ids=[i for i,r in enumerate(records) if r['split']=='train']
    if len(ids)!=2246:raise ValueError('Unexpected normal split')
    if a.smoke:ids=ids[:16]
    config=ChangeConfig(dimension=m['dimension'],duration_weight=a.duration_weight,nominal_frames=a.nominal_frames)
    zmap=np.load(a.source/'latent.npy',mmap_mode='r');frames=np.load(artifact_path(m['data'])/'frames.npy',mmap_mode='r')
    values={};targets={};unobserved=0
    for i in ids:
        r=records[i];z=torch.tensor(np.array(section(zmap,r)),device=a.device)
        values[i]=((z-norm['z_mean'].to(a.device))/norm['z_scale'].to(a.device)).clamp(-15,15)
        teacher=torch.tensor(np.array(frames[r['source_offset']:r['source_offset']+r['length'],76:]),device=a.device)
        target,centers,exists=teacher_boundary_targets(teacher,config)
        # Some recordings do not confidently reach the final teacher stage.
        # Keep their frames; never fabricate a boundary at recording end.
        unobserved+=int((~exists).sum())
        targets[i]=target
    a.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='latent_change_v6',seed=41,epochs=a.epochs,smoke=a.smoke,training_episodes=len(ids),
        frames_per_epoch=sum(records[i]['length'] for i in ids),config=asdict(config),unobserved_teacher_boundaries=unobserved,
        source=str(a.source.resolve()),source_manifest_sha256=sha(a.source/'manifest.json'),
        backbone_sha256=m['signature']['checkpoint_sha256'],normalization_sha256=m['normalization_sha256'],
        input='frozen causal fusion z128 and rolling latent contrasts; no teacher, duration or absolute time input',
        supervision='normal offline teacher boundary crossing targets; event BCE + .25 location KL + weak bounded duration SmoothL1',
        duration='default 200 frames; zero penalty in [100,500]; no terminal-stage loss, no runtime scheduled transitions',
        checkpoint_selection='fixed final epoch; no OOD/calibration/test checkpoint selection',
        code={n:sha(Path(__file__).parent/n) for n in ('change_subtask.py','train_change_subtask.py')})
    save_json(protocol,a.output/'protocol.json');model=LatentChangeHead(config).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);started=time.monotonic()
    for epoch in range(1,a.epochs+1):
        model.train();metrics=[];seen=set();covered=0
        lr=.001*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,a.epochs-1))))
        for group in optimizer.param_groups:group['lr']=lr
        for batch in batches(ids,records,a.batch_size,np.random.default_rng(41+epoch)):
            z=pad_sequence([values[i] for i in batch],batch_first=True);target=pad_sequence([targets[i] for i in batch],batch_first=True)
            lengths=torch.tensor([records[i]['length'] for i in batch],device=a.device)
            valid=torch.arange(z.shape[1],device=a.device)[None]<lengths[:,None]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(a.device).startswith('cuda')):out,_=model(z)
            loss,parts=objective(out,target,valid,config)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2,error_if_nonfinite=True);optimizer.step()
            metrics.append(dict(loss=float(loss.detach()),**parts));seen.update(batch);covered+=sum(records[i]['length'] for i in batch)
        assert seen==set(ids) and covered==protocol['frames_per_epoch']
        row=dict(epoch=epoch,frames=covered,seconds=time.monotonic()-started,**{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]})
        with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True);save_json(dict(state='complete' if epoch==a.epochs else 'training',**row),a.output/'status.json')
        if epoch==a.epochs or epoch%10==0:
            atomic_save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=asdict(config),epoch=epoch,
                protocol=protocol,optimizer=optimizer.state_dict()),a.output/f'checkpoint_epoch_{epoch:04d}.pt')
    print('CHANGE_SUBTASK_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
