"""Train an ordered causal subtask head on complete normal trajectories.

The pretrained shared encoder is frozen, and its hash-verified causal latent
cache is reused. No OOD episode, future feature, elapsed fraction, or teacher
phase is an input. Complete trajectories preserve memory across all frames.
"""
from .paths import artifact_path
import argparse,dataclasses,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from .online_subtask import OrderedConfig,OrderedSubtaskHead
from .fit_ood_v4 import load,save_json,sha,inference_precision,section
from .train_ood_v4 import atomic_save


def batches(ids,records,batch,rng):
    order=rng.permutation(ids);groups=[]
    for start in range(0,len(order),256):
        bucket=sorted(order[start:start+256],key=lambda i:records[i]['length'])
        groups.extend(bucket[j:j+batch] for j in range(0,len(bucket),batch))
    rng.shuffle(groups);return groups


def objective(out,y,valid):
    target=1-y.cumsum(-1)[...,:4];target=target.clamp(0,1)
    q=out['phase_probs'].clamp_min(1e-6)
    kl=(y*(y.clamp_min(1e-8).log()-q.log())).sum(-1)
    bce=F.binary_cross_entropy_with_logits(out['logits'],target,reduction='none').mean(-1)
    memory=(out['boundary_cdf']-target).abs().mean(-1)
    lengths=valid.sum(1).clamp_min(1)
    loss=(((kl+.5*bce+.5*memory)*valid).sum(1)/lengths).mean()
    metrics=torch.stack(((kl*valid).sum(),(bce*valid).sum(),(memory*valid).sum(),
        ((q.argmax(-1)==y.argmax(-1))*valid).sum(),valid.sum()))
    return loss,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);inference_precision();torch.manual_seed(29);np.random.seed(29)
    manifest=json.loads((a.source/'manifest.json').read_text());norm=load(a.source/'normalization.pt')
    if sha(artifact_path(manifest['checkpoint']))!=manifest['signature']['checkpoint_sha256']:raise ValueError('Backbone changed')
    if sha(a.source/'normalization.pt')!=manifest['normalization_sha256']:raise ValueError('Normalization changed')
    records=manifest['records'];train=[i for i,r in enumerate(records) if r['split']=='train']
    if len(train)!=2246:raise ValueError('Unexpected normal training split')
    if a.smoke:train=train[:16]
    zmap=np.load(a.source/'latent.npy',mmap_mode='r');frames=np.load(artifact_path(manifest['data'])/'frames.npy',mmap_mode='r')
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'history.jsonl').exists():raise FileExistsError('Use a fresh output; never overwrite training')
    values={};targets={}
    for i in train:
        r=records[i];z=torch.tensor(np.array(section(zmap,r)),device=a.device)
        values[i]=((z-norm['z_mean'].to(a.device))/norm['z_scale'].to(a.device)).clamp(-15,15)
        targets[i]=torch.tensor(np.array(frames[r['source_offset']:r['source_offset']+r['length'],76:]),device=a.device)
    cfg=OrderedConfig(dimension=manifest['dimension']);model=OrderedSubtaskHead(cfg).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    protocol=dict(version='ordered_online_v5',epochs=a.epochs,seed=29,batch_size=a.batch_size,smoke=a.smoke,
        training_episodes=len(train),training_frames=sum(records[i]['length'] for i in train),
        input='frozen causal z only; no teacher, future frames, absolute time or episode length',
        target='normal-only offline teacher soft boundary CDF and phase masks; not human labels',
        loss='episode-mean phase KL + 0.5 raw-boundary BCE + 0.5 memory CDF L1',
        memory='whole trajectory forward; no resets inside an episode; no teacher forcing',
        selection='fixed final epoch; no calibration/test/ood checkpoint selection',
        source=str(a.source.resolve()),source_manifest_sha256=sha(a.source/'manifest.json'),
        backbone_sha256=manifest['signature']['checkpoint_sha256'],normalization_sha256=manifest['normalization_sha256'],
        config=dataclasses.asdict(cfg),source_hashes={n:sha(Path(__file__).parent/n) for n in ('online_subtask.py','train_online_subtask.py')})
    save_json(protocol,a.output/'protocol.json');started=time.monotonic()
    torch.cuda.reset_peak_memory_stats(a.device)
    for epoch in range(1,a.epochs+1):
        model.train();totals=torch.zeros(5,device=a.device);losses=[];seen=set();begin=time.monotonic()
        for g in optimizer.param_groups:g['lr']=.001*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,a.epochs-1))))
        for ids in batches(train,records,a.batch_size,np.random.default_rng(29+epoch)):
            z=pad_sequence([values[i] for i in ids],batch_first=True);y=pad_sequence([targets[i] for i in ids],batch_first=True)
            length=torch.tensor([records[i]['length'] for i in ids],device=a.device)
            valid=torch.arange(z.shape[1],device=a.device)[None]<length[:,None]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out,_=model(z)
            loss,m=objective(out,y,valid)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite phase loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2,error_if_nonfinite=True);optimizer.step()
            totals+=m.detach();losses.append(float(loss.detach()));seen.update(ids)
        total=totals.cpu().tolist();assert seen==set(train) and round(total[-1])==protocol['training_frames']
        row=dict(epoch=epoch,loss=float(np.mean(losses)),teacher_kl=total[0]/total[-1],boundary_bce=total[1]/total[-1],
            boundary_cdf_mae=total[2]/total[-1],teacher_agreement=total[3]/total[-1],frames=round(total[-1]),
            seconds=time.monotonic()-begin,total_seconds=time.monotonic()-started,peak_memory_gib=torch.cuda.max_memory_allocated(a.device)/2**30)
        with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        save_json(dict(state='training' if epoch<a.epochs else 'complete',**row),a.output/'status.json');print(json.dumps(row),flush=True)
        if epoch==1 or epoch%10==0 or epoch==a.epochs:
            ck=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                config=dataclasses.asdict(cfg),epoch=epoch,protocol=protocol)
            atomic_save(ck,a.output/f'checkpoint_epoch_{epoch:04d}.pt')
    print('ORDERED_SUBTASK_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
