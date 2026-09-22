"""Frozen final-backbone LINe fitting, matched LOEO risk training and evaluation.

All fits use normal TRAIN references. OOD bags are separated by whole episode;
only the two held-in positive bags reach each fold's optimizer. No OOD teacher
or exact frame labels are used. Official runs require backbone epoch 2000.
"""
from .paths import artifact_path
import argparse
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .prepare_ood_v4 import sha
from .recovery import weak_episode_loss, intervention_trace, episode_metrics
from .streaming_fusion import FusionConfig, SharedFusion, readout_disagreement
from .temporal_evidence import EVIDENCE_NAMES, time_evidence
from .uncertainty_ood import FeatureLINe, SingleRiskHead


def save_json(value, path):
    path=Path(path);tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def load(path):return torch.load(artifact_path(path),map_location='cpu',weights_only=True)


def inference_precision():
    # TF32 cuDNN kernels depend on sequence shape and can produce different
    # offline vs single-step activations at thresholds. Explicit IEEE FP32
    # keeps both inference paths aligned; backbone training remains BF16.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def episode_seed(record):
    key=f"{record['source_root']}:{record['episode_index']}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:7],16)


def task_of(record):return Path(record['source_root']).parent.name


def section(array,record):
    return array[record['offset']:record['offset']+record['length']]


@torch.inference_mode()
def encode_episode(model,values,device,chunk=512):
    """Bounded exact left context; each target is emitted once, no future."""
    outputs=[]
    for start in range(0,len(values),chunk):
        begin=max(0,start-model.config.warmup);end=min(len(values),start+chunk)
        x=torch.tensor(np.array(values[begin:end,:76]),device=device)[None]
        z=model(x[:,:,:14],x[:,:,14:28],x[:,:,28:76])
        outputs.append(z[0,start-begin:].cpu())
    return torch.cat(outputs)


def prepare(args):
    inference_precision()
    folder=args.output;folder.mkdir(parents=True,exist_ok=True)
    ck=load(args.checkpoint);source=json.loads((args.data/'manifest.json').read_text())
    if ck['epoch']!=2000 and not args.smoke:raise ValueError('Official evaluation requires epoch 2000')
    if sha(args.data/'manifest.json')!=ck['protocol']['data_manifest_sha256']:raise ValueError('Data mismatch')
    signature=dict(checkpoint_sha256=sha(args.checkpoint),data_manifest_sha256=sha(args.data/'manifest.json'),
                   smoke=args.smoke,dropout_samples=4,code={n:sha(Path(__file__).parent/n) for n in
                   ('fit_ood_v4.py','streaming_fusion.py','uncertainty_ood.py','temporal_evidence.py')})
    if (folder/'manifest.json').exists():
        done=json.loads((folder/'manifest.json').read_text())
        if done['signature']!=signature or not done['complete']:raise ValueError('Stale/incomplete cached fit')
        return done
    records=source['records']
    if any(r['fps']!=30 for r in records):raise ValueError('This fixed-period protocol requires continuous 30 FPS input')
    if args.smoke:
        # Mechanism smoke only: explicitly not the official split/results.
        records=sum(([r for r in records if r['split']==s][:6] for s in ('train','calibration','validation','ood')),[])
    model=SharedFusion(FusionConfig(**ck['config'])).to(args.device).eval();model.load_state_dict(ck['model'])
    for p in model.parameters():p.requires_grad_(False)
    d=model.config.dim;frames=np.load(args.data/'frames.npy',mmap_mode='r')
    total=sum(r['length'] for r in records);zmap=np.lib.format.open_memmap(folder/'latent.npy',mode='w+',dtype='float32',shape=(total,d))
    phase=np.lib.format.open_memmap(folder/'phase.npy',mode='w+',dtype='float32',shape=(total,5))
    evidence=np.lib.format.open_memmap(folder/'evidence.npy',mode='w+',dtype='float32',shape=(total,7))
    out=[];offset=0;references=[];teachers=[];sums=np.zeros(d);squares=np.zeros(d);count=0
    quality={s:dict(frames=0,teacher_kl_sum=0.,teacher_agreement_sum=0,visual_loss_sum=0.,state_loss_sum=0.)
             for s in ('train','calibration','validation')};started=time.monotonic()
    for i,r in enumerate(records):
        values=section(frames,r);z=encode_episode(model,values,args.device);r2=dict(r,source_offset=r['offset'],offset=offset)
        sl=slice(offset,offset+len(z));zmap[sl]=z.numpy();offset+=len(z);out.append(r2)
        with torch.inference_mode():
            zg=z.to(args.device);logits=model.phase(zg).float();p=logits.softmax(-1)
            phase[sl]=p.cpu().numpy()
            evidence[sl,1]=readout_disagreement(zg,model.phase,torch.arange(len(z),device=args.device),episode_seed(r)).cpu().numpy()
            if r['split']!='ood':
                y=torch.tensor(np.array(values[:,76:]),device=args.device)
                q=quality[r['split']];q['frames']+=len(z)
                q['teacher_kl_sum']+=float(torch.nn.functional.kl_div(logits.log_softmax(-1),y,reduction='sum'))
                q['teacher_agreement_sum']+=int((logits.argmax(-1)==y.argmax(-1)).sum())
                q['visual_loss_sum']+=float(torch.nn.functional.smooth_l1_loss(model.visual_decoder(zg),torch.tensor(np.array(values[:,28:76]),device=args.device),reduction='sum'))/48
                q['state_loss_sum']+=float(torch.nn.functional.smooth_l1_loss(model.state_decoder(zg),torch.tensor(np.array(values[:,:14]),device=args.device),reduction='sum'))/14
        if r['split']=='train':
            idx=np.linspace(0,len(z)-1,min(128,len(z)),dtype=int)
            references.append(z[idx]);teachers.append(torch.from_numpy(np.array(values[idx,76:])))
            zd=z.numpy().astype('float64');sums+=zd.sum(0);squares+=(zd*zd).sum(0);count+=len(z)
        if i%100==0:print(json.dumps(dict(stage='encode',episode=i,total=len(records),seconds=time.monotonic()-started)),flush=True)
    mean=sums/count;scale=np.sqrt(np.maximum(squares/count-mean*mean,0)).clip(.05)
    line=FeatureLINe(d);fit=line.fit(torch.cat(references),torch.cat(teachers),model.phase.cpu());model.to(args.device)
    decoder=model.visual_decoder.weight.detach().cpu().numpy()
    importance=np.abs(decoder).mean(0)*mean
    keep=np.argsort(importance)[-max(1,d//2):];masked=np.zeros(d);masked[keep]=importance[keep]
    if masked.sum()<=0:raise ValueError('No visual-sensitive neurons')
    line=line.to(args.device);esum=np.zeros(7);esq=np.zeros(7);ecount=0
    for i,r in enumerate(out):
        z=section(zmap,r);sl=slice(r['offset'],r['offset']+r['length'])
        with torch.inference_mode():evidence[sl,0]=line(torch.tensor(np.array(z),device=args.device))[0].cpu().numpy()
        raw_state=frames[r['source_offset']:r['source_offset']+r['length'],:14]
        evidence[sl,2:]=time_evidence(raw_state,z,scale,masked)
        if r['split']=='train':
            e=section(evidence,r).astype('float64');esum+=e.sum(0);esq+=(e*e).sum(0);ecount+=len(e)
        if i%100==0:print(json.dumps(dict(stage='evidence',episode=i,total=len(out),seconds=time.monotonic()-started)),flush=True)
    emean=esum/ecount;escale=np.maximum(np.sqrt(np.maximum(esq/ecount-emean**2,0)),[.001,.001,.05,.05,.05,.05,.05])
    for array in (zmap,phase,evidence):
        if not np.isfinite(array).all():raise FloatingPointError('Nonfinite extracted features')
        array.flush()
    norm=dict(z_mean=torch.from_numpy(mean.astype('float32')),z_scale=torch.from_numpy(scale.astype('float32')),
        evidence_mean=torch.from_numpy(emean.astype('float32')),evidence_scale=torch.from_numpy(escale.astype('float32')),
        visual_importance=torch.from_numpy(masked.astype('float32')),line=line.cpu().state_dict(),line_fit=fit,
        input_normalization=source['normalization'],fit_split='train',fit_frames=count,dimension=d,
        evidence_names=EVIDENCE_NAMES,checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=signature['checkpoint_sha256'],
        dropout_samples=4,phase_dropout=model.config.phase_dropout,inference_precision='FP32; matmul and cuDNN TF32 disabled')
    torch.save(norm,folder/'normalization.pt')
    for q in quality.values():
        n=q['frames'];q.update(teacher_kl=q['teacher_kl_sum']/n,teacher_agreement=q['teacher_agreement_sum']/n,
                            visual_loss=q['visual_loss_sum']/n,state_loss=q['state_loss_sum']/n)
    result=dict(complete=True,signature=signature,records=out,total_frames=total,dimension=d,backbone_epoch=ck['epoch'],
        checkpoint=str(args.checkpoint.resolve()),data=str(args.data.resolve()),quality=quality,
        selection='fixed final backbone; normal-only feature statistics; no evaluation selection',
        normalization_sha256=sha(folder/'normalization.pt'),seconds=time.monotonic()-started)
    save_json(result,folder/'manifest.json');return result


def train_fold(args,manifest,fold):
    variant=getattr(args,'variant','full')
    root=args.output if variant=='full' else args.output/'ablations'/variant
    folder=root/f'fold_{fold}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():
        old=json.loads((folder/'metrics.json').read_text())
        if old['steps']!=args.steps or old['smoke']!=args.smoke:raise ValueError('Risk protocol mismatch')
        return old
    records=manifest['records'];norm=load(args.output/'normalization.pt');d=norm['dimension']
    zs=np.load(args.output/'latent.npy',mmap_mode='r');es=np.load(args.output/'evidence.npy',mmap_mode='r')
    data=[]
    for r in records:
        z=(torch.from_numpy(np.array(section(zs,r)))-norm['z_mean'])/norm['z_scale']
        e=(torch.from_numpy(np.array(section(es,r)))-norm['evidence_mean'])/norm['evidence_scale']
        e=e.clamp(-15,15)
        zero={'full':[],'no_line':[0],'no_uncertainty':[1],'no_loop':[2,3,4,5,6]}[variant]
        if zero:e[:,zero]=0
        data.append((z.clamp(-15,15),e))
    splits={s:[i for i,r in enumerate(records) if r['split']==s] for s in ('train','calibration','validation','ood')}
    heldout=None if fold=='all' else int(fold)
    positives=[i for i in splits['ood'] if records[i]['episode_index']!=heldout]
    tests=[i for i in splits['ood'] if records[i]['episode_index']==heldout]
    tasks=sorted({task_of(records[i]) for i in splits['train']});pools={}
    for task in tasks:
        ids=[i for i in splits['train'] if task_of(records[i])==task]
        rank=lambda v:np.argsort(np.argsort(v,kind='stable'),kind='stable')/max(1,len(v)-1)
        hardness=np.maximum(rank([section(es,records[i])[:,0].max() for i in ids]),rank([section(es,records[i])[:,2].max() for i in ids]))
        pools[task]=[i for i,h in zip(ids,hardness) if h>=.8] or ids
    torch.manual_seed(29);model=SingleRiskHead(d).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.01)
    rng=np.random.default_rng(29);order=rng.permutation(splits['train']).tolist();cursor=0;seen=set();history=[];started=time.monotonic()
    for step in range(1,args.steps+1):
        if cursor+4>len(order):order=order[cursor:]+rng.permutation(splits['train']).tolist();cursor=0
        ids=order[cursor:cursor+4];cursor+=4
        ids += [int(rng.choice(pools[tasks[(step+j)%len(tasks)]])) for j in range(2)]
        ids += positives;seen.update(ids)
        z=pad_sequence([data[i][0] for i in ids],batch_first=True).to(args.device)
        e=pad_sequence([data[i][1] for i in ids],batch_first=True).to(args.device)
        t=torch.arange(z.shape[1],device=args.device);length=torch.tensor([records[i]['length'] for i in ids],device=args.device)
        valid=t[None]<length[:,None];positive=torch.tensor([i in positives for i in ids],device=args.device)
        for row,i in enumerate(ids):
            if i in positives and records[i]['episode_index']==0:valid[row]&=t>=int(records[i]['length']*2/3)
        model.train();optimizer.zero_grad(set_to_none=True);logits=model(z,e)
        loss,bags=weak_episode_loss(logits,valid,positive)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite risk loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%100==0 or step==args.steps:
            row=dict(step=step,loss=float(loss.detach()),positive=float(bags[positive].sigmoid().mean().detach()),
                     negative=float(bags[~positive].sigmoid().mean().detach()),seconds=time.monotonic()-started)
            history.append(row);print(json.dumps(dict(stage='risk_train',fold=fold,**row)),flush=True)
    if not args.smoke and len(set(splits['train'])&seen)!=2246:raise AssertionError('Not all normal training episodes visited')
    model.eval();predictions={}
    with torch.inference_mode():
        for i in splits['calibration']+splits['validation']+splits['ood']:
            z,e=data[i];raw=model(z[None].to(args.device),e[None].to(args.device))[0].sigmoid().cpu().numpy()
            risk=intervention_trace(raw);risk[:15]=0
            predictions[i]=dict(frame_risk_score=raw,risk_score=risk,bag_score=float(risk.max()))
    threshold=float(np.quantile([predictions[i]['bag_score'] for i in splits['calibration']],.95,method='higher'))
    def group(ids):
        hit=[predictions[i]['risk_score']>threshold for i in ids];n=sum(len(x) for x in hit)
        return dict(episodes=len(ids),alarmed_episodes=sum(bool(x.any()) for x in hit),
            episode_alarm_fraction=float(np.mean([x.any() for x in hit])) if hit else None,
            frames=n,alarmed_frames=sum(int(x.sum()) for x in hit),frame_alarm_fraction=sum(int(x.sum()) for x in hit)/n if n else None)
    metrics=dict(fold=fold,variant=variant,zeroed_evidence=zero,dimension=d,backbone_epoch=manifest['backbone_epoch'],steps=args.steps,seed=29,smoke=args.smoke,
        threshold=threshold,threshold_source='95% normal calibration episode maximum, higher quantile; strict > alarm',
        calibration=group(splits['calibration']),normal_test=group(splits['validation']),
        normal_by_task={t:group([i for i in splits['validation'] if task_of(records[i])==t]) for t in tasks},
        positive_training_episode_indices=[records[i]['episode_index'] for i in positives],
        positive_test_episode_indices=[records[i]['episode_index'] for i in tests],normal_training_episodes_seen=len(set(splits['train'])&seen),
        positive_tests=[],label_semantics='weak episode candidates; episode0 held-in last-third only, prefix unknown',
        seconds=time.monotonic()-started,normalization_sha256=manifest['normalization_sha256'])
    if tests:
        metrics['episode_auroc']=episode_metrics([predictions[i]['bag_score'] for i in tests],[predictions[i]['bag_score'] for i in splits['validation']])['auroc']
    for i in tests:
        p=predictions[i];alarm=p['risk_score']>threshold;where=np.flatnonzero(alarm);tail=int(len(alarm)*2/3)
        metrics['positive_tests'].append(dict(episode_index=records[i]['episode_index'],detected=bool(alarm.any()),
            first_alarm_frame=int(where[0]) if len(where) else None,alarm_fraction=float(alarm.mean()),bag_score=p['bag_score'],
            last_third_start=tail,last_third_alarm_frames=int(alarm[tail:].sum()),earlier_alarm_frames=int(alarm[:tail].sum())))
    saved={}
    for i in splits['validation']+splits['ood']:
        p=predictions[i];tag='ood_oof_test' if i in tests else ('ood_training_only' if i in positives else 'id_test')
        saved[records[i]['file']]=dict(**{k:torch.from_numpy(v) for k,v in p.items() if isinstance(v,np.ndarray)},
            alarm=torch.from_numpy(p['risk_score']>threshold),split=tag,metadata=records[i])
    torch.save(saved,folder/'predictions.pt')
    torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},dimension=d,hidden=32,threshold=threshold,
        backbone=manifest['checkpoint'],backbone_sha256=manifest['signature']['checkpoint_sha256'],
        normalization=str((args.output/'normalization.pt').resolve()),normalization_sha256=manifest['normalization_sha256'],
        preprocessing=dict(smooth_frames=30,persist_frames=15,minimum_history=15),results=metrics,history=history,
        code_sha256=sha(Path(__file__))),folder/'checkpoint.pt')
    save_json(history,folder/'history.json');save_json(metrics,folder/'metrics.json');return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--folds',nargs='+',default=['0','1','2','all'],choices=['0','1','2','all'])
    p.add_argument('--variant',choices=['full','no_line','no_uncertainty','no_loop'],default='full')
    p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    if a.steps!=1200 and not a.smoke:raise ValueError('Official fixed protocol is 1200 risk steps per fold')
    manifest=prepare(a);metrics=[train_fold(a,manifest,f) for f in a.folds]
    root=a.output if a.variant=='full' else a.output/'ablations'/a.variant
    save_json(dict(complete=True,smoke=a.smoke,variant=a.variant,dimension=manifest['dimension'],backbone_epoch=manifest['backbone_epoch'],
                   folds=metrics,quality=manifest['quality']),root/'evaluation.json')
    print('OOD_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
