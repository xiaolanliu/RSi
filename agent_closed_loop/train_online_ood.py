"""Freeze ordered subtask + shared encoder; fit one OOD head with phase memory.

Normal-only subtask fitting and normalization; three whole-episode OOD folds.
Only episode0's held-IN last third has coarse positive supervision. Its prefix
is unknown. This does not supervise or certify irrecoverability probabilities.
"""
from .paths import artifact_path
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .fit_ood_v4 import load,save_json,sha,section,task_of,inference_precision
from .online_subtask import OrderedConfig,OrderedSubtaskHead,OnlineRiskHead,phase_context,StageConfirmation
from .recovery import weak_episode_loss,intervention_trace,episode_metrics


def prepare(a):
    a.output.mkdir(parents=True,exist_ok=True)
    source=json.loads((a.source/'manifest.json').read_text());ck=load(a.phase);norm=load(a.source/'normalization.pt')
    if ck['protocol']['smoke'] or ck['epoch']!=ck['protocol']['epochs']:raise ValueError('Require completed formal subtask checkpoint')
    if ck['protocol']['source_manifest_sha256']!=sha(a.source/'manifest.json'):raise ValueError('Source mismatch')
    signature=dict(phase_sha256=sha(a.phase),source_manifest_sha256=sha(a.source/'manifest.json'),
        normalization_sha256=sha(a.source/'normalization.pt'),code={n:sha(Path(__file__).parent/n) for n in ('online_subtask.py','train_online_ood.py')})
    if (a.output/'manifest.json').exists():
        old=json.loads((a.output/'manifest.json').read_text())
        if old['signature']!=signature:raise ValueError('Stale cached ordered predictions')
        return old
    model=OrderedSubtaskHead(OrderedConfig(**ck['config'])).to(a.device).eval();model.load_state_dict(ck['model'])
    zmap=np.load(a.source/'latent.npy',mmap_mode='r');frames=np.load(artifact_path(source['data'])/'frames.npy',mmap_mode='r')
    phase=np.lib.format.open_memmap(a.output/'ordered_phase.npy',mode='w+',dtype='float32',shape=(source['total_frames'],9))
    quality={s:dict(frames=0,teacher_agreement_sum=0,teacher_kl_sum=0.,cdf_absolute_sum=0.,episodes=0,
        raw_argmax_changes=0,raw_argmax_backtracks=0,confirmed_backtracks=0,confirmed_skips=0,confirmed_changes=0,
        final_confirmed_histogram=[0]*5) for s in ('train','calibration','validation')}
    started=time.monotonic()
    with torch.inference_mode():
        for i,r in enumerate(source['records']):
            z=torch.tensor(np.array(section(zmap,r)),device=a.device)
            z=((z-norm['z_mean'].to(a.device))/norm['z_scale'].to(a.device)).clamp(-15,15)
            # Deliberately carry state across arbitrary chunks for every record.
            cache=None;rows=[]
            for chunk in z.split(257):
                out,cache=model(chunk[None],cache);rows.append(phase_context(out)[0].cpu().numpy())
            p=np.concatenate(rows);phase[r['offset']:r['offset']+r['length']]=p
            if not np.isfinite(p).all() or (np.diff(p[:,5:],axis=0)<-1e-7).any():raise ValueError('Invalid ordered phase')
            if r['split']!='ood':
                y=np.array(frames[r['source_offset']:r['source_offset']+r['length'],76:]);q=quality[r['split']]
                target=np.clip(1-y.cumsum(-1)[:,:4],0,1);hard=p[:,:5].argmax(-1)
                controller=StageConfirmation(model.config.confirmation_threshold,model.config.confirmation_frames)
                confirmed=np.array([controller.step(c)[0] for c in p[:,5:]])
                q['frames']+=len(p);q['episodes']+=1;q['teacher_agreement_sum']+=int((hard==y.argmax(-1)).sum())
                q['teacher_kl_sum']+=float((y*(np.log(y.clip(1e-8))-np.log(p[:,:5].clip(1e-6)))).sum())
                q['cdf_absolute_sum']+=float(np.abs(p[:,5:]-target).sum())
                q['raw_argmax_changes']+=int((np.diff(hard)!=0).sum());q['raw_argmax_backtracks']+=int((np.diff(hard)<0).sum())
                q['confirmed_changes']+=int((np.diff(confirmed)>0).sum());q['confirmed_backtracks']+=int((np.diff(confirmed)<0).sum())
                q['confirmed_skips']+=int((np.diff(confirmed)>1).sum());q['final_confirmed_histogram'][confirmed[-1]-1]+=1
            if i%200==0:print(json.dumps(dict(stage='ordered_phase',episode=i,total=len(source['records']),seconds=time.monotonic()-started)),flush=True)
    phase.flush()
    for q in quality.values():
        q['teacher_agreement']=q['teacher_agreement_sum']/q['frames'];q['teacher_kl']=q['teacher_kl_sum']/q['frames']
        q['boundary_cdf_mae']=q['cdf_absolute_sum']/(4*q['frames']);q['mean_raw_argmax_changes']=q['raw_argmax_changes']/q['episodes']
        q['mean_confirmed_changes']=q['confirmed_changes']/q['episodes']
    result=dict(complete=True,signature=signature,source=str(a.source.resolve()),phase_checkpoint=str(a.phase.resolve()),
        config=ck['config'],phase_epochs=ck['epoch'],backbone_epoch=source['backbone_epoch'],dimension=source['dimension'],
        records=source['records'],total_frames=source['total_frames'],quality=quality,seconds=time.monotonic()-started)
    save_json(result,a.output/'manifest.json');return result


def train_fold(a,m,fold):
    folder=a.output/f'fold_{fold}';folder.mkdir(exist_ok=True)
    if (folder/'metrics.json').exists():return json.loads((folder/'metrics.json').read_text())
    norm=load(a.source/'normalization.pt');records=m['records'];dimension=m['dimension']
    zmap=np.load(a.source/'latent.npy',mmap_mode='r');emap=np.load(a.source/'evidence.npy',mmap_mode='r')
    pmap=np.load(a.output/'ordered_phase.npy',mmap_mode='r');data=[]
    for r in records:
        z=(torch.tensor(np.array(section(zmap,r)))-norm['z_mean'])/norm['z_scale']
        e=(torch.tensor(np.array(section(emap,r)))-norm['evidence_mean'])/norm['evidence_scale']
        data.append((z.clamp(-15,15),e.clamp(-15,15),torch.tensor(np.array(section(pmap,r)))))
    splits={s:[i for i,r in enumerate(records) if r['split']==s] for s in ('train','calibration','validation','ood')}
    positives=[i for i in splits['ood'] if fold=='all' or records[i]['episode_index']!=int(fold)]
    tests=[] if fold=='all' else [i for i in splits['ood'] if records[i]['episode_index']==int(fold)]
    tasks=sorted({task_of(records[i]) for i in splits['train']});pools={}
    for task in tasks:
        ids=[i for i in splits['train'] if task_of(records[i])==task]
        rank=lambda x:np.argsort(np.argsort(x,kind='stable'),kind='stable')/max(1,len(x)-1)
        hardness=np.maximum(rank([section(emap,records[i])[:,0].max() for i in ids]),rank([section(emap,records[i])[:,2].max() for i in ids]))
        pools[task]=[i for i,h in zip(ids,hardness) if h>=.8]
    torch.manual_seed(29);model=OnlineRiskHead(dimension).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.01)
    rng=np.random.default_rng(29);order=rng.permutation(splits['train']).tolist();cursor=0;seen=set();history=[];started=time.monotonic()
    for step in range(1,a.steps+1):
        if cursor+4>len(order):order=order[cursor:]+rng.permutation(splits['train']).tolist();cursor=0
        ids=order[cursor:cursor+4];cursor+=4
        ids+=[int(rng.choice(pools[tasks[(step+j)%len(tasks)]])) for j in range(2)];ids+=positives;seen.update(ids)
        z,e,p=[pad_sequence([data[i][j] for i in ids],batch_first=True).to(a.device) for j in range(3)]
        t=torch.arange(z.shape[1],device=a.device);lengths=torch.tensor([records[i]['length'] for i in ids],device=a.device)
        valid=t[None]<lengths[:,None];positive=torch.tensor([i in positives for i in ids],device=a.device)
        for row,i in enumerate(ids):
            if i in positives and records[i]['episode_index']==0:valid[row]&=t>=int(records[i]['length']*2/3)
        model.train();optimizer.zero_grad(set_to_none=True);logits=model(z,e,p);loss,bags=weak_episode_loss(logits,valid,positive)
        if not torch.isfinite(loss):raise FloatingPointError('Risk loss nonfinite')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%100==0:
            row=dict(step=step,loss=float(loss.detach()),seconds=time.monotonic()-started);history.append(row);print(json.dumps(dict(fold=fold,**row)),flush=True)
    assert len(set(splits['train'])&seen)==2246
    model.eval();predictions={}
    with torch.inference_mode():
        for i in splits['calibration']+splits['validation']+tests:
            z,e,p=data[i];raw=model(z[None].to(a.device),e[None].to(a.device),p[None].to(a.device))[0].sigmoid().cpu().numpy()
            score=intervention_trace(raw);score[:15]=0;predictions[i]=dict(frame_risk_score=raw,risk_score=score)
    threshold=float(np.quantile([predictions[i]['risk_score'].max() for i in splits['calibration']],.95,method='higher'))
    def group(ids):
        hits=[predictions[i]['risk_score']>threshold for i in ids]
        return dict(episodes=len(ids),alarmed_episodes=sum(h.any() for h in hits).item(),episode_alarm_fraction=float(np.mean([h.any() for h in hits])))
    metrics=dict(fold=fold,steps=a.steps,seed=29,normal_training_episodes_seen=2246,threshold=threshold,
        normal_test=group(splits['validation']),calibration=group(splits['calibration']),
        normal_by_task={t:group([i for i in splits['validation'] if task_of(records[i])==t]) for t in tasks},
        positive_training_episode_indices=[records[i]['episode_index'] for i in positives],positive_tests=[])
    if tests:metrics['episode_auroc']=episode_metrics([predictions[i]['risk_score'].max() for i in tests],[predictions[i]['risk_score'].max() for i in splits['validation']])['auroc']
    for i in tests:
        score=predictions[i]['risk_score'];hits=np.flatnonzero(score>threshold)
        metrics['positive_tests'].append(dict(episode_index=records[i]['episode_index'],detected=bool(len(hits)),
            first_alarm_frame=int(hits[0]) if len(hits) else None,alarm_frames=len(hits),last_third_alarm_frames=int((score[len(score)*2//3:]>threshold).sum())))
    saved={}
    for i in splits['validation']+tests:
        p=predictions[i];r=records[i];alarm=p['risk_score']>threshold
        control=StageConfirmation(m['config']['confirmation_threshold'],m['config']['confirmation_frames'])
        phases=[control.step(c,bool(hit)) for c,hit in zip(section(pmap,r)[:,5:],alarm)]
        confirmed,accepted=np.asarray(phases,dtype=np.int64).T
        assert ((np.diff(confirmed)>=0)&(np.diff(confirmed)<=1)).all()
        saved[r['file']]=dict(**{k:torch.from_numpy(v) for k,v in p.items()},alarm=torch.from_numpy(alarm),
            confirmed_phase=torch.from_numpy(confirmed),accepted_phase=torch.from_numpy(accepted),
            split='ood_oof_test' if i in tests else 'id_test')
    checkpoint=dict(model={k:v.cpu() for k,v in model.state_dict().items()},dimension=dimension,hidden=32,threshold=threshold,
        source=str(a.source.resolve()),phase_checkpoint=m['phase_checkpoint'],phase_sha256=m['signature']['phase_sha256'],
        source_manifest_sha256=m['signature']['source_manifest_sha256'],normalization_sha256=m['signature']['normalization_sha256'],
        preprocessing=dict(smooth_frames=30,persist_frames=15,minimum_history=15),results=metrics,history=history)
    torch.save(checkpoint,folder/'checkpoint.pt');torch.save(saved,folder/'predictions.pt');save_json(metrics,folder/'metrics.json')
    return metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--phase',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=1200)
    a=p.parse_args();torch.set_num_threads(2);inference_precision()
    if a.steps!=1200:raise ValueError('Fixed matched risk protocol is 1200 steps')
    m=prepare(a);folds=[train_fold(a,m,f) for f in ['0','1','2','all']]
    save_json(dict(complete=True,folds=folds,quality=m['quality']),a.output/'evaluation.json');print('ONLINE_OOD_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
