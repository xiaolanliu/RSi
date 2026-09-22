"""Fit a normal-only projection memory and calibrate three-signal OOD."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .fit_ood_v4 import load, save_json, sha, section, inference_precision
from .three_signal_ood import ProjectionMemory, ThreeSignalRisk, SIGNAL_NAMES, sustained_trace, calibration_tail


def summary(raw, threshold, preprocessing):
    smooth, score = sustained_trace(raw, preprocessing)
    alarm = score > threshold
    where = np.flatnonzero(alarm)
    return dict(frame_risk_score=torch.from_numpy(raw), smoothed_risk_score=torch.from_numpy(smooth),
                risk_score=torch.from_numpy(score), alarm=torch.from_numpy(alarm)), dict(
        frames=len(raw), alarm=bool(alarm.any()), alarm_frames=int(alarm.sum()),
        first_alarm_frame=int(where[0]) if len(where) else None,
        peak_score=float(score.max()), peak_frame=int(score.argmax()),
        peak_threshold_ratio=float(score.max()/threshold))


@torch.inference_mode()
def fit(args):
    torch.set_num_threads(2); inference_precision()
    args.output.mkdir(parents=True, exist_ok=True)
    target=args.bundle
    if target.exists():raise FileExistsError(f'Refusing to overwrite fitted bundle: {target}')
    base=load(args.base); norm=base['normalization']
    source=args.source; manifest=json.loads((source/'manifest.json').read_text()); records=manifest['records']
    if sha(source/'normalization.pt')!=base['provenance']['normalization_sha256']:
        raise ValueError('Cached features and bundle normalization differ')
    if manifest['signature']['checkpoint_sha256']!=base['provenance']['backbone_sha256']:
        raise ValueError('Cached features and bundle encoder differ')
    zmap=np.load(source/'latent.npy',mmap_mode='r'); emap=np.load(source/'evidence.npy',mmap_mode='r')
    train=[(i,r) for i,r in enumerate(records) if r['split']=='train']
    sampled=[]; sample_episode=[]; bank_indices=[]; bank_episode=[]
    for i,r in train:
        sampled.extend(r['offset']+np.linspace(0,r['length']-1,min(128,r['length']),dtype=int))
        sample_episode.extend([i]*min(128,r['length']))
        ix=np.unique(np.linspace(0,r['length']-1,min(8,r['length']),dtype=int))
        bank_indices.extend(r['offset']+ix);bank_episode.extend([i]*len(ix))
    sampled=np.asarray(sampled);bank_indices=np.asarray(bank_indices)
    def standardize(z):
        return ((torch.as_tensor(np.array(z),dtype=torch.float32)-norm['z_mean'])/norm['z_scale']).clamp(-15,15)
    normal=standardize(zmap[sampled]).double(); center=normal.mean(0); delta=normal-center
    eig, vectors=torch.linalg.eigh(delta.T@delta/max(1,len(normal)-1))
    projection=vectors[:,-32:].flip(-1).float(); center=center.float()
    reference=(standardize(zmap[bank_indices])-center)@projection
    memory=ProjectionMemory(center,projection,reference,torch.tensor(bank_episode)).to(args.device)
    retained=float(eig[-32:].sum()/eig.sum())
    def distances(z, groups=None):
        outputs=[]
        for start in range(0,len(z),1024):
            part=z[start:start+1024].to(args.device)
            group=None if groups is None else groups[start:start+1024]
            outputs.append(memory(part,group).cpu())
        return torch.cat(outputs)
    started=time.monotonic();print(json.dumps(dict(stage='normal_reference',samples=len(sampled),references=len(reference),retained_variance=retained)),flush=True)
    d=distances(normal.float(),torch.tensor(sample_episode,device=args.device))
    fit_e=torch.column_stack((d,torch.from_numpy(np.array(emap[sampled,2])),torch.from_numpy(np.array(emap[sampled,0]))))
    reference=fit_e.T.contiguous().sort(-1).values
    head=ThreeSignalRisk(reference=reference)
    quantiles=np.quantile(fit_e.numpy(),[.5,.95,.99],axis=0).tolist()
    print(json.dumps(dict(stage='empirical_quantiles',p50_p95_p99=quantiles,seconds=time.monotonic()-started)),flush=True)
    predictions={};calibration=[]
    selected=[r for r in records if r['split'] in ('calibration','validation','ood')]
    for i,r in enumerate(selected):
        d=distances(standardize(section(zmap,r)))
        e=torch.column_stack((d,torch.from_numpy(np.array(section(emap,r)[:,2])),torch.from_numpy(np.array(section(emap,r)[:,0]))))
        raw=head(e).numpy(); smooth,score=sustained_trace(raw,base['preprocessing'])
        predictions[r['file']]=dict(evidence=e,frame_risk_score=torch.from_numpy(raw),
            smoothed_risk_score=torch.from_numpy(smooth),risk_score=torch.from_numpy(score))
        if r['split']=='calibration':calibration.append(float(score.max()))
        if i%50==0:print(json.dumps(dict(stage='score',episode=i,total=len(selected),seconds=time.monotonic()-started)),flush=True)
    calibration=np.sort(np.asarray(calibration,dtype='float32'))
    alpha=.05
    # Exact boundary for finite-sample rank p <= alpha; ties do not alarm.
    order=int(np.ceil((len(calibration)+1)*(1-alpha)))-1
    if order>=len(calibration):raise ValueError('Insufficient calibration episodes for requested tail fraction')
    threshold=float(calibration[order])
    if not np.isfinite(threshold) or threshold<=0:raise ValueError('Invalid calibrated threshold')
    rows=[]
    for r in selected:
        p=predictions[r['file']]; _,row=summary(p['frame_risk_score'].numpy(),threshold,base['preprocessing'])
        p['alarm']=p['risk_score']>threshold
        p['risk_percentile']=torch.from_numpy((1-calibration_tail(p['risk_score'].numpy(),calibration)).astype('float32'))
        p['signal_percentiles']=head.percentiles(p['evidence'])
        np.testing.assert_array_equal(p['alarm'],calibration_tail(p['risk_score'].numpy(),calibration)<=alpha)
        if r['split']=='ood' and r['episode_index']==0:
            row['last_third_alarm']=bool(p['alarm'][int(r['length']*2/3):].any())
        rows.append(dict(id=r.get('review_id') or r['file'],file=r['file'],split=r['split'],**row))
    # External failure episodes are evaluated only after the normal-only fit and
    # threshold are fixed; none supplies a projection, scale, or threshold.
    external=[]
    for path in sorted(args.external.glob('*_predictions.npz')):
        if '_rgb_' in path.name:continue
        old=np.load(path);name=path.stem.removesuffix('_predictions')
        d=distances(standardize(old['latent']))
        e=torch.column_stack((d,torch.from_numpy(old['evidence'][:,2].copy()),torch.from_numpy(old['evidence'][:,0].copy())))
        p,row=summary(head(e).numpy(),threshold,base['preprocessing']);p['evidence']=e
        p['risk_percentile']=torch.from_numpy((1-calibration_tail(p['risk_score'].numpy(),calibration)).astype('float32'))
        p['signal_percentiles']=head.percentiles(e)
        predictions[name]=p;external.append(dict(id=name,split='external_failure',**row))
    def aggregate(split):
        group=[r for r in rows if r['split']==split]
        return dict(episodes=len(group),alarmed_episodes=sum(r['alarm'] for r in group),
            episode_alarm_rate=float(np.mean([r['alarm'] for r in group])),
            frame_alarm_rate=sum(r['alarm_frames'] for r in group)/sum(r['frames'] for r in group))
    protocol=dict(fit_split='normal train only',reference_episodes=len(train),reference_frames=len(bank_indices),
        pca_fit_frames=len(sampled),projection_dim=32,retained_variance=retained,
        scale_fit='normal train 128 evenly spaced samples/episode; nearest search excludes own episode',
        reference_selection='8 evenly spaced observed features per normal training trajectory',
        projection='normal standardized/clipped shared state-motion-visual latent; unwhitened PCA32',
        combination='empirical training tail p=(1+count(reference>=value))/(N+1); max(-log10(p))',
        threshold_source='normal calibration episode-maximum tail rank <= 5%; finite-sample correction; conservative ties',
        abnormal_fit_episodes=0,source_manifest_sha256=sha(source/'manifest.json'),base_bundle_sha256=sha(args.base))
    result=dict(complete=True,threshold=threshold,signal_names=SIGNAL_NAMES,
        p50_p95_p99=quantiles,alarm_tail_fraction=alpha,protocol=protocol,
        calibration=aggregate('calibration'),validation=aggregate('validation'),ood=aggregate('ood'),
        external=dict(episodes=len(external),alarmed_episodes=sum(r['alarm'] for r in external)),
        records=rows,external_records=external,seconds=time.monotonic()-started)
    for k in ('risk','risk_hidden'):base.pop(k,None)
    for k in ('evidence_mean','evidence_scale','evidence_names','dropout_samples','phase_dropout'):norm.pop(k,None)
    base.update(risk_kind='three_signal_v8',projection_memory={k:v.cpu() for k,v in memory.state_dict().items()},
        signal_normalization=dict(reference=reference),threshold=threshold,
        calibration_reference=torch.from_numpy(calibration),alarm_tail_fraction=alpha,
        ood_protocol=protocol,signal_names=SIGNAL_NAMES)
    base['provenance']['previous_risk_results']=base['provenance'].pop('risk_results',None)
    base['provenance']['risk_results']={k:result[k] for k in ('calibration','validation','ood','external')}
    base['provenance']['risk_checkpoint_sha256']=None
    target.parent.mkdir(parents=True,exist_ok=True);torch.save(base,target)
    result['bundle_sha256']=sha(target)
    torch.save(predictions,args.output/'predictions.pt');save_json(result,args.output/'evaluation.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','external_records')},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('runs/ood_v4_2000_20260920/dim128/ood'))
    p.add_argument('--base',type=Path,default=Path('models/v7/fold_all.pt'))
    p.add_argument('--external',type=Path,default=Path('runs/pant_fail_v7_20260921'))
    p.add_argument('--output',type=Path,default=Path('runs/three_signal_v8_20260921'))
    p.add_argument('--bundle',type=Path,default=Path('models/v8/fold_all.pt'))
    p.add_argument('--device',default='cuda:0');fit(p.parse_args())


if __name__=='__main__':main()
