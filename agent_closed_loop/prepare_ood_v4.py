"""Pack raw causal inputs and ID-only teacher labels, retaining split provenance."""
from .paths import artifact_path,PROJECT
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with open(artifact_path(path),'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('runs/recovery_line_v2_2000/features'))
    p.add_argument('--teacher-split',type=Path,default=PROJECT/'data/teacher_split_epoch2000.json')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    if (a.output/'manifest.json').exists():raise FileExistsError('Prepared output already exists')
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.source/'manifest.json').read_text());records=manifest['records']
    if not manifest['complete']:raise ValueError('Incomplete source')
    original=json.loads(a.teacher_split.read_text())
    teacher_train={(r['source_root'],r['episode_index']) for r in original['train']}
    for r in records:
        if r['split'] in ('calibration','validation','ood') and (r['source_root'],r['episode_index']) in teacher_train:
            raise ValueError('Teacher train overlaps evaluation')
    total=sum(r['length'] for r in records)
    data=np.lib.format.open_memmap(a.output/'frames.npy',mode='w+',dtype='float32',shape=(total,81))
    sums=np.zeros(76);squares=np.zeros(76);count=0;offset=0;out=[]
    seen=set();start=time.monotonic()
    for i,r in enumerate(records):
        key=(r['source_root'],r['episode_index'])
        if key in seen:raise ValueError('Duplicate source episode')
        seen.add(key)
        arr=torch.load(a.source/r['file'],map_location='cpu',weights_only=True)
        s=arr['states'].numpy();v=arr['features'][:,-48:].numpy()
        ds=np.zeros_like(s);ds[1:]=s[1:]-s[:-1]
        x=np.concatenate((s,ds,v),axis=1)
        if not np.isfinite(x).all():raise ValueError('Nonfinite input')
        y=arr['phase_teacher'].numpy();y=y/np.maximum(y.sum(-1,keepdims=True),1e-8)
        if len(x)!=r['length']:raise ValueError('Length mismatch')
        data[offset:offset+len(x),:76]=x
        # OOD teacher labels are intentionally inaccessible to training/evaluation.
        data[offset:offset+len(x),76:]=y if r['split']!='ood' else 0
        if r['split']=='train':
            xd=x.astype('float64');sums+=xd.sum(0);squares+=(xd*xd).sum(0);count+=len(x)
        out.append(dict(r,offset=offset,source_sha256=sha(a.source/r['file'])))
        offset+=len(x)
        if i%200==0:print(json.dumps(dict(packed=i,total=len(records),seconds=time.monotonic()-start)),flush=True)
    mean=sums/count;scale=np.sqrt(np.maximum(squares/count-mean*mean,0))
    floor=np.r_[np.full(14,1e-3),np.full(14,1e-5),np.full(48,1e-3)];scale=np.maximum(scale,floor)
    for r in out:
        sl=slice(r['offset'],r['offset']+r['length'])
        data[sl,:76]=np.clip((data[sl,:76]-mean)/scale,-15,15)
    data.flush()
    norm=dict(mean=mean.tolist(),scale=scale.tolist(),fit_split='train',fit_frames=count,input_columns=['state14','observed_delta14','visual48'],teacher_columns=5)
    (a.output/'normalization.json').write_text(json.dumps(norm,indent=2))
    result=dict(complete=True,source=str(a.source.resolve()),source_manifest_sha256=sha(a.source/'manifest.json'),records=out,
        total_frames=total,normalization=norm,data_sha256=sha(a.output/'frames.npy'),
        teacher_provenance=manifest['signature'],teacher_train_disjoint_from_evaluation=True,
        teacher_split_sha256=sha(a.teacher_split),
        input_semantics='state, observed backward difference, raw pooled visual; no old hidden features or teacher as input',
        epochs_semantics='one pass over ALL training target frames; warmup is context, never counted twice',seconds=time.monotonic()-start)
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(complete=True,frames=total,train_frames=count)),flush=True)


if __name__=='__main__':main()
