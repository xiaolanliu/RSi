"""Evaluate fixed V7 checkpoints; calibration and test are separate operations."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .duration_subtask import DurationConfig,DurationOrderedHead
from .fit_ood_v4 import load,save_json,sha,section,inference_precision
from .paths import artifact_path
from .evaluate_change_subtask import summarize


def predict(model,z,pause=None,chunk=257):
    memory=None;rows=[]
    for start in range(0,len(z),chunk):
        part=z[start:start+chunk][None]
        alarm=None if pause is None else torch.as_tensor(pause[start:start+chunk],device=z.device,dtype=torch.bool)[None]
        out,memory=model(part,memory,alarm);rows.append({k:v[0].cpu() for k,v in out.items()})
    return {k:torch.cat([r[k] for r in rows]) for k in rows[0]}


def evaluate(stage,output,split,device,review=False):
    ck=load(stage)
    if ck['epoch']!=ck['protocol']['epochs'] or ck['protocol']['smoke']:raise ValueError('Require final full-data checkpoint')
    config=DurationConfig(**ck['config']);source=Path(ck['protocol']['source'])
    m=json.loads((source/'manifest.json').read_text());norm=load(source/'normalization.pt')
    zm=np.load(source/'latent.npy',mmap_mode='r');frames=np.load(artifact_path(m['data'])/'frames.npy',mmap_mode='r')
    model=DurationOrderedHead(config).to(device).eval();model.load_state_dict(ck['model'])
    records=[r for r in m['records'] if r['split']==split];predictions={}
    def latent(r):
        z=torch.tensor(np.array(section(zm,r)),device=device)
        return ((z-norm['z_mean'].to(device))/norm['z_scale'].to(device)).clamp(-15,15)
    with torch.inference_mode():
        for r in records:
            p=predict(model,latent(r));predictions[r['file']]={k:v.numpy() for k,v in p.items() if k in ('phase_probs','confirmed_phase')}
    metrics=summarize(records,lambda r:predictions[r['file']],frames)
    metrics['posterior_expected_phase_backtracks']=sum(int((np.diff(p['phase_probs']@np.arange(1,6))<-1e-5).sum()) for p in predictions.values())
    metrics['probability_argmax_backtracks']=sum(int((np.diff(p['phase_probs'].argmax(-1))<0).sum()) for p in predictions.values())
    output.mkdir(parents=True,exist_ok=True)
    result=dict(complete=True,split=split,config=ck['config'],quality=metrics,stage=str(stage.resolve()),
                stage_sha256=sha(stage),protocol=ck['protocol'],code_sha256=sha(Path(__file__).parent/'duration_subtask.py'))
    save_json(result,output/f'{split}.json');print(json.dumps(result,ensure_ascii=False),flush=True)
    if review:
        risks={k:load(source/f'fold_{k}/predictions.pt') for k in range(3)};saved={}
        with torch.inference_mode():
            for r in m['records']:
                if not r.get('review_id'):continue
                fold=r['episode_index'] if r['split']=='ood' else 0;risk=risks[fold][r['file']]
                expected='ood_oof_test' if r['split']=='ood' else 'id_test'
                if risk['split']!=expected:raise ValueError('Review split leakage')
                p=predict(model,latent(r),risk['alarm'].numpy())
                p['accepted_phase']=p['confirmed_phase'].masked_fill(risk['alarm'],0)
                p.update({k:risk[k] for k in ('risk_score','frame_risk_score','alarm')})
                p.update(fold=fold,split=expected,record=r['file'])
                saved[r['review_id']]=p
        torch.save(saved,output/'review_predictions.pt')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['calibration','validation','train'],required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--review',action='store_true');a=p.parse_args();torch.set_num_threads(2);inference_precision()
    evaluate(a.stage,a.output,a.split,a.device,a.review)

if __name__=='__main__':main()
