"""Select a conservative update on normal calibration trajectories only."""
import argparse,json
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import torch
from .duration_subtask import DurationConfig,DurationOrderedHead,ordered_duration
from .fit_ood_v4 import load,save_json,sha,section,inference_precision
from .paths import artifact_path
from .evaluate_change_subtask import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(2);inference_precision();ck=load(a.stage);source=Path(ck['protocol']['source']);m=json.loads((source/'manifest.json').read_text())
    norm=load(source/'normalization.pt');zm=np.load(source/'latent.npy',mmap_mode='r');frames=np.load(artifact_path(m['data'])/'frames.npy',mmap_mode='r')
    base=DurationConfig(**ck['config']);model=DurationOrderedHead(base).to(a.device).eval();model.load_state_dict(ck['model'])
    records=[r for r in m['records'] if r['split']=='calibration'];features={}
    with torch.inference_mode():
        for r in records:
            z=torch.tensor(np.array(section(zm,r)),device=a.device);z=((z-norm['z_mean'].to(a.device))/norm['z_scale'].to(a.device)).clamp(-15,15)
            out,_=model.features(z[None]);features[r['file']]=(out['logits'],(out['change_sizes'][...,1:].amax(-1)>=base.change_floor).float())
        rows=[]
        for threshold in (.65,.75,.77,.8,.85,.9,.95,.975):
            for width in (5,10,15):
                cfg=replace(base,confirmation_threshold=threshold,evidence_frames=width);pred={}
                for r in records:
                    x,s=features[r['file']];out,_=ordered_duration(x,s,cfg)
                    pred[r['file']]={k:out[k][0].cpu().numpy() for k in ('phase_probs','confirmed_phase')}
                quality=summarize(records,lambda r:pred[r['file']],frames)
                row=dict(config=asdict(cfg),quality=quality);rows.append(row)
                print(json.dumps(dict(threshold=threshold,width=width,quality=quality)),flush=True)
    old=json.loads(Path('runs/latent_change_v6_20260920/memory_evaluation/evaluation.json').read_text())['quality']['calibration']
    eligible=[r for r in rows if r['quality']['confirmed_teacher_agreement']>=old['confirmed_teacher_agreement'] and
              r['quality']['early_more_than_60_frames']<=old['early_more_than_60_frames'] and
              r['quality']['missed_boundary_fraction']<=.03 and r['quality']['completed_durations_under_60']<=old['completed_durations_under_60']]
    result=dict(stage_sha256=sha(a.stage),baseline_v6=old,candidates=rows,
        policy='normal calibration only: agreement >= V6, early>60 <= V6, duration<60 <= V6, missed<=3%; maximize agreement then minimize absolute boundary error',
        selected=max(eligible,key=lambda r:(r['quality']['confirmed_teacher_agreement'],-r['quality']['matched_boundary_absolute_error_frames']['p50'])) if eligible else None)
    a.output.mkdir(parents=True,exist_ok=True);save_json(result,a.output/'calibration_sweep.json')
    if result['selected']:
        ck['training_config']=ck['config'];ck['config']=result['selected']['config'];ck['calibration']=dict(policy=result['policy'],split='calibration',source_stage_sha256=sha(a.stage))
        torch.save(ck,a.output/'calibrated.pt');print('SELECTED',json.dumps(result['selected']),flush=True)
    else:print('NO ELIGIBLE CONFIGURATION',flush=True)

if __name__=='__main__':main()
