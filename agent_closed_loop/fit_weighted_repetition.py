"""Increase repetition priority after ranking; recalibrate with normal episodes."""
from pathlib import Path
import json
import numpy as np
import torch
from .fit_ood_v4 import load,save_json,sha
from .fit_three_signal_ood import summary
from .three_signal_ood import ThreeSignalRisk,calibration_tail,sustained_trace


@torch.inference_mode()
def main(*, base_path=Path('models/v10/fold_all.pt'),
         oldroot=Path('runs/early_repetition_v10_20260921'),
         root=Path('runs/command_constraints_v11_20260921'), target=Path('models/v11/fold_all.pt')):
    torch.set_num_threads(2)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    target=Path(target);base_path=Path(base_path);oldroot=Path(oldroot)
    if target.exists():raise FileExistsError(target)
    b=load(base_path)
    old=load(oldroot/'predictions.pt');ev=json.loads((oldroot/'evaluation.json').read_text())
    # Fixed before inspecting the weighted failure traces; final calibration
    # uses the same normal-only split and alpha as V10.
    weights=torch.tensor([1.,1.5,1.]);b['signal_normalization']['weights']=weights
    head=ThreeSignalRisk(**b['signal_normalization']);pred={}
    for key,p in old.items():
        raw=head(p['evidence']).numpy();sm,score=sustained_trace(raw,b['preprocessing'])
        pred[key]=dict(evidence=p['evidence'],signal_percentiles=head.percentiles(p['evidence']),
            frame_risk_score=torch.from_numpy(raw),smoothed_risk_score=torch.from_numpy(sm),risk_score=torch.from_numpy(score))
    cal=np.sort(np.array([float(pred[r['file']]['risk_score'].max()) for r in ev['records'] if r['split']=='calibration'],dtype='float32'))
    threshold=float(cal[int(np.ceil((len(cal)+1)*.95))-1])
    def finish(r,key):
        p=pred[key];p['alarm']=p['risk_score']>threshold
        p['risk_percentile']=torch.tensor(1-calibration_tail(p['risk_score'].numpy(),cal),dtype=torch.float32)
        _,row=summary(p['frame_risk_score'].numpy(),threshold,b['preprocessing'])
        ans={k:r[k] for k in ['id','file','split'] if k in r};ans.update(row)
        if 'last_third_alarm' in r:ans['last_third_alarm']=bool(p['alarm'][len(p['alarm'])*2//3:].any())
        return ans
    rows=[finish(r,r['file']) for r in ev['records']];external=[finish(r,r['id']) for r in ev['external_records']]
    def aggregate(split):
        g=[r for r in rows if r['split']==split]
        return dict(episodes=len(g),alarmed_episodes=sum(r['alarm'] for r in g),episode_alarm_rate=float(np.mean([r['alarm'] for r in g])),
            frame_alarm_rate=sum(r['alarm_frames'] for r in g)/sum(r['frames'] for r in g))
    b.update(risk_kind='three_signal_v11',threshold=threshold,calibration_reference=torch.from_numpy(cal))
    b['ood_protocol'].update(criterion_weights=weights.tolist(),combination='max(c_distance,1.5*c_repeat,c_LINE); c=-log10(normal training tail)',
        weight_selection='User requested increased repetition penalty; fixed 1.5 before weighted failure evaluation.',
        command_evidence_status='Real failure-run command logs audited separately; no normal command calibration, not included in the OOD alarm.',
        base_v10_sha256=sha(base_path))
    result=dict(complete=True,threshold=threshold,signal_names=b['signal_names'],criterion_weights=weights.tolist(),protocol=b['ood_protocol'],alarm_tail_fraction=.05,
        calibration=aggregate('calibration'),validation=aggregate('validation'),ood=aggregate('ood'),
        external=dict(episodes=len(external),alarmed_episodes=sum(r['alarm'] for r in external)),records=rows,external_records=external)
    b['provenance']['risk_results']={k:result[k] for k in ['calibration','validation','ood','external']}
    target.parent.mkdir(parents=True,exist_ok=True);torch.save(b,target);result['bundle_sha256']=sha(target)
    torch.save(pred,root/'predictions.pt');save_json(result,root/'evaluation.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','protocol']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
