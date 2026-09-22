"""Fixed visual-input sensitivity probes; not real unknown-task OOD metrics.

Hold recorded state/motion fixed and alter a 120-frame visual interval. These
inputs are synthetic, not labelled failures. No fitting, tuning, or calibration
uses their scores. This checks whether trained outputs actually use vision.
"""
from .paths import artifact_path
import json
from pathlib import Path
import numpy as np
import torch
from .fit_ood_v4 import encode_episode,episode_seed,section,load,save_json,task_of
from .ood_v4_inference import OODV4Stream
from .streaming_fusion import readout_disagreement
from .temporal_evidence import time_evidence
from .recovery import intervention_trace,episode_metrics
from .prepare_ood_v4 import sha
from .uncertainty_ood import SingleRiskHead


@torch.inference_mode()
def score(model,values,seed):
    z=encode_episode(model.encoder,values,model.device).to(model.device)
    energy,logits=model.line(z)
    u=readout_disagreement(z,model.encoder.phase,torch.arange(len(z),device=model.device),seed)
    h=time_evidence(values[:,:14],z.cpu().numpy(),model.norm['z_scale'].numpy(),model.norm['visual_importance'].numpy())
    e=torch.cat((energy[:,None],u[:,None],torch.tensor(h,device=model.device)),-1)
    raw=model.head(((z-model.zmean)/model.zscale).clamp(-15,15)[None],((e-model.emean)/model.escale).clamp(-15,15)[None])[0].sigmoid().cpu().numpy()
    risk=intervention_trace(raw);risk[:15]=0
    return dict(z=z.cpu().numpy(),phase=logits.softmax(-1).cpu().numpy(),energy=energy.cpu().numpy(),uncertainty=u.cpu().numpy(),risk=risk,raw=raw)


def diagnose(folder,device):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    target=folder/'visual_sensitivity.json';checkpoint=folder/'fold_0/checkpoint.pt';signature=sha(checkpoint)
    if target.exists():
        old=json.loads(target.read_text())
        if old['checkpoint_sha256']!=signature:raise ValueError('Stale visual diagnostic')
        return old
    frames=np.load(artifact_path(manifest['data'])/'frames.npy',mmap_mode='r');model=OODV4Stream(checkpoint,device)
    records=[r for r in manifest['records'] if r['split']=='validation'];tasks=sorted({task_of(r) for r in records})
    selected=[r for task in tasks for r in [r for r in records if task_of(r)==task][:2]]
    rows=[]
    for r in selected:
        values=np.array(frames[r['source_offset']:r['source_offset']+r['length'],:76]);seed=episode_seed(r)
        base=score(model,values,seed);begin=len(values)//3;end=min(len(values),begin+120)
        # Include causal propagation and persistent-score washout after edit.
        region=slice(begin,min(len(values),end+model.encoder.config.warmup+84+44))
        for variant in ('visual_at_training_mean','visual_four_sigma_shift'):
            changed=values.copy()
            if variant=='visual_at_training_mean':changed[begin:end,28:76]=0
            else:
                signs=np.where(np.arange(48)%2,1.,-1.)
                changed[begin:end,28:76]=np.clip(changed[begin:end,28:76]+4*signs,-15,15)
            new=score(model,changed,seed)
            np.testing.assert_allclose(base['risk'][:begin],new['risk'][:begin],atol=1e-6,rtol=1e-6)
            rows.append(dict(record=r['file'],task=task_of(r),variant=variant,begin=begin,end_exclusive=end,
                latent_relative_change=float(np.linalg.norm(new['z'][begin:end]-base['z'][begin:end])/max(np.linalg.norm(base['z'][begin:end]),1e-8)),
                phase_mean_l1_change=float(np.abs(new['phase'][begin:end]-base['phase'][begin:end]).sum(-1).mean()),
                line_mean_change=float((new['energy'][begin:end]-base['energy'][begin:end]).mean()),
                uncertainty_mean_change=float((new['uncertainty'][begin:end]-base['uncertainty'][begin:end]).mean()),
                baseline_risk_max=float(base['risk'][region].max()),modified_risk_max=float(new['risk'][region].max()),
                baseline_alarm_frames=int((base['risk'][region]>model.threshold).sum()),
                modified_alarm_frames=int((new['risk'][region]>model.threshold).sum())))
    result=dict(dimension=manifest['dimension'],backbone_epoch=manifest['backbone_epoch'],checkpoint_sha256=signature,
        threshold=model.threshold,normal_test_episodes=len(selected),rows=rows,
        semantics='fixed synthetic visual sensitivity only; no failure labels, no OOD accuracy, no threshold or hyperparameter tuning',
        intervention='120 frames at 1/3 episode: standardized visual set to zero OR fixed alternating +/-4 training standard deviations; state/delta unchanged')
    save_json(result,target);return result


@torch.inference_mode()
def diagnose_readout_seeds(folder,device):
    """Keep trained heads and original thresholds; change only M=4 masks.

    Seed 0 exercises the streaming API default, and 29 is a second fixed
    seed. Neither is selected for deployment based on these test results.
    """
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    path=folder/'readout_seed_sensitivity.json'
    signatures={str(i):sha(folder/f'fold_{i}/checkpoint.pt') for i in range(3)}
    if path.exists():
        old=json.loads(path.read_text())
        if old['checkpoint_sha256']!=signatures:raise ValueError('Stale sampling diagnostic')
        return old
    stream=OODV4Stream(folder/'fold_0/checkpoint.pt',device);models={};thresholds={};baseline={}
    for i in range(3):
        ck=load(folder/f'fold_{i}/checkpoint.pt');model=SingleRiskHead(ck['dimension'],ck['hidden']).to(device).eval()
        model.load_state_dict(ck['model']);models[i]=model;thresholds[i]=ck['threshold'];baseline[i]=load(folder/f'fold_{i}/predictions.pt')
    records=[r for r in manifest['records'] if r['split'] in ('validation','ood')]
    zmap=np.load(folder/'latent.npy',mmap_mode='r');emap=np.load(folder/'evidence.npy',mmap_mode='r')
    scores={seed:{fold:dict(normal=[],positive=[],frame_alarm_changes=0,frames=0,max_score_difference=0.) for fold in range(3)} for seed in (0,29)}
    for r in records:
        z=torch.tensor(np.array(section(zmap,r)),device=device);e=torch.tensor(np.array(section(emap,r)),device=device)
        zn=((z-stream.zmean)/stream.zscale).clamp(-15,15)[None]
        folds=[r['episode_index']] if r['split']=='ood' else [0,1,2]
        for seed in (0,29):
            e[:,1]=readout_disagreement(z,stream.encoder.phase,torch.arange(len(z),device=device),seed)
            en=((e-stream.emean)/stream.escale).clamp(-15,15)[None]
            for fold in folds:
                raw=models[fold](zn,en)[0].sigmoid().cpu().numpy();risk=intervention_trace(raw);risk[:15]=0
                ref=baseline[fold][r['file']];dst=scores[seed][fold];alarm=risk>thresholds[fold]
                dst['positive' if r['split']=='ood' else 'normal'].append(dict(record=r['file'],max_score=float(risk.max()),detected=bool(alarm.any())))
                dst['frame_alarm_changes']+=int((alarm!=ref['alarm'].numpy()).sum());dst['frames']+=len(risk)
                dst['max_score_difference']=max(dst['max_score_difference'],float(np.abs(risk-ref['risk_score'].numpy()).max()))
    result=[]
    for seed,folds in scores.items():
        rows=[]
        for fold,s in folds.items():
            rows.append(dict(fold=fold,threshold=thresholds[fold],normal_episodes=len(s['normal']),
                normal_alarmed_episodes=sum(x['detected'] for x in s['normal']),positive_test=s['positive'],
                episode_auroc=episode_metrics([x['max_score'] for x in s['positive']],[x['max_score'] for x in s['normal']])['auroc'],
                frame_alarm_changes=s['frame_alarm_changes'],compared_frames=s['frames'],max_risk_difference=s['max_score_difference']))
        result.append(dict(seed=seed,folds=rows,detected=sum(r['positive_test'][0]['detected'] for r in rows),
            mean_normal_episode_alarm_fraction=float(np.mean([r['normal_alarmed_episodes']/r['normal_episodes'] for r in rows])),
            macro_episode_auroc=float(np.mean([r['episode_auroc'] for r in rows]))))
    out=dict(dimension=manifest['dimension'],checkpoint_sha256=signatures,seeds=result,
        baseline='deterministic episode-identity/frame masks used in training, calibration and main report',
        semantics='same frozen models and same original thresholds; change only dropout sampling seed; no selection/tuning')
    save_json(out,path);return out
