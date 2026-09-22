"""Normal-only calibration, fixed-checkpoint testing, and causal replay export."""
import argparse,json,time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import torch
from .change_subtask import ChangeConfig,LatentChangeHead,FreshStageMemory
from .online_subtask import StageConfirmation
from .fit_ood_v4 import load,save_json,sha,section,inference_precision
from .paths import artifact_path


def prepare(stage_checkpoint,output,device):
    ck=load(stage_checkpoint)
    if ck['protocol']['smoke'] or ck['epoch']!=60:raise ValueError('Require final formal 60-epoch checkpoint')
    source=artifact_path(ck['protocol']['source']);m=json.loads((source/'manifest.json').read_text());norm=load(source/'normalization.pt')
    output.mkdir(parents=True,exist_ok=True)
    signature=dict(stage_sha256=sha(stage_checkpoint),source_manifest_sha256=sha(source/'manifest.json'),code_sha256=sha(Path(__file__).parent/'change_subtask.py'))
    if (output/'manifest.json').exists():
        old=json.loads((output/'manifest.json').read_text())
        if old['signature']!=signature:raise ValueError('Stale prediction cache')
        return old
    model=LatentChangeHead(ChangeConfig(**ck['config'])).to(device).eval();model.load_state_dict(ck['model'])
    zmap=np.load(source/'latent.npy',mmap_mode='r');n=m['total_frames']
    signals=np.lib.format.open_memmap(output/'signals.npy',mode='w+',dtype='float32',shape=(n,7))
    recent=np.lib.format.open_memmap(output/'recent_latent.npy',mode='w+',dtype='float32',shape=(n,m['dimension']))
    started=time.monotonic()
    with torch.inference_mode():
        for i,r in enumerate(m['records']):
            z=torch.tensor(np.array(section(zmap,r)),device=device)
            z=((z-norm['z_mean'].to(device))/norm['z_scale'].to(device)).clamp(-15,15)
            memory=None;offset=r['offset']
            for chunk in z.split(257):
                out,memory=model(chunk[None],memory);length=chunk.shape[0]
                signals[offset:offset+length]=torch.cat((out['boundary_event_probs'],out['change_sizes']),-1)[0].cpu().numpy()
                recent[offset:offset+length]=out['recent_latent'][0].cpu().numpy();offset+=length
            if i%250==0:print(json.dumps(dict(stage='features',episode=i,seconds=time.monotonic()-started)),flush=True)
    signals.flush();recent.flush()
    result=dict(signature=signature,stage_checkpoint=str(stage_checkpoint.resolve()),source=str(source.resolve()),
        config=ck['config'],records=m['records'],total_frames=n,data=str(artifact_path(m['data']).resolve()),complete=True)
    save_json(result,output/'manifest.json');return result


def replay(signals,recent,config,alarm=None,details=False):
    control=FreshStageMemory(config);rows=[]
    for t,(signal,z) in enumerate(zip(signals,recent)):
        rows.append(control.step(signal[:4],signal[4:],z,False if alarm is None else bool(alarm[t])))
    keys=['phase_probs','confirmed_phase','accepted_phase']
    if details:keys+=['boundary_event_probs','boundary_evidence','sustained_boundary_evidence','confirmation_count','latent_change','anchor_distance','stage_age','transition_armed']
    return {k:np.asarray([r[k] for r in rows]) for k in keys}


def summarize(records,get_prediction,frames):
    total=0;agree=0;soft_agree=0;expected=0;missed=0;false=0;errors=[];durations=[];changes=0;back=0;skips=0;last=[0]*5
    for r in records:
        pred=get_prediction(r);hard=pred['confirmed_phase'];q=pred['phase_probs']
        teacher=np.array(frames[r['source_offset']:r['source_offset']+r['length'],76:]);truth=teacher.argmax(-1)+1
        total+=len(hard);agree+=int((hard==truth).sum());soft_agree+=int((q.argmax(-1)+1==truth).sum())
        cdf=np.maximum.accumulate(np.clip(1-teacher.cumsum(-1)[:,:4],0,1),axis=0)
        previous=0
        for k in range(4):
            target=np.flatnonzero(cdf[:,k]>=.5);actual=np.flatnonzero(hard>=k+2)
            if len(target):
                expected+=1
                if len(actual):errors.append(int(actual[0]-target[0]))
                else:missed+=1
            elif len(actual):false+=1
            if len(actual):durations.append(int(actual[0]-previous));previous=int(actual[0])
        d=np.diff(hard);changes+=int((d>0).sum());back+=int((d<0).sum());skips+=int((d>1).sum());last[hard[-1]-1]+=1
    arr=np.asarray(errors);d=np.asarray(durations)
    quant=lambda a:dict(p10=float(np.quantile(a,.1)),p50=float(np.median(a)),p90=float(np.quantile(a,.9))) if len(a) else None
    return dict(episodes=len(records),frames=total,confirmed_teacher_agreement=agree/total,soft_argmax_teacher_agreement=soft_agree/total,
        expected_boundaries=expected,missed_boundaries=missed,missed_boundary_fraction=missed/max(1,expected),false_boundaries=false,
        early_more_than_60_frames=int((arr<-60).sum()),early_boundary_fraction=float((arr<-60).sum()/max(1,expected)),
        matched_boundary_error_frames=quant(arr),matched_boundary_absolute_error_frames=quant(np.abs(arr)),
        confirmed_changes=changes,confirmed_backtracks=back,confirmed_skips=skips,final_stage_histogram=last,
        completed_durations=quant(d),completed_duration_std=float(d.std()) if len(d) else None,
        completed_durations_under_60=int((d<60).sum()),completed_durations_under_100=int((d<100).sum()),
        completed_durations_near_200=int((np.abs(d-200)<=10).sum()),completed_durations_over_500=int((d>500).sum()))


def evaluate(a,m):
    signals=np.load(a.output/'signals.npy',mmap_mode='r');recent=np.load(a.output/'recent_latent.npy',mmap_mode='r')
    frames=np.load(Path(m['data'])/'frames.npy',mmap_mode='r');records=m['records'];base=ChangeConfig(**m['config'])
    splits={s:[r for r in records if r['split']==s] for s in ('train','calibration','validation','ood')}
    def predicted(r,config):return replay(section(signals,r),section(recent,r),config)
    sweep=[]
    for threshold in (.55,.65,.75,.85):
        cfg=replace(base,confirmation_threshold=threshold)
        metrics=summarize(splits['calibration'],lambda r:predicted(r,cfg),frames)
        row=dict(threshold=threshold,**metrics);sweep.append(row)
        print(json.dumps(dict(stage='normal_only_calibration',**row)),flush=True)
    # User objective is less premature acceptance. Do not select the most
    # permissive threshold merely because it improves aggregate frame fit.
    # Require >=10% fewer >60-frame early crossings than V5 on NORMAL CAL,
    # while missing <=3% of observed teacher boundaries; then maximize fit.
    old=np.load('runs/online_subtask_v5_20260920/ood/ordered_phase.npy',mmap_mode='r')
    def baseline(r):
        p=section(old,r);control=StageConfirmation();hard=np.array([control.step(c)[0] for c in p[:,5:]])
        return dict(phase_probs=p[:,:5],confirmed_phase=hard)
    baseline_calibration=summarize(splits['calibration'],baseline,frames)
    eligible=[r for r in sweep if r['early_boundary_fraction']<=.9*baseline_calibration['early_boundary_fraction'] and r['missed_boundary_fraction']<=.03]
    if not eligible:raise ValueError('No normal-calibration setting reduces premature acceptance within the missed-boundary budget')
    selected=max(eligible,key=lambda r:(r['confirmed_teacher_agreement'],r['threshold']))
    config=replace(base,confirmation_threshold=selected['threshold'])
    quality={}
    for split in ('train','validation'):
        quality[split]=summarize(splits[split],lambda r:predicted(r,config),frames)
        print(json.dumps(dict(stage='phase_quality',split=split,**quality[split])),flush=True)
    quality['calibration']={k:v for k,v in selected.items() if k!='threshold'}
    # V5 reference is replayed with OOD off for the identical phase diagnostic.
    v5=summarize(splits['validation'],baseline,frames)
    # Review uses held-out OOD risk folds; risk mechanism/weights are unchanged V4.
    risk_root=Path(m['source']);predictions={k:load(risk_root/f'fold_{k}/predictions.pt') for k in range(3)};saved={}
    for r in records:
        if not r.get('review_id'):continue
        fold=r['episode_index'] if r['split']=='ood' else 0;risk=predictions[fold][r['file']]
        expected='ood_oof_test' if r['split']=='ood' else 'id_test'
        if risk['split']!=expected:raise ValueError('Review split leakage')
        p=replay(section(signals,r),section(recent,r),config,risk['alarm'].numpy(),details=True)
        saved[r['review_id']]=dict(**{k:torch.from_numpy(v) for k,v in p.items()},risk_score=risk['risk_score'],
            frame_risk_score=risk['frame_risk_score'],alarm=risk['alarm'],fold=fold,split=expected,record=r['file'])
    torch.save(saved,a.output/'review_predictions.pt')
    result=dict(complete=True,config=asdict(config),calibration_sweep=sweep,quality=quality,baseline_v5=v5,baseline_calibration=baseline_calibration,
        calibration_policy='Normal CAL only: >=10% fewer teacher-relative early crossings than V5; <=3% missed teacher boundaries; maximize confirmed-phase teacher agreement among eligible settings',
        phase_checkpoint_sha256=m['signature']['stage_sha256'],risk='unchanged V4 single head; independent of committed phase to avoid feedback distribution mismatch',
        interpretation='All phase labels/early/missing metrics use offline teacher, not human truth. Recording end does not force P5.')
    save_json(result,a.output/'evaluation.json');print('CHANGE_EVALUATION_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();torch.set_num_threads(2);inference_precision();m=prepare(a.stage,a.output,a.device);evaluate(a,m)

if __name__=='__main__':main()
