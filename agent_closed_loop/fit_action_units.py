"""Replace only the repetition signal; retain normal-only quantile calibration."""
import json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .fit_ood_v4 import load,save_json,sha
from .paths import artifact_path
from .action_units import UnitConfig,unit_trace,repetition_strength
from .three_signal_ood import ThreeSignalRisk,sustained_trace,calibration_tail
from .fit_three_signal_ood import summary


def main(version='v9', *, source=None, base_path=None, old_path=None, output=None,
         target=None, external_root=None, skip_external=False):
    torch.set_num_threads(2)
    run='action_dynamics_v9_20260921' if version=='v9' else 'early_repetition_v10_20260921'
    root=Path(output) if output is not None else Path('runs')/run;root.mkdir(exist_ok=True,parents=True)
    target=Path(target) if target is not None else Path(f'models/{version}/fold_all.pt')
    if target.exists():raise FileExistsError(f'Refusing to overwrite fitted bundle: {target}')
    source=Path(source or 'runs/ood_v4_2000_20260920/dim128/ood');manifest=json.loads((source/'manifest.json').read_text())
    base_path=Path(base_path or 'models/v8/fold_all.pt')
    frames=np.load(artifact_path(manifest['data'])/'frames.npy',mmap_mode='r');base=load(base_path)
    old=load(old_path or 'runs/three_signal_v8_20260921/predictions.pt');records=manifest['records']
    sample_ids=np.concatenate([r['source_offset']+np.linspace(0,r['length']-1,min(128,r['length']),dtype=int) for r in records if r['split']=='train'])
    limits=np.quantile(frames[sample_ids][:,[12,13]],[.05,.95],axis=0)
    config=UnitConfig(close=tuple(limits[0]+.25*(limits[1]-limits[0])),open=tuple(limits[0]+.65*(limits[1]-limits[0])))
    if version=='v10':
        # Fixed physical time scales before evaluating the revised failure
        # traces. All empirical references and alarm boundaries are refitted.
        config.recency_frames=90.
        base['preprocessing']=dict(smooth_frames=10,persist_frames=5,minimum_history=15)
    if (root/'unit_config.json').exists():
        if json.loads((root/'unit_config.json').read_text())!=json.loads(json.dumps(asdict(config))):
            raise ValueError('Cached action-unit configuration differs')
    save_json(asdict(config),root/'unit_config.json')
    cache=root/'unit_scores.npy';event_path=root/'review_events.json';started=time.monotonic()
    if cache.exists() and event_path.exists():
        scores=np.load(cache,mmap_mode='r');events=json.loads(event_path.read_text())
    else:
        scores=np.lib.format.open_memmap(cache,mode='w+',dtype='float32',shape=(len(frames),));events={}
        for i,r in enumerate(records):
            start=r['source_offset'];stop=start+r['length'];x=np.array(frames[start:stop,:76])
            scores[start:stop],ev=unit_trace(x[:,:14],x[:,28:76],config)
            if r.get('review_id'):events[r['review_id']]=ev
            if i%100==0:print(json.dumps(dict(stage='action_units',episode=i,total=len(records),seconds=time.monotonic()-started)),flush=True)
        scores.flush();save_json(events,event_path)
    reference=base['signal_normalization']['reference'].clone()
    repeat_reference=dict(periodic=reference[1].numpy().copy(),units=np.sort(np.array(scores[sample_ids])))
    legacy_evidence=np.load(source/'evidence.npy',mmap_mode='r')
    sample_offsets=np.concatenate([r['offset']+np.linspace(0,r['length']-1,min(128,r['length']),dtype=int) for r in records if r['split']=='train'])
    if version=='v10':
        from .uncertainty_ood import FeatureLINe
        base['line_route_tolerance']=1e-5
        line=FeatureLINe(base['encoder_config']['dim'],route_tolerance=base['line_route_tolerance']).eval()
        line.load_state_dict(base['normalization']['line'])
        latent=np.load(source/'latent.npy',mmap_mode='r')
        @torch.inference_mode()
        def line_energy(z):
            return torch.cat([line(torch.from_numpy(np.array(z[j:j+4096])))[0] for j in range(0,len(z),4096)])
        reference[2]=line_energy(latent[sample_offsets]).sort().values
    combined=repetition_strength(legacy_evidence[sample_offsets,2],scores[sample_ids],repeat_reference)
    reference[1]=torch.from_numpy(np.sort(combined))
    head=ThreeSignalRisk(reference=reference);predictions={};cal=[]
    for r in records:
        if r['split']=='train':continue
        p=old[r['file']];e=p['evidence'].clone()
        if version=='v10':e[:,2]=line_energy(latent[r['offset']:r['offset']+r['length']])
        e[:,1]=torch.from_numpy(repetition_strength(e[:,1].numpy(),scores[r['source_offset']:r['source_offset']+r['length']],repeat_reference))
        raw=head(e).numpy();smooth,s=sustained_trace(raw,base['preprocessing'])
        predictions[r['file']]=dict(evidence=e,frame_risk_score=torch.from_numpy(raw),smoothed_risk_score=torch.from_numpy(smooth),risk_score=torch.from_numpy(s))
        if r['split']=='calibration':cal.append(float(s.max()))
    cal=np.sort(np.array(cal,dtype='float32'));threshold=float(cal[int(np.ceil((len(cal)+1)*.95))-1])
    rows=[]
    def finish(key,split,metadata):
        p=predictions[key];p['alarm']=p['risk_score']>threshold
        p['signal_percentiles']=head.percentiles(p['evidence'])
        p['risk_percentile']=torch.from_numpy((1-calibration_tail(p['risk_score'].numpy(),cal)).astype('float32'))
        _,row=summary(p['frame_risk_score'].numpy(),threshold,base['preprocessing'])
        if metadata.get('id')=='test_lwy_ep000000':row['last_third_alarm']=bool(p['alarm'][len(p['alarm'])*2//3:].any())
        return dict(split=split,**metadata,**row)
    for r in records:
        if r['split']!='train':rows.append(finish(r['file'],r['split'],dict(id=r.get('review_id') or r['file'],file=r['file'])))
    external=[];norm=base['normalization']['input_normalization'];mean=np.array(norm['mean']);scale=np.array(norm['scale'])
    external_root=Path(external_root or 'runs/pant_fail_v7_20260921')
    for i in ([] if skip_external else range(4)):
        name=f'pant_fail_ep{i:06d}';state=np.load(external_root/f'{name}_input.npz')['states'];visual=np.load(external_root/'visual'/f'episode_{i:06d}.npz')['features']
        normalized_s=np.clip((state-mean[:14])/scale[:14],-15,15).astype('float32');normalized_v=np.clip((visual-mean[28:76])/scale[28:76],-15,15).astype('float32')
        loop,ev=unit_trace(normalized_s,normalized_v,config);events[name]=ev
        e=old[name]['evidence'].clone()
        if version=='v10':e[:,2]=line_energy(np.load(external_root/f'{name}_predictions.npz')['latent'])
        e[:,1]=torch.from_numpy(repetition_strength(e[:,1].numpy(),loop,repeat_reference));raw=head(e).numpy();smooth,s=sustained_trace(raw,base['preprocessing'])
        predictions[name]=dict(evidence=e,frame_risk_score=torch.from_numpy(raw),smoothed_risk_score=torch.from_numpy(smooth),risk_score=torch.from_numpy(s))
        external.append(finish(name,'external_failure',dict(id=name)))
    def aggregate(split):
        g=[r for r in rows if r['split']==split]
        return dict(episodes=len(g),alarmed_episodes=sum(r['alarm'] for r in g),episode_alarm_rate=float(np.mean([r['alarm'] for r in g])),
            frame_alarm_rate=sum(r['alarm_frames'] for r in g)/sum(r['frames'] for r in g))
    names=['projection_distance','action_unit_repetition','line_energy']
    base.update(risk_kind=f'three_signal_{version}',unit_config=asdict(config),signal_names=names,
        signal_normalization=dict(reference=reference),calibration_reference=torch.from_numpy(cal),threshold=threshold,
        repetition_reference={k:torch.from_numpy(v) for k,v in repeat_reference.items()})
    base['ood_protocol'].update(repetition='per-arm observed close-to-close units; arc-length alignment; accumulated body/visual endpoint recurrence',
        action_semantics='observed execution only; recorded action==state is not an independent command',
        unit_config=asdict(config),base_v8_sha256=sha(base_path))
    base['ood_protocol']['repetition_fusion']='max of normal-tail strengths for gripper units and gripper-free periodic motion; recalibrated as one repetition criterion'
    if version=='v10':
        base['ood_protocol'].update(repetition_recency='exp(-(current_end-old_end)/90); causal decay between completed units',
            line_routing='stable lowest-index route within 1e-5 logit tolerance; normal training energy distribution and calibration recomputed',
            preprocessing=base['preprocessing'],
            parameter_selection='3-second recency and mean10/min5 fixed from causal latency design; no failure-onset labels or failure-tuned thresholds',
            evaluation_scope='pant_fail and existing validation have been inspected in prior versions; regression evaluation, not new independent generalization')
    result=dict(complete=True,threshold=threshold,signal_names=names,protocol=base['ood_protocol'],alarm_tail_fraction=.05,
        calibration=aggregate('calibration'),validation=aggregate('validation'),ood=aggregate('ood'),
        external=dict(episodes=len(external),alarmed_episodes=sum(r['alarm'] for r in external)),records=rows,external_records=external,seconds=time.monotonic()-started)
    base['provenance']['risk_results']={k:result[k] for k in ('calibration','validation','ood','external')}
    target.parent.mkdir(parents=True,exist_ok=True);torch.save(base,target);result['bundle_sha256']=sha(target)
    save_json(result,root/'evaluation.json');save_json(events,root/'review_events.json');torch.save(predictions,root/'predictions.pt')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','external_records','protocol')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['v9','v10'],default='v9')
    main(parser.parse_args().version)
