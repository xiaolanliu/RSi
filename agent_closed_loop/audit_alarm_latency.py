"""Explain alarm latency with cached causal evidence; no failure-onset labels."""
import json
from pathlib import Path
import numpy as np
import torch
from .fit_ood_v4 import load,save_json
from .three_signal_ood import sustained_trace,calibration_tail


def first(mask):
    ix=np.flatnonzero(mask)
    return int(ix[0]) if len(ix) else None


def alarm_segments(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,dtype=bool),False].astype(int))
    return [(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]


def main():
    torch.set_num_threads(2)
    paths=dict(v9=Path('runs/action_dynamics_v9_20260921'),v10=Path('runs/early_repetition_v10_20260921'))
    pred={k:load(v/'predictions.pt') for k,v in paths.items()}
    bundles={k:load(f'models/{k}/fold_all.pt') for k in paths}
    metrics={k:json.loads((v/'evaluation.json').read_text()) for k,v in paths.items()}
    events={k:json.loads((v/'review_events.json').read_text()) for k,v in paths.items()}
    fast_config=dict(smooth_frames=10,persist_frames=5,minimum_history=15)
    # Ablation: shortening confirmation alone must be recalibrated too.
    fast={k:sustained_trace(p['frame_risk_score'].numpy(),fast_config)[1] for k,p in pred['v9'].items()}
    maxima=np.sort(np.array([fast[r['file']].max() for r in metrics['v9']['records'] if r['split']=='calibration'],dtype='float32'))
    fast_threshold=float(maxima[int(np.ceil((len(maxima)+1)*.95))-1])
    rows=[]
    for i in range(4):
        key=f'pant_fail_ep{i:06d}';r=dict(id=key)
        for version in paths:
            p=pred[version][key];th=bundles[version]['threshold']
            values={name:first(p[name].numpy()>th) for name in ['frame_risk_score','smoothed_risk_score','risk_score']}
            start=values['risk_score']
            segments=alarm_segments(p['alarm'])
            stable=next((a for a,b in segments if b-a>=15),None)
            r[version]=dict(threshold=th,first_crossing_frames=values,first_alarm_seconds=None if start is None else start/30,
                alarm_frames=int(p['alarm'].sum()),signal_percentiles_at_alarm=None if start is None else p['signal_percentiles'][start].tolist(),
                alarm_segments=segments,first_alarm_segment_frames=0 if not segments else segments[0][1]-segments[0][0],
                first_segment_at_least_15_frames=stable,
                risk_percentile_at_alarm=None if start is None else float(p['risk_percentile'][start]),
                unit_events_before_first_alarm=[] if start is None else [e for e in events[version][key] if e['end']<=start][-6:])
        r['v9_shorter_filter_first_alarm_frame']=first(fast[key]>fast_threshold)
        a=r['v9']['first_alarm_seconds'];b=r['v10']['first_alarm_seconds']
        r['first_alarm_advance_seconds']=None if a is None or b is None else a-b
        rows.append(r)
    ablation={}
    for split in ['calibration','validation','ood']:
        selected=[fast[r['file']]>fast_threshold for r in metrics['v9']['records'] if r['split']==split]
        ablation[split]=dict(episodes=len(selected),alarmed_episodes=sum(bool(x.any()) for x in selected),
            alarm_frames=sum(int(x.sum()) for x in selected),frames=sum(len(x) for x in selected))
    result=dict(onset_labels_used=False,interpretation='First-alarm comparison on historical regression trajectories, not true detection delay. Segment lengths are retrospective evaluation only, not future inputs to the monitor.',
        versions={k:{field:metrics[k][field] for field in ['threshold','calibration','validation','ood','external']} for k in paths},
        v9_shorter_filter_only=dict(preprocessing=fast_config,threshold=fast_threshold,**ablation),records=rows)
    out=Path('reports/early_repetition_v10_20260921');out.mkdir(parents=True,exist_ok=True)
    save_json(result,out/'latency_analysis.json');save_json(result,paths['v10']/'latency_analysis.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    key='pant_fail_ep000000';states=np.load(f'runs/pant_fail_v7_20260921/{key}_input.npz')['states'];seconds=np.arange(len(states))/30
    fig,axes=plt.subplots(5,1,figsize=(14,12),sharex=True,constrained_layout=True)
    axes[0].plot(seconds,states[:,1],label='Left joint 2');axes[0].plot(seconds,states[:,7],label='Right joint 2');axes[0].set_ylabel('Joint angle (rad)')
    axes[1].plot(seconds,states[:,12],label='Left gripper');axes[1].plot(seconds,states[:,13],label='Right gripper');axes[1].set_ylabel('Gripper width')
    colors=dict(v9='#7a8496',v10='#b95112')
    for version in paths:
        p=pred[version][key];color=colors[version];th=bundles[version]['threshold']
        axes[2].plot(seconds,100*p['signal_percentiles'][:,1],color=color,label=f'{version.upper()} repetition percentile')
        axes[3].plot(seconds,p['frame_risk_score']/th,color=color,alpha=.4,ls=':',label=f'{version.upper()} raw / own boundary')
        axes[3].plot(seconds,p['risk_score']/th,color=color,label=f'{version.upper()} confirmed / own boundary')
        axes[4].plot(seconds,100*p['risk_percentile'],color=color,label=f'{version.upper()} calibration percentile')
        start=first(p['alarm'].numpy())
        if start is not None:
            for ax in axes:ax.axvline(start/30,color=color,ls='--',lw=1)
            axes[4].annotate(f'{version.upper()}: {start/30:.2f}s',(start/30,95),xytext=(start/30+.12,75 if version=='v10' else 45),arrowprops=dict(arrowstyle='->',color=color),color=color)
    axes[2].set_ylabel('Training percentile');axes[2].set_ylim(85,100.5)
    axes[3].set_ylabel('Score / boundary');axes[3].axhline(1,color='red',ls='--',lw=1)
    axes[4].set_ylabel('Calibration percentile');axes[4].axhline(95,color='red',ls='--',lw=1);axes[4].set_ylim(-2,102)
    axes[-1].set_xlabel('Seconds (30 FPS)');axes[-1].set_xlim(6,20)
    for ax in axes:ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=8,ncol=2)
    fig.suptitle('pant_fail EP0: observed joints, dense repetition and causal alarm timing\nNo manually supplied fault-onset labels')
    fig.savefig(out/'pant_fail_ep0_latency.png',dpi=150);fig.savefig(out/'pant_fail_ep0_latency.pdf');plt.close(fig)
    print(json.dumps(dict(short_filter_threshold=fast_threshold,short_filter_validation=ablation['validation'],
        first_alarms=[{k:r[k] for k in ['id','first_alarm_advance_seconds','v9_shorter_filter_first_alarm_frame']} for r in rows]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
