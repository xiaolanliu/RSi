"""Joint plots, raw-data/action audit, and quantitative EP0 coverage comparison."""
import ast,dataclasses,hashlib,json
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyarrow.compute as pc
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from .fit_ood_v4 import save_json


def spans(alarm):
    d=np.diff(np.r_[0,np.asarray(alarm,dtype='int8'),0]);return list(zip(np.flatnonzero(d==1).tolist(),np.flatnonzero(d==-1).tolist()))


def main():
    root=Path('runs/action_dynamics_v9_20260921');out=Path('reports/action_dynamics_v9_20260921');out.mkdir(parents=True,exist_ok=True)
    source=Path('/data/dataset/laudry_fold_dataset/test_lwy/pant_fail/data/chunk-000/file-000.parquet')
    table=pq.read_table(source);table=table.filter(pc.equal(table['episode_index'],0))
    joints=np.array(table['state.joints'].to_pylist(),dtype='float32');grip=np.array(table['state.gripper_w'].to_pylist(),dtype='float32')
    s=np.load('runs/pant_fail_v7_20260921/pant_fail_ep000000_input.npz')['states'];np.testing.assert_array_equal(s,np.c_[joints,grip])
    actions=np.c_[table['action.joints'].to_pylist(),table['action.gripper_w'].to_pylist()]
    old=torch.load('runs/three_signal_v8_20260921/predictions.pt',weights_only=True)['pant_fail_ep000000']
    new=torch.load(root/'predictions.pt',weights_only=True)['pant_fail_ep000000'];events=json.loads((root/'review_events.json').read_text())['pant_fail_ep000000']
    t=np.arange(len(s))/30;left=np.zeros(len(t));right=np.zeros(len(t))
    for arm,trace in enumerate((left,right)):
        ev=[e for e in events if e['arm']==arm]
        for i,e in enumerate(ev):trace[e['end']:(ev[i+1]['end'] if i+1<len(ev) else len(t))]=e['score']
    fig,axs=plt.subplots(10,1,figsize=(16,21),sharex=True)
    for j in range(6):
        axs[j].plot(t,s[:,j],lw=.9,label=f'L{j+1}');axs[j].plot(t,s[:,j+6],lw=.9,label=f'R{j+1}');axs[j].legend(loc='upper right',ncol=2);axs[j].set_ylabel('rad')
    axs[6].plot(t,s[:,12],label='Left gripper');axs[6].plot(t,s[:,13],label='Right gripper');axs[6].set_ylabel('native width');axs[6].legend(loc='upper right')
    axs[7].plot(t,left,label='Left matched units');axs[7].plot(t,right,label='Right matched units');axs[7].legend();axs[7].set_ylabel('soft count')
    axs[8].plot(t,old['signal_percentiles'][:,1]*100,label='V8 repetition');axs[8].plot(t,new['signal_percentiles'][:,1]*100,label='V9 unit repetition');axs[8].legend();axs[8].set_ylabel('train percentile')
    axs[9].plot(t,old['risk_percentile']*100,label='V8 risk');axs[9].plot(t,new['risk_percentile']*100,label='V9 risk');axs[9].axhline(95,color='red',ls='--');axs[9].legend();axs[9].set_ylabel('cal. percentile');axs[9].set_xlabel('seconds')
    for ax in axs:
        ax.grid(alpha=.2)
        for start,end in spans(new['alarm']):ax.axvspan(start/30,end/30,color='green',alpha=.075,lw=0)
        for start,end in spans(old['alarm']):ax.axvspan(start/30,end/30,color='orange',alpha=.2,lw=0)
    fig.suptitle('pant_fail EP0 | 6 joints per arm (radians) + observed grippers\nGreen: V9 alarm; orange: V8 alarm | cached layout = L6, R6, gL, gR',fontsize=14)
    fig.tight_layout();fig.savefig(out/'pant_fail_ep0_joints.png',dpi=140);fig.savefig(out/'pant_fail_ep0_joints.pdf');plt.close(fig)
    fig,axs=plt.subplots(4,1,figsize=(15,9),sharex=True)
    for a,j in zip(axs[:2],[1,2]):
        a.plot(t,s[:,j],label=f'L{j+1}');a.plot(t,s[:,j+6],label=f'R{j+1}');a.set_ylabel('rad');a.legend()
    axs[2].plot(t,s[:,12],label='left grip');axs[2].plot(t,s[:,13],label='right grip');axs[2].legend()
    axs[3].plot(t,old['risk_percentile']*100,label='V8');axs[3].plot(t,new['risk_percentile']*100,label='V9');axs[3].axhline(95,ls='--',color='red');axs[3].legend();axs[3].set_xlabel('seconds')
    for a in axs:a.set_xlim(8,18);a.grid(alpha=.2)
    fig.suptitle('Zoom: short repeated joint/gripper units and delayed detection');fig.tight_layout();fig.savefig(out/'pant_fail_ep0_zoom.png',dpi=140);plt.close(fig)
    # Inspect actual local pi05 transforms without importing or changing policy.
    transforms=Path('/data/users/liuweiyuan/Code/pizero_anke/src/openpi/transforms.py');tree=ast.parse(transforms.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('DeltaActions','AbsoluteActions')]
    env=dict(dataclasses=dataclasses,np=np,Sequence=Sequence,DataDict=dict,DataTransformFn=object)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(transforms),'exec'),env)
    state=np.arange(14,dtype='float32')/20;target=np.tile(state,(3,1))+np.arange(3,dtype='float32')[:,None]/10
    mask=np.array([True]*6+[False]+[True]*6+[False]);data=dict(state=state.copy(),actions=target.copy())
    encoded=env['DeltaActions'](mask)(data)['actions'].copy()
    decoded=env['AbsoluteActions'](mask)(data)['actions']
    np.testing.assert_allclose(decoded,target,atol=1e-7);np.testing.assert_allclose(encoded[:,[6,13]],target[:,[6,13]])
    b=torch.load('models/v8/fold_all.pt',weights_only=True);scale=np.array(b['normalization']['input_normalization']['scale'])[:12]
    sm=uniform_filter1d(s[:,:12]/scale,size=5,axis=0,origin=2);vel=np.diff(sm,axis=0);corr=[];lags=np.arange(15,301)
    for lag in lags:
        x,y=vel[lag:],vel[:-lag];corr.append(float(2*(x*y).sum()/(np.square(x).sum()+np.square(y).sum()+1e-9)))
    peaks,_=find_peaks(corr,distance=12);peaks=sorted(peaks,key=lambda i:corr[i],reverse=True)[:10]
    result=dict(frames=len(s),fps=30,raw_action_equals_state=bool(np.array_equal(actions,s)),
        ee_position_all_zero=bool((np.array(table['state.ee_pos'].to_pylist())==0).all()),
        cached_layout='L6,R6,gL,gR',policy_layout='L6,gL,R6,gR',joint_unit='radian, confirmed by user',
        gripper_unit='native recorded width; not assumed to be radians',
        old_alarm_frames=int(old['alarm'].sum()),new_alarm_frames=int(new['alarm'].sum()),
        old_alarm_spans=spans(old['alarm']),new_alarm_spans=spans(new['alarm']),
        old_repeat_above_training_p99_frames=int((old['signal_percentiles'][:,1]>.99).sum()),
        units=len(events),per_arm_units=[sum(e['arm']==arm for e in events) for arm in (0,1)],
        unit_duration_quantiles=np.quantile([e['frames'] for e in events],[.1,.5,.9]).tolist(),
        motion_autocorrelation_peaks=[dict(lag_frames=int(lags[i]),correlation=corr[i]) for i in peaks],
        pi05=dict(delta_absolute_roundtrip_passed=True,mask=mask.tolist(),
            semantics='Joint predictions inside the model are relative to chunk-anchor state; Policy.infer returns post-transform absolute targets, not per-step increments.',
            transforms_sha256=hashlib.sha256(transforms.read_bytes()).hexdigest()),
        interpretation='EP0 is a development diagnosis, not an independent new test. No actual-command dynamics model was fitted from duplicate-action logs.')
    save_json(result,root/'diagnosis.json');save_json(result,out/'diagnosis.json');print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
