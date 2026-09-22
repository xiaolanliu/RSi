"""Audit uploaded action chunks and actual commands against measured episodes."""
from pathlib import Path
import csv,json,html
import numpy as np
from scipy.spatial import cKDTree
from .command_logs import read_command_log,CausalCommandEvidence,compare_future_plans,JOINTS,KEYS
from .action_units import packed_state
from .fit_ood_v4 import load,save_json,sha
from .paths import artifact_path
from .audit_alarm_latency import alarm_segments


def normal_motion_envelope(bundle):
    manifest=json.loads(Path('runs/ood_v4_2000_20260920/dim128/ood/manifest.json').read_text())
    frames=np.load(artifact_path(manifest['data'])/'frames.npy',mmap_mode='r')
    scale=np.asarray(bundle['normalization']['input_normalization']['scale'])[:12]
    velocities=[];accelerations=[]
    for r in manifest['records']:
        if r['split']!='train':continue
        q=np.array(frames[r['source_offset']:r['source_offset']+r['length'],:12])*scale
        ix=np.linspace(0,len(q)-3,min(128,len(q)-2),dtype=int)
        velocities.append(np.abs(np.diff(q,axis=0)[ix+1])*30)
        accelerations.append(np.abs(np.diff(q,n=2,axis=0)[ix])*900)
    return dict(fit_split='normal train only',samples=sum(len(x) for x in velocities),quantile=.999,
        speed_rad_s=np.quantile(np.concatenate(velocities),.999,axis=0).tolist(),
        acceleration_rad_s2=np.quantile(np.concatenate(accelerations),.999,axis=0).tolist(),
        interpretation='Observed normal motion envelope, not manufacturer limits or a calibrated command OOD threshold.')


def first_time(t,mask):
    ix=np.flatnonzero(mask)
    return None if not len(ix) else float(t[ix[0]])


def main():
    import torch
    torch.set_num_threads(2)
    out=Path('reports/command_constraints_v11_20260921/command_audit');out.mkdir(parents=True,exist_ok=True)
    run=Path('runs/command_constraints_v11_20260921');run.mkdir(parents=True,exist_ok=True)
    b=load('models/v11/fold_all.pt');envelope=normal_motion_envelope(b)
    predictions={v:load(f'runs/{folder}/predictions.pt') for v,folder in [('v10','early_repetition_v10_20260921'),('v11','command_constraints_v11_20260921')]}
    offline=[np.load(f'runs/pant_fail_v7_20260921/pant_fail_ep{i:06d}_input.npz')['states'] for i in range(4)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for directory in sorted(Path('dataset_inference').glob('check_*')):
        log=read_command_log(directory);ticks=log['ticks'];chunks=log['chunks'];summary=log['summary'];time=log['seconds']
        q=packed_state(log['measured'],'left7_right7')
        tree=cKDTree(q[:,:12]);matches=[tree.query(s[:,:12]) for s in offline]
        medians=[float(np.median(d)) for d,ix in matches];ep=int(np.argmin(medians))
        if medians[ep]>1e-4 or sorted(medians)[1]<.01:raise ValueError('Episode identity not established')
        distances,indices=matches[ep];ts=np.arange(len(offline[ep]))/30
        offsets=time[indices]-ts;good=distances<1e-5;offset=float(np.median(offsets[good]))
        stable=good&(np.abs(offsets-offset)<.1)
        align=dict(episode=f'pant_fail_ep{ep:06d}',candidate_median_l2_rad=medians,median_l2_rad=medians[ep],
            matching_fraction=float(good.mean()),clock_offset_seconds=offset,
            clock_residual_abs_p95_seconds=float(np.quantile(np.abs(offsets[stable]-offset),.95)),
            note='Offline clock registration by measured state; repeated/held states excluded from clock fit. No inference features use future measured states.')
        time=time-offset;key=align['episode'];provider=np.array([t['source']=='provider' for t in ticks]);policy=np.array([t['source']!='transition' for t in ticks])
        limit=np.deg2rad(summary['joint_limit_deg_per_tick']);history=CausalCommandEvidence(limit)
        stream=[history.step(float(t),a,u,q,tick['source']) for t,a,u,q,tick in zip(log['seconds'],log['requested'],log['sent'],log['measured'],ticks)]
        clip=np.array([x['clip_ratio'] for x in stream]);duty=np.array([x['clip_duty'] for x in stream]);tracking=np.array([np.nan if x['lagged_tracking_rms_rad'] is None else x['lagged_tracking_rms_rad'] for x in stream])
        requested=log['requested'];sent=log['sent'];measured=log['measured'];lookup={c['request_id']:c for c in chunks}
        raw=np.full_like(requested,np.nan);provider_count={};index_errors=[]
        for i,t in enumerate(ticks):
            if t['source']!='provider':continue
            c=lookup[t['request_id']];index=t['chunk_step_index']
            if index!=t['sent_policy_action_count']-1-c['observed_sent_action_count']:raise ValueError('Policy step indexing differs')
            raw[i]=c['absolute_targets'][index];provider_count[t['request_id']]=provider_count.get(t['request_id'],0)+1
            index_errors.append(float(np.max(np.abs(raw[i,JOINTS]-requested[i,JOINTS]))))
        recomputed_clip=np.abs(requested-sent)
        maxclip=np.rad2deg(recomputed_clip[:,JOINTS].max(1));clipped=maxclip>.05
        gripper_clip=np.max(recomputed_clip[:,[6,13]],1)*1000>.5
        if int(((clipped|gripper_clip)&policy).sum())!=summary['clipped_tick_count']:raise ValueError('Summary clipping count differs')
        chunk_rows=[];previous=None
        for c in chunks:
            overlap=compare_future_plans(c,previous);skip=c['apply_sent_action_count']-c['observed_sent_action_count']
            a=c['absolute_targets'][max(0,skip):,JOINTS]
            speed=float((np.abs(np.diff(a,axis=0))*30/np.maximum(envelope['speed_rad_s'],1e-8)).max()) if len(a)>1 else None
            accel=float((np.abs(np.diff(a,n=2,axis=0))*900/np.maximum(envelope['acceleration_rad_s2'],1e-8)).max()) if len(a)>2 else None
            chunk_rows.append(dict(request_id=c['request_id'],received_seconds=(c['received_at_monotonic_ns']-int(log['monotonic_ns'][0]))*1e-9-offset,
                stale_policy_steps=skip,used_provider_steps=provider_count.get(c['request_id'],0),overlap_steps=overlap['overlap_steps'],
                future_plan_disagreement_rms_deg=None if overlap['rms_rad'] is None else float(np.rad2deg(overlap['rms_rad'])),
                future_speed_envelope_ratio=speed,future_acceleration_envelope_ratio=accel))
            previous=c
        firsts={}
        for version in predictions:
            alarm=predictions[version][key]['alarm'].numpy();segments=alarm_segments(alarm)
            firsts[version]=dict(first_alarm_seconds=None if not segments else segments[0][0]/30,
                first_segment_frames=0 if not segments else segments[0][1]-segments[0][0],
                first_half_second_segment=next((a/30 for a,z in segments if z-a>=15),None))
        metrics=dict(directory=directory.name,alignment=align,chunks=len(chunks),provider_ticks=int(provider.sum()),interpolation_ticks=int((policy&~provider).sum()),
            gripper_adapter='All rows verified: raw opening <20 mm becomes 0; opening capped at70 mm before requested/sent processing.',
            transition_ticks=int((~policy).sum()),used_chunk_row_fraction=float(provider.sum()/sum(len(c['absolute_targets']) for c in chunks)),
            unused_request_ids=[c['request_id'] for c in chunks if c['request_id'] not in provider_count],
            joint_limit_deg_per_tick=summary['joint_limit_deg_per_tick'],clipped_tick_fraction=float(((clipped|gripper_clip)&policy).sum()/policy.sum()),
            max_joint_clipping_deg=float(maxclip[policy].max()),provider_raw_modified_fraction=float(np.mean(np.array(index_errors)>1e-6)),
            stale_policy_steps_p50_p95=np.quantile([r['stale_policy_steps'] for r in chunk_rows],[.5,.95]).tolist(),
            median_tick_interval_seconds=float(np.median(np.diff(log['seconds']))),
            first_removed_more_than_step_limit_seconds=first_time(time,(clip>1)&policy&(time>=0)),
            lagged_tracking_rms_deg_p50_p95=np.rad2deg(np.nanquantile(tracking[policy],[.5,.95])).tolist(),
            alarm_comparison=firsts,files_sha256={p.name:sha(p) for p in directory.glob('*.json*')})
        rows.append(metrics)
        np.savez_compressed(out/f'{key}_commands.npz',timestamps_monotonic_ns=log['monotonic_ns'],seconds=time,
            requested_rad_m=requested,sent_rad_m=sent,measured_rad_m=measured,source=np.array([t['source'] for t in ticks]),
            clip_ratio=clip,clip_duty=duty,lagged_tracking_rms_rad=tracking)
        with (out/f'{key}_chunks.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(chunk_rows[0]));writer.writeheader();writer.writerows(chunk_rows)
        with (out/f'{key}_ticks.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['seconds','source','request_id','chunk_step_index','clip_ratio','clip_duty','lagged_tracking_rms_rad'])
            for i,tick in enumerate(ticks):writer.writerow([time[i],tick['source'],tick['request_id'],tick['chunk_step_index'],clip[i],duty[i],None if np.isnan(tracking[i]) else tracking[i]])
        j=KEYS.index(summary['worst_joint']);fig,axes=plt.subplots(4,1,figsize=(14,10),sharex=True,constrained_layout=True)
        for values,label,color in [(raw,'Raw model row','#b1a5cb'),(requested,'Requested target','#d99435'),(sent,'Actually sent','#3391c7'),(measured,'Measured joint','#222222')]:
            valid=np.isfinite(values[:,j]);axes[0].plot(time[valid],np.rad2deg(values[valid,j]),label=label,color=color,lw=1)
        axes[0].set_ylabel(f'{KEYS[j]} (deg)')
        axes[1].plot(time,clip,label='Removed target / configured step limit',color='#b85524');axes[1].plot(time,duty,label='Clipped tick fraction in past 0.5 s',color='#6866a1');axes[1].set_ylabel('Constraint diagnostics')
        axes[2].plot(time,np.rad2deg(tracking),label='Measured vs command sent >=100 ms earlier',color='#397158');axes[2].set_ylabel('RMS joint error (deg)')
        for version,color in [('v10','#929aa8'),('v11','#ba5415')]:
            p=predictions[version][key];axes[3].plot(ts,100*p['risk_percentile'],label=f'{version.upper()} OOD percentile',color=color)
        axes[3].axhline(95,color='red',ls='--',lw=1);axes[3].set_ylim(-2,102);axes[3].set_ylabel('Calibrated OOD');axes[3].set_xlabel('Episode seconds; command/video clock registration is approximate')
        for ax in axes:ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=8,ncol=2)
        axes[-1].set_xlim(0,len(ts)/30);fig.suptitle(f'{key} / {directory.name}\nCommand diagnostics are not additional alarm criteria; 100 ms is not an identified servo delay')
        fig.savefig(out/f'{key}.png',dpi=130);fig.savefig(out/f'{key}.pdf');axes[-1].set_xlim(0,min(20,len(ts)/30));fig.savefig(out/f'{key}_zoom.png',dpi=130);plt.close(fig)
        print(json.dumps(metrics,ensure_ascii=False),flush=True)
    result=dict(normal_motion_envelope=envelope,records=rows,command_alarm_enabled=False,
        limitations=['Four logs are failure/regression runs; no independent normal real-command calibration set.',
            'Chunk arrival timestamps exist; observation capture timestamps and exact measured-state read times are absent.',
            'Configured per-tick clipping is not a verified mechanical velocity/acceleration limit.',
            'Prediction chunk rows are proposals; requested, sent, and measured are distinct quantities.',
            'A real controller can track commands well while the task fails. Visual effect is still needed.'])
    save_json(result,out/'audit.json');save_json(result,run/'command_audit.json')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V11 真实命令审计</title><style>body{max-width:1200px;margin:auto;padding:24px;font:16px/1.7 system-ui;background:#101925;color:#e5eef8}a{color:#72dfcd}img{width:100%}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap}td,th{padding:9px;border-bottom:1px solid #526e88}section{margin:28px 0}</style><h1>真实 action：预测、请求、下发与反馈</h1><p><a href="../index.html">返回 V11 三项判据</a> · <a href="../../../docs/V11_COMMAND_CONSTRAINTS_CN.md">计算与接入方案</a> · <a href="audit.json">审计数据</a></p><p>所有 chunk 和执行 tick 已读取并校验单位、请求编号及 policy 步下标。下图中的命令诊断尚未接入硬报警：新日志均对应历史失败轨迹，缺少正常真实命令校准。最终 OOD 仍为三项，V11将重复尾部强度权重提高到1.5。</p><p>原始 chunk 50 步存在重叠、丢弃和下发前目标处理；应以 sent 表示真实命令，以 measured 表示机械臂状态。约60 Hz的执行 tick 不可直接当作30 Hz的模型观测。</p><div class="scroll"><table><tr><th>日志</th><th>对应轨迹</th><th>chunk</th><th>provider / 插值 tick</th><th>限幅比例</th><th>最大关节裁剪</th></tr>'
    for r in rows:page+=f'<tr><td>{r["directory"]}</td><td>{r["alignment"]["episode"]}</td><td>{r["chunks"]}</td><td>{r["provider_ticks"]} / {r["interpolation_ticks"]}</td><td>{100*r["clipped_tick_fraction"]:.2f}%</td><td>{r["max_joint_clipping_deg"]:.2f}°</td></tr>'
    page+='</table></div><p>全部预测行还验证了夹爪处理：小于20 mm的开口置零，超过70 mm封顶。原始预测夹爪值的变化不一定形成不同的实际动作。</p><p>100 ms历史命令比较仅用于可复现诊断；必须用正常实际命令辨识响应时延后再决定残差阈值。正常示教速度/加速度的99.9%分位只用于chunk运动包络审计，不等于硬件极限或经过校准的命令报警。</p>'
    for r in rows:
        key=r['alignment']['episode'];page+=f'<section><h2>{html.escape(key)} · {r["directory"]}</h2><p><a href="{key}.png">全程图</a> · <a href="{key}.pdf">PDF</a> · <a href="{key}_commands.npz">统一弧度/米数据</a> · <a href="{key}_chunks.csv">chunk逐项诊断</a> · <a href="{key}_ticks.csv">执行tick诊断</a></p><img src="{key}_zoom.png" alt="预测目标、实际请求、下发、反馈与报警同步对照"></section>'
    (out/'index.html').write_text(page+'</html>')


if __name__=='__main__':main()
