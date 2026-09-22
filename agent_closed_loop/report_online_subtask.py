"""Audited V5 streaming report: identical 18 review trajectories, OOD held out."""
from .paths import artifact_path
import argparse,csv,json,os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .fit_ood_v4 import load,save_json,sha,section


def build(a):
    folder=a.root/'ood';m=json.loads((folder/'manifest.json').read_text())
    ev=json.loads((folder/'evaluation.json').read_text());phase=load(artifact_path(m['phase_checkpoint']))
    if not ev['complete'] or phase['protocol']['smoke'] or phase['epoch']!=60:raise ValueError('Incomplete official run')
    folds=ev['folds'][:3]
    if [f['fold'] for f in ev['folds']]!=['0','1','2','all']:raise ValueError('Missing folds')
    if any(f['steps']!=1200 or f['normal_training_episodes_seen']!=2246 for f in ev['folds']):raise ValueError('Incomplete risk training')
    audit={}
    for name in ('cpu','cuda_0'):
        check=json.loads((folder/f'inference_validation_{name}.json').read_text())
        if not check['passed'] or len(check['records'])!=6 or check['phase_sha256']!=sha(artifact_path(m['phase_checkpoint'])):raise ValueError('Missing final-weight validation')
        by_file={r['file']:r for r in m['records']}
        for r in check['records']:
            if r['frames']!=by_file[r['record']]['length'] or r['checkpoint_sha256']!=sha(folder/f"fold_{r['fold']}/checkpoint.pt"):raise ValueError('Truncated/stale validation')
        audit[name]=check
    old_path=Path('reports/ood_v4_2000_20260920/data/report.json')
    old=json.loads(old_path.read_text());a.output.mkdir(parents=True,exist_ok=True);(a.output/'data').mkdir(exist_ok=True)
    pmap=np.load(folder/'ordered_phase.npy',mmap_mode='r');records={r['review_id']:r for r in m['records'] if r.get('review_id')}
    predictions={f:load(folder/f'fold_{f}/predictions.pt') for f in range(3)};episodes=[]
    for ep in old['episodes']:
        r=records[ep['id']];fold=r['episode_index'] if r['split']=='ood' else 0;pred=predictions[fold][r['file']]
        if pred['split']!=('ood_oof_test' if ep['ood'] else 'id_test') or r['length']!=ep['length']:raise ValueError('Display leakage/length')
        p=section(pmap,r);result={k:pred[k].tolist() for k in ('frame_risk_score','risk_score','alarm','confirmed_phase','accepted_phase')}
        result.update(phase_probs=p[:,:5].tolist(),boundary_cdf=p[:,5:].tolist(),threshold=folds[fold]['threshold'],fold=fold,split=pred['split'])
        baseline=ep['results']['128'];hard=p[:,:5].argmax(-1);confirmed=np.asarray(result['confirmed_phase'])
        assert np.allclose(p[:,:5].sum(-1),1) and (p[:,:5]>=-1e-7).all()
        assert ((np.diff(confirmed)>=0)&(np.diff(confirmed)<=1)).all()
        result['statistics']=dict(soft_argmax_changes=int((np.diff(hard)!=0).sum()),soft_argmax_backtracks=int((np.diff(hard)<0).sum()),
            confirmed_changes=int((np.diff(confirmed)>0).sum()),confirmed_backtracks=int((np.diff(confirmed)<0).sum()),
            baseline_argmax_changes=int((np.diff(np.asarray(baseline['phase_probs']).argmax(-1))!=0).sum()),
            baseline_argmax_backtracks=int((np.diff(np.asarray(baseline['phase_probs']).argmax(-1))<0).sum()),
            alarm_frames=sum(result['alarm']),final_confirmed=int(confirmed[-1]))
        result['csv']=f"data/{ep['id']}.csv"
        fields=['frame','seconds','confirmed_phase','accepted_phase',*[f'P{k}' for k in range(1,6)],*[f'C{k}' for k in range(1,5)],'risk_score','frame_risk_score','threshold','alarm','fold','split']
        with (a.output/result['csv']).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for t in range(ep['length']):
                row={k:result[k][t] for k in ('confirmed_phase','accepted_phase','risk_score','frame_risk_score','alarm')}
                row.update(frame=t,seconds=t/30,threshold=result['threshold'],fold=fold,split=result['split'],
                    **{f'P{k+1}':float(p[t,k]) for k in range(5)},**{f'C{k+1}':float(p[t,5+k]) for k in range(4)})
                w.writerow(row)
        media=dict(ep['media'])
        def relative(v):return os.path.relpath((old_path.parent.parent/v).resolve(),a.output.resolve())
        for k in ('video','webm','poster'):
            if media.get(k):media[k]=relative(media[k])
        media['sprites']=[relative(v) for v in media['sprites']]
        for v in media['sprites']:
            if not (a.output/v).exists():raise FileNotFoundError(v)
        episodes.append(dict(**{k:ep[k] for k in ('id','task','task_label','episode_index','length','fps','ood','tail_start')},media=media,result=result,
            baseline=dict(phase_probs=baseline['phase_probs'],risk_score=baseline['risk_score'],threshold=baseline['threshold'],alarm=baseline['alarm'])))
    assert len(episodes)==18 and sum(e['ood'] for e in episodes)==3
    summary=dict(detected=sum(f['positive_tests'][0]['detected'] for f in folds),
        normal_episode_alarm_fraction=float(np.mean([f['normal_test']['episode_alarm_fraction'] for f in folds])),
        macro_episode_auroc=float(np.mean([f['episode_auroc'] for f in folds])))
    training=json.loads((a.root/'phase/status.json').read_text());latency=json.loads((folder/'latency.json').read_text())
    if latency['checkpoint_sha256']!=sha(folder/'fold_all/checkpoint.pt'):raise ValueError('Stale latency')
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),summary=summary,folds=folds,quality=m['quality'],training=training,
        phase_config=m['config'],protocol=phase['protocol'],deployment=ev['folds'][-1],latency=latency,audit=audit,
        baseline=dict(summary=old['models']['128']['summary'],quality=old['models']['128']['quality']),episodes=episodes,
        limitations=[
            '软阶段为有序边界记忆的占据分布；argmax仍可能回摆。confirmed_phase只允许保持或前进一步，不能据此认定边界准确。',
            '训练目标来自原离线CompILE teacher，可含完整轨迹信息；在线模型输入只有当前和历史，既不输入teacher，也不输入剩余时长或完成比例。',
            '共享128维主干沿用V4第2000轮且冻结；本次新训练有序head 60轮、四个风险head各1200步，没有重训2000轮主干。',
            '阶段以P1启动；每条新任务必须reset。P5不是任务成功判定，不支持任意半程状态自动重定位。',
            'OOD仅暂停已确认阶段的推进，并令accepted_phase=0；观测缓存及候选软阶段继续更新，报警解除后重新累计确认。',
            'LINE和4次dropout读出仍使用冻结的瞬时阶段分类方向，只作为内部证据；不会将有序概率很尖锐直接解释成低OOD。',
            '风险是弱标签学得的复查候选分数，不是经过校准的不可恢复概率。没有逐帧异常真值，也没有自动调用GPT。',
            '正常训练2246、校准275、测试275条；每折复用同一批275条正常测试。阈值来自正常校准轨迹风险最大值的95%分位。',
            '三条test_lwy分别整条留出，展示对应留出折；全部三条训练的fold_all只作为部署候选，不用于这些异常的测试指标。',
            'test_lwy已参与历史版本开发，整条留一不是全新域外任务测试。episode 0的990–1484帧只是“尾部”提示的粗范围，不是真实起错帧。',
            '动作历史来自已观测state的差分，并非策略原始action chunk。现有视觉缓存短袖32×32，其余256×256，仍有任务相关分辨率混杂。',
            '连续30 FPS、因果视觉编码是接口前提。计算耗时不含VAE、解码、动作生成、机器人通信或GPT；不是报警等待时间。'])
    save_json(report,a.output/'data/report.json')
    payload=json.dumps(report,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    (a.output/'index.html').write_text((a.output/'template.html').read_text().replace('__REPORT_JSON__',payload))
    save_json(dict(passed=True,episodes=18,phase_sha256=sha(artifact_path(m['phase_checkpoint'])),report_sha256=sha(a.output/'data/report.json'),
        heldout_folds=True,full_cpu_gpu_validation=True,probability_mass=True,confirmed_no_skip_no_regression=True),a.output/'data_validation.json')
    print(json.dumps(dict(summary=summary,quality=m['quality']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs/online_subtask_v5_20260920'))
    p.add_argument('--output',type=Path,default=Path('reports/online_subtask_v5_20260920'));build(p.parse_args())
