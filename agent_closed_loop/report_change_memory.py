"""Build the V6 review from held-out predictions and audited portable bundles."""
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .fit_ood_v4 import load,sha,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('runs/latent_change_v6_20260920'))
    p.add_argument('--output',type=Path,default=Path('reports/latent_change_v6_20260920'));a=p.parse_args()
    ev=json.loads((a.run/'memory_evaluation/evaluation.json').read_text());abl=json.loads((a.run/'memory_evaluation_no_duration/evaluation.json').read_text())
    if not ev['complete'] or not abl['complete']:raise ValueError('Incomplete phase experiments')
    old=json.loads(Path('reports/online_subtask_v5_20260920/data/report.json').read_text());saved=load(a.run/'memory_evaluation/review_predictions.pt')
    risk=json.loads(Path('runs/ood_v4_2000_20260920/dim128/ood/evaluation.json').read_text())
    audits={k:json.loads(Path(f'models/v6/validation_{k}.json').read_text()) for k in ('cpu','cuda_0')}
    for x in audits.values():
        assert x['passed'] and len(x['records'])==6
        for r in x['records']:assert r['bundle_sha256']==sha(Path(f'models/v6/fold_{r["fold"]}.pt'))
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'data').mkdir(exist_ok=True);episodes=[]
    for e in old['episodes']:
        pred=saved[e['id']];result={k:v.tolist() for k,v in pred.items() if hasattr(v,'tolist')}
        result.update(fold=pred['fold'],split=pred['split'],threshold=risk['folds'][pred['fold']]['threshold'])
        assert result['split']==('ood_oof_test' if e['ood'] else 'id_test')
        q=np.asarray(result['phase_probs']);hard=np.asarray(result['confirmed_phase']);assert ((np.diff(hard)>=0)&(np.diff(hard)<=1)).all()
        np.testing.assert_allclose(q.sum(-1),1,atol=1e-6)
        baseline={k:e['result'][k] for k in ('phase_probs','confirmed_phase','accepted_phase','risk_score','alarm','threshold')}
        result['csv']=f"data/{e['id']}.csv"
        fields=['frame','seconds','confirmed_phase','accepted_phase',*[f'P{k}' for k in range(1,6)],*[f'crossing_{k}' for k in range(1,5)],
            'risk_score','alarm','boundary_evidence','sustained_boundary_evidence','latent_change','anchor_distance','stage_age','transition_armed','fold','split']
        with (a.output/result['csv']).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for t in range(e['length']):
                row={k:result[k][t] for k in ('confirmed_phase','accepted_phase','risk_score','alarm','boundary_evidence','sustained_boundary_evidence','latent_change','anchor_distance','stage_age','transition_armed')}
                row.update(frame=t,seconds=t/30,fold=pred['fold'],split=pred['split'],**{f'P{k+1}':float(q[t,k]) for k in range(5)},
                    **{f'crossing_{k+1}':result['boundary_event_probs'][t][k] for k in range(4)});w.writerow(row)
        durations=lambda x:np.diff(np.r_[0,np.flatnonzero(np.diff(x)>0)+1]).tolist()
        episodes.append(dict(**{k:e[k] for k in ('id','task','task_label','episode_index','length','fps','ood','tail_start','media')},
            result=result,baseline=baseline,statistics=dict(v5_completed_durations=durations(baseline['confirmed_phase']),
            v6_completed_durations=durations(hard),v5_final=baseline['confirmed_phase'][-1],v6_final=int(hard[-1]),alarm_frames=sum(result['alarm']))))
    assert len(episodes)==18
    for e in episodes:
        for v in e['media']['sprites']:assert (a.output/v).exists()
    training=json.loads((a.run/'memory_phase/status.json').read_text());protocol=json.loads((a.run/'memory_phase/protocol.json').read_text())
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),evaluation=ev,ablation=abl,training=training,protocol=protocol,
        risk=risk,episodes=episodes,audits=audits,baseline_v5_ood=old['summary'],
        limitations=[
            '所有阶段一致率、早切和漏边界指标都以离线 teacher 为参照，不是人工语义真值。阶段诊断关闭 OOD 暂停；逐帧展示开启 OOD 暂停。',
            '选择阈值只使用正常校准集：相对 V5 至少降低10%的提前超过60帧边界，漏边界比例不超过3%，再比较确认阶段一致率。测试集没有用于选择阈值。',
            '200帧为低权重训练参考，100–500帧内无长度惩罚；在线没有固定200帧计时切换。未观察到的边界和末尾未完成阶段不补造结束点。',
            '软概率只分配给当前已确认阶段和相邻下一阶段，候选置信度可回落；执行阶段只能相邻前进，不能回退。',
            '下一阶段必须先观察到低跨界证据，再出现持续高证据，并有相对阶段参考的 latent 变化；未来边界的高分不会被提前记账。',
            '视觉和动作信息融合到128维因果latent。现有动作列与state重复，动作信息实际为已执行state后向差分，尚无独立策略action chunk。',
            'OOD采用已经训练的 V4 单风险 head，保持 LINe、读出不稳定性与循环证据；与执行阶段记忆解耦，防止报警暂停改变自身训练输入。旧 V5 的完整模型另行保留。',
            '三条test_lwy使用各自整条留出风险折；fold_all是全部异常训练的候选，不具有独立异常测试成绩。风险不是不可恢复概率。',
            '主干冻结在原128维模型；本次新head训练60轮，另有相同步数的零长度权重对照。对照结果接近，没有证据证明低权重先验本身带来独立收益。',
            '每条任务从P1开始并reset；P5不代表任务成功。不接收未来帧、轨迹总长或完成比例。视觉编码上游须因果，输入保持连续30FPS。'])
    save_json(report,a.output/'data/report.json');template=(a.output/'template.html').read_text()
    (a.output/'index.html').write_text(template.replace('__REPORT_JSON__',json.dumps(report,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')))
    save_json(dict(passed=True,episodes=18,report_sha256=sha(a.output/'data/report.json'),phase_sha256=ev['phase_checkpoint_sha256'],
        full_bundle_validation=True,held_out_ood=True,confirmed_no_skip_no_regression=True),a.output/'data_validation.json')
    print(json.dumps(dict(v5=ev['baseline_v5'],v6=ev['quality']['validation']),indent=2))

if __name__=='__main__':main()
