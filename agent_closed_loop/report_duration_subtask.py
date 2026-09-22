"""V7 report with unchanged raw risk, threshold-relative risk, and teacher/V6."""
import csv,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from .fit_ood_v4 import load,save_json,sha
from .paths import artifact_path


def main():
    root=Path('runs/ordered_duration_v7_20260921');out=Path('reports/ordered_duration_v7_20260921');(out/'data').mkdir(parents=True,exist_ok=True)
    old=json.loads(Path('reports/latent_change_v6_20260920/data/report.json').read_text())
    selection=json.loads((root/'calibration/calibration_sweep.json').read_text())
    evaluation=json.loads((root/'evaluation/validation.json').read_text());ablation=json.loads((root/'ablation_evaluation/validation.json').read_text())
    saved=load(root/'evaluation/review_predictions.pt');source=Path(evaluation['protocol']['source'])
    m=json.loads((source/'manifest.json').read_text());records={r['review_id']:r for r in m['records'] if r.get('review_id')}
    frames=np.load(artifact_path(m['data'])/'frames.npy',mmap_mode='r');episodes=[]
    for e in old['episodes']:
        p=saved[e['id']];r={k:v.tolist() for k,v in p.items() if hasattr(v,'tolist')};fold=p['fold']
        r.update(fold=fold,split=p['split'],threshold=old['risk']['folds'][fold]['threshold'],csv=f'data/{e["id"]}.csv')
        raw=np.array(r['frame_risk_score']);c=np.r_[0.,raw.cumsum()];t=np.arange(len(raw));start=np.maximum(0,t-29)
        r['smoothed_risk_score']=((c[t+1]-c[start])/(t+1-start)).tolist()
        r['risk_threshold_ratio']=(np.array(r['risk_score'])/r['threshold']).tolist()
        np.testing.assert_array_equal(r['alarm'],e['result']['alarm']);np.testing.assert_array_equal(r['risk_score'],e['result']['risk_score'])
        rec=records[e['id']];teacher=frames[rec['source_offset']:rec['source_offset']+rec['length'],76:]
        fields=['frame','seconds','confirmed_phase','accepted_phase',*[f'P{k}' for k in range(1,6)],
                'frame_risk_score','smoothed_risk_score','risk_score','threshold','risk_threshold_ratio','alarm',
                *[f'boundary_cdf_{k}' for k in range(1,5)],*[f'duration_log_bias_{k}' for k in range(1,5)]]
        with (out/r['csv']).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for t in range(e['length']):
                row={k:r[k][t] for k in ('confirmed_phase','accepted_phase','frame_risk_score','smoothed_risk_score','risk_score','risk_threshold_ratio','alarm')}
                row.update(frame=t,seconds=t/30,threshold=r['threshold'],**{f'P{k+1}':r['phase_probs'][t][k] for k in range(5)},
                    **{f'boundary_cdf_{k+1}':r['boundary_cdf'][t][k] for k in range(4)},**{f'duration_log_bias_{k+1}':r['duration_log_bias'][t][k] for k in range(4)})
                w.writerow(row)
        durations=lambda hard:np.diff(np.r_[0,np.flatnonzero(np.diff(hard)>0)+1]).tolist()
        episodes.append({**{k:e[k] for k in ('id','task','task_label','episode_index','length','fps','ood','tail_start','media')},
            'result':r,'baseline':e['result'],'teacher_phase':(teacher.argmax(-1)+1).tolist(),
            'teacher_probs':teacher.tolist(),'durations':dict(v6=durations(e['result']['confirmed_phase']),v7=durations(r['confirmed_phase']))})
    extension_path=Path('runs/pant_fail_v7_20260921/report_extension.json')
    extension=json.loads(extension_path.read_text()) if extension_path.exists() else None
    if extension:
        assert extension['evaluation']['complete']
        assert sha(extension['evaluation']['bundle'])==extension['evaluation']['bundle_sha256']
        episodes.extend(extension['episodes'])
        assert len({e['id'] for e in episodes})==len(episodes)
    for e in episodes:
        for s in e['media']['sprites']:assert (out/s).exists()
    report=dict(version='V7',created_at=datetime.now(timezone.utc).isoformat(),evaluation=evaluation,
        baseline_v6=old['evaluation']['quality']['validation'],ablation=ablation,calibration=selection,
        training=json.loads((root/'centered/status.json').read_text()),risk=old['risk'],episodes=episodes,
        external_evaluation=extension['evaluation'] if extension else None,
        limitations=[
          '阶段指标使用原离线teacher，不是人工语义真值；离线teacher可利用完整轨迹，在线模型仍存在泛化差距。',
          '模型用2246条正常轨迹完整训练60轮。确认参数只用275条正常校准轨迹选定，再评估275条正常测试轨迹。',
          '已确认阶段与累计跨界概率单向推进，不能回溯；概率分布的argmax不作为执行阶段。均值阶段也不回退。',
          '200帧是软先验中心；长度偏离时产生梯度，强观测可覆盖先验。没有读取episode总长，也不因到时而强制推进。',
          'OOD报警期间冻结阶段记忆与阶段年龄，持续观察视觉；解除报警需重新建立持续证据。',
          '风险机制和权重保持V4，页面展示原始数值及阈值倍数，未把风险重新标成概率。风险超过本折阈值才报警。',
          '原始三条test_lwy分别使用其整条留出风险折。新增pant_fail四条使用既有fold_all，未用于拟合或调整阈值；不能与原三折成绩混算。',
          'pant_fail是用户指定的失败采集数据，没有逐帧OOD/不可恢复真值；报警表示模型复查候选，未报警也不等于确认正常。新增轨迹没有离线teacher或V6对照，不补造这些曲线。',
          '阶段汇总关闭OOD暂停以隔离阶段质量；逐帧报告开启OOD暂停。末阶段未完成时不补造长度监督。',
          '该版本是受CompILE启发的因果蒸馏和有序更新，不是原论文离线VAE的等价实现。',
          '目前只有监测与暂停输出，没有自动机器人恢复或GPT调用；P5不代表成功。'])
    save_json(report,out/'data/report.json')
    template=(out/'template.html').read_text();(out/'index.html').write_text(template.replace('__REPORT_JSON__',json.dumps(report,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')))
    save_json(dict(passed=True,episodes=len(episodes),original_episodes=18,original_risk_unchanged=True,
        external_episodes=len(extension['episodes']) if extension else 0,
        report_sha256=sha(out/'data/report.json'),html_sha256=sha(out/'index.html')),out/'data_validation.json')

if __name__=='__main__':main()
