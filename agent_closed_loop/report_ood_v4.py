"""Build a self-contained interactive matched-width, held-out OOD report."""
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch

from .fit_ood_v4 import load,section,save_json
from .prepare_ood_v4 import sha
from .diagnose_ood_v4_visual import diagnose,diagnose_readout_seeds


def summary(folds):
    folds=[f for f in folds if f['fold']!='all']
    return dict(detected=sum(f['positive_tests'][0]['detected'] for f in folds),ood_episodes=len(folds),
        normal_episode_alarm_fraction=float(np.mean([f['normal_test']['episode_alarm_fraction'] for f in folds])),
        macro_episode_auroc=float(np.mean([f['episode_auroc'] for f in folds])),folds=folds)


def experiment_markdown(report):
    models=report['models'];baseline=report['baseline_v3']
    lines=['# V4：128 / 256 维共享主干完整 2000 epoch 对比','',
        '本报告使用固定第2000轮权重，不根据测试结果挑选checkpoint。每个epoch完整监督2,579,110帧，共5,158,220,000帧次、352,000次主干参数更新。',
        '正常训练2246条、校准275条、测试275条；三条test_lwy整条留一，每折仅两条参与风险头训练。三折复用同一批正常测试轨迹，不能将其视为825条独立轨迹。','',
        '## 结果与解释','',
        '| 模型 | OOD检出 | 正常轨迹报警率（三折均值） | 宏平均episode AUROC | 正常测试teacher一致性 |',
        '|---|---:|---:|---:|---:|']
    for d,m in models.items():
        s=m['summary'];q=m['quality']['validation']
        lines.append(f"| V4 {d} | {s['detected']}/3 | {s['normal_episode_alarm_fraction']:.2%} | {s['macro_episode_auroc']:.6f} | {q['teacher_agreement']:.2%} |")
    lines.append(f"| V3 历史基线 | {baseline['detected_ood_episodes']}/3 | {baseline['mean_normal_episode_alarm_fraction']:.2%} | {baseline['macro_episode_auroc']:.6f} | {baseline['phase_quality']['normal_splits']['validation']['teacher_agreement']:.2%} |")
    lines+=['','整体误报率会掩盖任务差异，下面仍是同一275条正常测试轨迹：','',
        '| 任务 | 轨迹数 | 128维报警率 | 256维报警率 |','|---|---:|---:|---:|']
    for task,label in [('fold_pants','裤子'),('fold_short_sleeve','短袖'),('fold_long_sleeve','长袖')]:
        rows={d:[f['normal_by_task'][task] for f in m['summary']['folds']] for d,m in models.items()}
        values={d:np.mean([r['episode_alarm_fraction'] for r in rr]) for d,rr in rows.items()}
        lines.append(f"| {label} | {rows['128'][0]['episodes']} | {values['128']:.2%} | {values['256']:.2%} |")
    lines+=['','3条已知异常的检出与AUROC不能单独支持“新版本全面优于V3”或“已能判断不可恢复”。V3属于历史基线，结构和训练方式不同；严格匹配宽度实验是本轮V4的128与256维。',
        '阶段指标是与同一冻结teacher的一致性，不是人工语义阶段准确率。尤其应看训练到留出轨迹的差距：','',
        '| D | 训练一致性 | 校准一致性 | 测试一致性 | 训练KL | 测试KL |','|---|---:|---:|---:|---:|---:|']
    for d,m in models.items():
        q=m['quality'];lines.append(f"| {d} | {q['train']['teacher_agreement']:.2%} | {q['calibration']['teacher_agreement']:.2%} | {q['validation']['teacher_agreement']:.2%} | {q['train']['teacher_kl']:.4f} | {q['validation']['teacher_kl']:.4f} |")
    lines+=['','这些差距需要解释和修正，不宜仅凭训练拟合提高继续扩到512维。新旧模型的上下文长度、绝对位置输入和蒸馏结构也有差异；目前没有消融证明其中某一项就是差距的原因。','',
        '## Episode 0 尾部复查','',
        '| D | 首次报警帧 | 990–1484帧中的报警帧数 | 更早片段中的报警帧数 |','|---|---:|---:|---:|']
    ep=next(e for e in report['episodes'] if e['id']=='test_lwy_ep000000')
    for d,r in ep['results'].items():
        alarm=r['alarm'];first=next((i for i,v in enumerate(alarm) if v),None)
        lines.append(f"| {d} | {first} | {sum(alarm[990:])} / 495 | {sum(alarm[:990])} / 990 |")
    lines+=['','990只是将用户“尾部”提示转成的粗复查范围，不是真实失败起点。上表不构成逐帧准确率或真实检测延迟，前段报警也不能自动标为误报。','',
        '## 单一输出下，各证据是否必要','',
        '| D | 配置 | OOD检出 | 正常轨迹报警率 | AUROC |','|---|---|---:|---:|---:|']
    for d,m in models.items():
        for name,s in dict(full=m['summary'],**m['ablations']).items():
            lines.append(f"| {d} | {name} | {s['detected']}/3 | {s['normal_episode_alarm_fraction']:.2%} | {s['macro_episode_auroc']:.6f} |")
    lines+=['','每个风险头均冻结对应最终主干，训练1200个优化步，并独立使用正常校准集设阈值；1200步不是主干epoch。no_line移除LINE输入，困难正常采样池保持相同，因此不是去除LINE所有间接影响。',
        '若删去某项仍3/3，且正常报警率仅小幅变化，就没有充分证据认定该项带来了稳定检出收益。不能根据这三条测试轨迹直接选择最佳消融并宣称泛化改善。','',
        '## 视觉与不确定性机制检查','',
        '| D | 视觉扰动 | 表征相对变化均值 | 扰动后有报警的片段 | LINE能量升高的片段 |','|---|---|---:|---:|---:|']
    for d,m in models.items():
        for variant in ('visual_at_training_mean','visual_four_sigma_shift'):
            rows=[r for r in m['visual_sensitivity']['rows'] if r['variant']==variant]
            lines.append(f"| {d} | {variant} | {np.mean([r['latent_relative_change'] for r in rows]):.4f} | {sum(r['modified_alarm_frames']>0 for r in rows)} / {len(rows)} | {sum(r['line_mean_change']>0 for r in rows)} / {len(rows)} |")
    lines+=['','六条正常留出轨迹中，各改动120帧视觉，保持state和运动不变。它只检验视觉依赖，不能当真实OOD检出率。表征变化大但能量更低，说明陌生特征仍可能激活某个阶段的分类方向；“保留视觉信息”与“正确判断视觉域外”是两个不同要求。',
        '下一步应优先比较阶段与视觉联合的重要神经元选择，并检查截断前的正常支持范围；保持一个风险头和一个最终阈值。具体候选见[修订方案](../../OOD_V4_ARCHITECTURE_PLAN_CN.md)第11节，尚未训练，不混入本轮指标。','',
        '| D | dropout种子 | OOD检出 | 原阈值下正常轨迹报警率 |','|---|---:|---:|---:|']
    for d,m in models.items():
        for r in m['readout_seed_sensitivity']['seeds']:
            lines.append(f"| {d} | {r['seed']} | {r['detected']}/3 | {r['mean_normal_episode_alarm_fraction']:.2%} |")
    lines+=['','仅改变读出采样掩码，保持权重、标准化和原阈值不变，不选择最好种子。该分歧只描述阶段读出的不稳定性，并不等于整个机器人策略的贝叶斯不确定性。','',
        '## 架构、时延与验收','',
        '阶段输出语义更正：本版phase_probs来自无单调约束的逐帧五分类蒸馏头，不是原CompILE根据顺序边界生成的segment_masks。保留五个输出字段不等于保留了原顺序分段机制；不能将其称为“原五阶段概率”。原始边界模型与权重仍保留，但未用于本版报告的阶段输出。',
        '视觉48维直接进入attention内部K/V投影，state14和已执行state差分14提供查询与残差；融合层、四层因果TCN均使用D，输出z[D]。原始state/visual不再后拼接。当前视觉残差和训练用重建目标帮助保留信息。',
        '一次共享编码后，五阶段读出、LINE、四次轻量dropout读出和历史循环证据进入单个时序风险头。对外只有原始/平滑风险、一个阈值与报警，以及五阶段概率；报警时accepted_phase=0。','',
        '| D | 主干训练分钟 | 峰值已分配显存GiB | 单帧P50 ms | 单帧P95 ms |','|---|---:|---:|---:|---:|']
    for d,m in models.items():
        t=m['training'];l=m.get('latency',{}).get('total_ms',{})
        lines.append(f"| {d} | {t['seconds']/60:.2f} | {t['peak_memory_gib']:.3f} | {l.get('p50',float('nan')):.3f} | {l.get('p95',float('nan')):.3f} |")
    lines+=['','最终权重、batch=1、真实缓存step接口，400帧预热后400帧计时。包含标准化、状态差分与风险输出，不含视觉VAE、视频解码、Pi动作生成、机器人通信和GPT。阶段分项见网页；各阶段分位数不可相加。',
        'V3历史延迟基于全前缀重算，本轮使用有限历史缓存，不能将全部加速归因于宽度或LINe删减。计算耗时也不等于报警等待：mean30/min15和三块循环需要积累历史。',
        '128/256维最终权重分别通过CPU与GPU完整轨迹在线/离线对齐：三条异常加每折最接近阈值的一条正常轨迹；包含reset、未来扰动不改变前缀、原始输入标准化及网页保存预测一致性。网页另有18条轨迹、逐帧概率、导航、CSV和移动端浏览器验证。','',
        '目前保留V3作为历史参考，V4作为已完成训练和测试的实验候选；不据此自动部署或触发GPT。512维尚未训练，显存可容纳不等于扩大宽度有检测收益。','',
        '## 数据和解释限制','']+['- '+v for v in report['limitations']]
    return '\n'.join(lines)+'\n'


def build(args):
    root=args.root;out=args.output;out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir(exist_ok=True)
    data_manifest=json.loads((root/'data/manifest.json').read_text())
    if sha(root/'data/frames.npy')!=data_manifest['data_sha256']:
        raise ValueError('Packed inputs changed since training preparation')
    oldroot=Path('reports/pants_subtasks_1500_2000_20260918').resolve()
    old=json.loads((oldroot/'data/report.json').read_text());models={};cache={};audits={};post_signatures=[]
    for dim in (128,256):
        run=root/f'dim{dim}';folder=run/'ood';manifest=json.loads((folder/'manifest.json').read_text())
        evaluation=json.loads((folder/'evaluation.json').read_text());protocol=json.loads((run/'protocol.json').read_text())
        post_signatures.append(manifest['signature']['code'])
        history=[json.loads(x) for x in (run/'history.jsonl').read_text().splitlines() if x.strip()]
        if not evaluation['complete'] or evaluation['smoke'] or manifest['backbone_epoch']!=2000:raise ValueError('Incomplete official evaluation')
        if [f['fold'] for f in evaluation['folds']]!=['0','1','2','all']:raise ValueError('Missing risk folds')
        for f in evaluation['folds']:
            if f['steps']!=1200 or f['seed']!=29 or f['normal_training_episodes_seen']!=2246:
                raise ValueError('Incomplete or unmatched risk training budget')
        if [r['epoch'] for r in history]!=list(range(1,2001)):raise ValueError('Missing/duplicate epochs')
        if any(r['supervised_frames']!=2579110 for r in history):raise ValueError('Incomplete training frame coverage')
        if history[-1]['optimizer_steps']!=2000*protocol['optimizer_steps_per_epoch']:raise ValueError('Step count mismatch')
        ck=load(run/'checkpoint_epoch_2000.pt')
        if ck['epoch']!=2000 or ck['optimizer_steps']!=history[-1]['optimizer_steps']:raise ValueError('Checkpoint mismatch')
        variants={}
        for variant in ('no_line','no_uncertainty','no_loop'):
            ev=json.loads((folder/'ablations'/variant/'evaluation.json').read_text())
            if ev['smoke'] or len(ev['folds'])!=3:raise ValueError('Incomplete ablation')
            if [f['fold'] for f in ev['folds']]!=['0','1','2'] or any(f['steps']!=1200 or f['seed']!=29 or f['normal_training_episodes_seen']!=2246 for f in ev['folds']):
                raise ValueError('Incomplete ablation protocol')
            variants[variant]=summary(ev['folds'])
        models[str(dim)]=dict(dimension=dim,summary=summary(evaluation['folds']),ablations=variants,
            quality=evaluation['quality'],training=dict(epochs=2000,frames_per_epoch=2579110,
                total_supervised_frames=2000*2579110,optimizer_steps=history[-1]['optimizer_steps'],
                seconds=history[-1]['total_seconds'],peak_memory_gib=history[-1]['max_memory_GiB'],protocol=protocol,
                history=history),deployment=next(f for f in evaluation['folds'] if f['fold']=='all'))
        models[str(dim)]['visual_sensitivity']=diagnose(folder,f'cuda:{0 if dim==128 else 1}')
        models[str(dim)]['readout_seed_sensitivity']=diagnose_readout_seeds(folder,f'cuda:{0 if dim==128 else 1}')
        latency=folder/'latency.json'
        if latency.exists():models[str(dim)]['latency']=json.loads(latency.read_text())
        cache[dim]=dict(records={r['review_id']:r for r in manifest['records'] if r.get('review_id')},
            phase=np.load(folder/'phase.npy',mmap_mode='r'),evidence=np.load(folder/'evidence.npy',mmap_mode='r'),
            predictions={i:load(folder/f'fold_{i}/predictions.pt') for i in range(3)},folder=folder)
        audits[str(dim)]=dict(backbone_sha256=sha(run/'checkpoint_epoch_2000.pt'),normalization_sha256=sha(folder/'normalization.pt'),
            inference_cpu=json.loads((folder/'inference_validation_cpu.json').read_text()),
            inference_gpu=json.loads((folder/f'inference_validation_cuda_{0 if dim==128 else 1}.json').read_text()))
        for key in ('inference_cpu','inference_gpu'):
            check=audits[str(dim)][key]
            if not check['passed'] or check['smoke'] or check['quick'] or len(check['records'])!=6:
                raise ValueError('Incomplete final-weight inference validation')
            if check['backbone_sha256']!=audits[str(dim)]['backbone_sha256'] or check['normalization_sha256']!=manifest['normalization_sha256']:
                raise ValueError('Stale inference validation')
            by_file={r['file']:r for r in manifest['records']}
            if sorted(r['fold'] for r in check['records'] if r['split']=='ood')!=[0,1,2]:raise ValueError('Missing full OOD parity')
            for row in check['records']:
                if row['frames']!=by_file[row['record']]['length'] or row['risk_checkpoint_sha256']!=sha(folder/f"fold_{row['fold']}/checkpoint.pt"):
                    raise ValueError('Truncated or stale risk parity check')
    # Protocol differences allowed only for width and dimensional config.
    a=dict(models['128']['training']['protocol']);b=dict(models['256']['training']['protocol'])
    for x in (a,b):x.pop('dim')  # config comparison handled below
    ac=a.pop('config');bc=b.pop('config');ac=dict(ac);bc=dict(bc);ac.pop('dim');bc.pop('dim')
    if a!=b or ac!=bc:raise ValueError('Unmatched backbone protocols')
    if post_signatures[0]!=post_signatures[1]:raise ValueError('Different posttraining implementations between widths')
    episodes=[]
    import os
    media_prefix=os.path.relpath(oldroot,out.resolve())
    for ep in old['episodes']:
        results={};length=ep['length']
        for dim in (128,256):
            c=cache[dim];r=c['records'][ep['id']];fold=r['episode_index'] if r['split']=='ood' else 0
            p=c['predictions'][fold][r['file']]
            expected='ood_oof_test' if r['split']=='ood' else 'id_test'
            if p['split']!=expected or length!=r['length']:raise ValueError('Display split/length mismatch')
            phase=section(c['phase'],r);risk=p['risk_score'].numpy();alarm=p['alarm'].numpy()
            threshold=models[str(dim)]['summary']['folds'][fold]['threshold']
            result=dict(fold=fold,split=p['split'],threshold=threshold,frame_risk_score=p['frame_risk_score'].tolist(),
                risk_score=risk.tolist(),alarm=alarm.tolist(),phase_probs=phase.tolist(),
                accepted_phase=np.where(alarm,0,phase.argmax(-1)+1).tolist(),evidence=section(c['evidence'],r).tolist())
            results[str(dim)]=result
            fields=['frame','seconds','risk_score','frame_risk_score','threshold','alarm','accepted_phase',*[f'phase_P{k+1}' for k in range(5)],'fold','split']
            csvname=f"data/{ep['id']}_dim{dim}.csv";result['csv']=csvname
            with (out/csvname).open('w',newline='',encoding='utf-8-sig') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for t in range(length):writer.writerow(dict(frame=t,seconds=t/30,risk_score=float(risk[t]),frame_risk_score=result['frame_risk_score'][t],
                    threshold=threshold,alarm=bool(alarm[t]),accepted_phase=result['accepted_phase'][t],
                    **{f'phase_P{k+1}':float(phase[t,k]) for k in range(5)},fold=fold,split=p['split']))
        media=dict(ep['media'])
        for k in ('video','webm','poster'):
            if media.get(k):media[k]=f'{media_prefix}/{media[k]}'
        media['sprites']=[f'{media_prefix}/{v}' for v in media['sprites']]
        episodes.append(dict(id=ep['id'],task=ep['task'],task_label=ep['task_label'],episode_index=ep['episode_index'],
            length=length,fps=30,ood=ep['task']=='test_lwy',tail_start=990 if ep['id']=='test_lwy_ep000000' else None,
            media=media,results=results))
    if len(episodes)!=18 or sum(e['ood'] for e in episodes)!=3:raise ValueError('Expected 15 normal + 3 held-out OOD')
    previous=json.loads(Path('runs/action_recovery_v3/evaluation_summary.json').read_text())
    baseline_phase=json.loads((root/'baseline_phase_quality.json').read_text())
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),models=models,episodes=episodes,audit=audits,
        baseline_v3=dict(previous['summary'],phase_quality=baseline_phase),limitations=[
        '风险是弱标签学得的复查候选分数，不是校准后的不可恢复概率；未自动调用 GPT。',
        '三条 test_lwy 已参与多个版本开发；本次整条留一比较不等于全新域外任务泛化测试。',
        'Episode 0 的 990–1484 帧仅为粗复查区域。此前片段未知，不计算逐帧错误准确率或真值检测延迟。',
        '正常训练2246、校准275、测试275。三折重复使用同一批正常测试轨迹；展示正常统一 fold0，异常展示各自留出折。',
        '阶段质量指标是与冻结 teacher 的一致性，不是人工阶段真值准确率。',
        '不同维度/折的原始风险尺度不同；各自按正常校准轨迹最大风险95%分位设阈值，应比较同误报预算下的报警与排序。',
        '运动为已观测 state 差分，并非真实 Pi action chunk；重复/视觉进展属于代理指标。',
        '在线接口要求连续30 FPS；每条新轨迹必须reset，缺帧或采样率改变时需重新处理时间窗口，不能直接沿用固定帧周期。',
        '沿用现有视觉缓存：短袖32×32，裤子/长袖/test_lwy为256×256，存在任务相关分辨率差异。',
        '最终候选模型使用全部三条异常训练，单独存储，不用于本页异常测试。'])
    save_json(report,out/'data/report.json')
    template=Path('reports/ood_v4_2000_20260920/template.html').read_text()
    payload=json.dumps(report,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    (out/'index.html').write_text(template.replace('__REPORT_JSON__',payload))
    (out/'EXPERIMENT_CN.md').write_text(experiment_markdown(report))
    save_json(dict(passed=True,episodes=18,dimensions=[128,256],epoch_checks=True,matched_protocol=True,
                   source_sha256=sha(out/'data/report.json')),out/'data_validation.json')
    print(json.dumps({d:m['summary'] for d,m in models.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs/ood_v4_2000_20260920'))
    p.add_argument('--output',type=Path,default=Path('reports/ood_v4_2000_20260920'));a=p.parse_args();torch.set_num_threads(2);build(a)
