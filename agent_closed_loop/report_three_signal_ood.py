"""New report, preserving historical reports and recomputing stage pauses."""
import csv,json,os
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from .fit_ood_v4 import load,save_json,sha,section
from .monitor import OnlineMonitor
from .evaluate_duration_subtask import predict
from .three_signal_ood import SIGNAL_NAMES


@torch.inference_mode()
def main(version='v8'):
    torch.set_num_threads(2)
    run=dict(v8='three_signal_v8_20260921',v9='action_dynamics_v9_20260921',v10='early_repetition_v10_20260921',v11='command_constraints_v11_20260921')[version]
    root=Path('runs')/run;out=Path('reports')/run
    (out/'data').mkdir(parents=True,exist_ok=True)
    old_dir=Path('reports')/dict(v8='ordered_duration_v7_20260921',v9='three_signal_v8_20260921',v10='action_dynamics_v9_20260921',v11='early_repetition_v10_20260921')[version]
    old=json.loads((old_dir/'data/report.json').read_text());evaluation=json.loads((root/'evaluation.json').read_text())
    bundle=Path(f'models/{version}/fold_all.pt');assert sha(bundle)==evaluation['bundle_sha256']
    monitor=OnlineMonitor(bundle);saved=load(root/'predictions.pt')
    source=Path('runs/ood_v4_2000_20260920/dim128/ood');manifest=json.loads((source/'manifest.json').read_text())
    lookup={r['review_id']:r for r in manifest['records'] if r.get('review_id')}
    latent=np.load(source/'latent.npy',mmap_mode='r');episodes=[]
    for e in old['episodes']:
        if e['id'] in lookup:
            r=lookup[e['id']];z=torch.from_numpy(np.array(section(latent,r)));p=saved[r['file']]
        else:
            z=torch.from_numpy(np.load(Path('runs/pant_fail_v7_20260921')/f'{e["id"]}_predictions.npz')['latent']);p=saved[e['id']]
        zn=((z-monitor.zmean)/monitor.zscale).clamp(-15,15)
        stage=predict(monitor.stage,zn,p['alarm'].numpy())
        r={k:v.tolist() for k,v in stage.items() if k in ('confirmed_phase','phase_probs')}
        r.update({k:v.tolist() for k,v in p.items()})
        r.update(threshold=monitor.threshold,csv=f'data/{e["id"]}.csv',
            accepted_phase=np.where(p['alarm'],0,stage['confirmed_phase']).tolist(),
            risk_threshold_ratio=(p['risk_score']/monitor.threshold).tolist(),
            components=monitor.head.components(p['evidence']).tolist())
        alarm=np.asarray(r['alarm']);where=np.flatnonzero(alarm);peak=int(p['risk_score'].argmax())
        tail=e.get('tail_start');tail_alarm=bool(alarm[tail:].any()) if tail is not None else None
        summary=dict(alarm=bool(alarm.any()),alarm_frames=int(alarm.sum()),first_alarm_frame=int(where[0]) if len(where) else None,
            peak_frame=peak,peak_threshold_ratio=float(p['risk_score'].max()/monitor.threshold),tail_alarm=tail_alarm)
        media=deepcopy(e['media']);media['sprites']=[os.path.relpath((old_dir/s).resolve(),out.resolve()) for s in media['sprites']]
        for s in media['sprites']:assert (out/s).exists()
        episode={k:e[k] for k in ('id','task','task_label','episode_index','length','fps','ood','tail_start')}
        episode.update(media=media,result=r,summary=summary,
            baseline=dict(confirmed_phase=e['result']['confirmed_phase'],risk_threshold_ratio=e['result']['risk_threshold_ratio'],
                          alarm=e['result']['alarm']),teacher_phase=e.get('teacher_phase'))
        if 'risk_percentile' in e['result']:
            episode['baseline']['risk_percentile']=e['result']['risk_percentile']
        episodes.append(episode)
        fields=['frame','seconds','confirmed_phase','accepted_phase','risk_percentile','alarm',*monitor.signal_names]
        with (out/r['csv']).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for t in range(e['length']):
                writer.writerow(dict(frame=t,seconds=t/30,**{k:r[k][t] for k in fields[2:6]},**dict(zip(monitor.signal_names,r['evidence'][t]))))
        print(json.dumps(dict(episode=e['id'],**summary)),flush=True)
    legacy=json.loads((source/'fold_all/metrics.json').read_text())['normal_test']
    if version!='v8':legacy=dict(alarmed_episodes=old['evaluation']['validation']['alarmed_episodes'],episode_alarm_fraction=old['evaluation']['validation']['episode_alarm_rate'])
    report=dict(version=version.upper(),created_at=datetime.now(timezone.utc).isoformat(),evaluation=evaluation,
        signal_names=monitor.signal_names,episodes=episodes,old_normal=legacy,
        old_ood_count=3 if version=='v8' else old['evaluation']['ood']['alarmed_episodes'],
        old_external_count=0 if version=='v8' else old['evaluation']['external']['alarmed_episodes'],
        limitations=[
            '当前观测指 state、已执行状态差分与因果 visual48 的共享投影特征；不是只比较原始关节数值。',
            '参考库与尺度仅拟合正常训练集，阈值只由正常校准集确定；新OOD判据未使用失败轨迹拟合。',
            '最终只使用投影距离、重复且少进展、LINe能量三项；没有读出分歧或隐特征直连风险网络。',
            '阶段Head仍为V7；此页根据新报警重新计算阶段暂停，因此阶段轨迹可以与旧页不同。',
            f'正常测试报警{evaluation["validation"]["alarmed_episodes"]}/275，旧fold_all为11/275；简化不等于全面提高检测精度。',
            '原三条异常的旧版使用各自留一折，新版仅正常数据拟合，协议不同。EP0需单独查看指定尾段是否报警；前段报警不能算作正确定位尾部异常。',
            f'pant_fail四条有{evaluation["external"]["alarmed_episodes"]}条报警；没有逐帧故障真值，报警次数不等于正确检测所有失败。',
            '三项百分位来自正常训练参考；最终风险百分位来自正常校准轨迹峰值。均不是故障概率。',
            '最终尾部秩p≤5%才报警，使用有限样本修正和保守的相等值处理。固定的是尾部比例，原始数值边界由数据决定。',
            '参考分布冻结；重新训练或校准后可更新，不用当前未确认正常的在线帧滚动更新。静止且特征仍在正常支持内的失败可能漏检。',
            '只输出监测与阶段暂停，没有自动机器人恢复。旧版计时未移植到新版页面。'])
    if version in ('v9','v10','v11'):
        report['limitations']=[
            '总判据仍为三项：正常投影距离、动作重复、LINe。重复项覆盖独立左右臂的夹爪开合单元和夹爪不动时的周期运动。',
            '动作单元按关节路径弧长对齐以容忍变速；在相同夹爪阶段比较视觉端点，累计多次相似且效果接近的单元。仅为视觉效果代理，不等同于物体成功或失败真值。',
            '缓存排列为L6,R6,gL,gR；部署pi05排列为L6,gL,R6,gR，需要显式layout转换。关节角为弧度，夹爪按记录原生宽度处理。',
            '本地pi05内部预测相对chunk起始state的关节增量；Policy.infer经过反归一化与AbsoluteActions后返回绝对目标，不应再次累加。',
            '离线action逐帧等于state，末端位姿全零；本版检测已执行运动重复，尚未以真实命令训练action-conditioned动力学。不得把未来state当作独立命令制造预测准确率。',
            '分位数只拟合正常训练集，阈值只用正常校准集。但pant_fail已用于诊断设计，本轮回放不是新的独立泛化测试。',
            f'正常测试报警{evaluation["validation"]["alarmed_episodes"]}/275，V8为16/275；新增重复覆盖同时存在误报代价。',
            '原三条异常仍有漏报，EP0指定尾部仍未报警；不能将前段报警称为正确定位尾部故障。',
            '真实action动力学接口已经准备，并拒绝action==state伪命令训练；尚需真实正常执行日志做拟合、延迟对齐和独立校准，当前不会把未拟合动力学值混入报警。',
            '相同控制命令可以被机械臂准确执行但任务仍失败：动力学跟踪正常不等于任务有进展，需要同时判断重复动作的视觉效果。',
            '报警暂停V7阶段记忆与年龄；不控制机器人、不自动恢复。旧版本和旧评测保留。']
    if version in ('v10','v11'):
        report['limitations'][1]='动作单元仍按弧长对齐；视觉端点相似是任务进展的代理。历史单元贡献按3秒时间常数衰减，单元完成后也逐帧衰减，衡量近期密集重复。'
        report['limitations'][6]=f'正常测试报警{evaluation["validation"]["alarmed_episodes"]}/275，V9为18/275；同时报告误报与检测时间，不能把更早报警视为每次都正确。'
        report['limitations'][7]='原三条异常与pant_fail均为历史回归集，没有人工故障起点。首次报警提前不等同于检测延迟降低的真值评估；指定尾部单独报告。'
        report['limitations'].extend([
            '三项训练尾部强度取最大值后做10帧均值、5帧持续确认；正常校准集重新确定数值阈值，尾部比例仍为5%。',
            '3秒记忆、10帧均值及5帧确认是在新版失败回放前按响应时间设计固定的参数，并非学习出的最优物理时间。没有新故障起点标签。',
            '当前仍需完整动作单元才更新重复证据；首次动作或外观相近但物体状态不同的失败仍可能漏检。'])
    if version=='v11':
        report['limitations'][4]='原示教缓存action等于state；本次新增真实chunk、requested、sent和measured日志已审计，单位与时间线不同，不能直接拼接。'
        report['limitations'][6]=f'正常轨迹报警{evaluation["validation"]["alarmed_episodes"]}/275，V10为14/275；正常报警帧112→140，轨迹误报与帧误报并非同一指标。'
        report['limitations'][8]='新增四条命令日志均对应pant_fail历史失败轨迹，尚无正常实际命令校准。命令约束图是诊断，尚未影响最终报警。'
        report['limitations'][7]='原三条异常均出现报警，EP0指定尾部也有报警；部分触发仅几帧且首次发生在前段，没有故障起点真值，不能按3/3宣称正确定位全部失败。'
        report['limitations'].append('V11重复权重为1.5，作用于正常训练尾部强度之后；三项卡片保持原经验百分位。最终阈值使用正常校准集重新确定，尾部比例仍5%。')
    save_json(report,out/'data/report.json')
    template=Path('tools/three_signal_report_template.html').read_text()
    if version in ('v9','v10','v11'):
        template=template.replace('V8','V9').replace('三项 OOD 判据','动作单元重复与三项判据').replace('重复且少进展','动作单元重复')
        template=template.replace('../../docs/V9_THREE_SIGNAL_CN.md','../../docs/V9_ACTION_DYNAMICS_CN.md')
        template=template.replace('重复动作和缺少进展合成一项；','左右臂分别提取完整动作单元，按关节路径对齐并累积相似单元，比较相近姿态下的视觉变化；')
        template=template.replace('旧三条异常使用留一折，新版没有用异常拟合；正常误报对照使用旧fold_all。','两版均只用正常数据拟合OOD；pant_fail已用于本轮诊断设计，本页为回归评测，不是新独立测试。')
        template=template.replace('旧版','V8').replace('../ordered_duration_v7_20260921/index.html','../three_signal_v8_20260921/index.html').replace('V7 历史报告与计时','V8 历史报告')
        card='<section class="card"><h2>先看关节轨迹：pant_fail EP0 为什么漏检？</h2><p>原来仅80/2018帧报警，重复单元没有被持续累计。主要往复周期约24帧，旧周期网格未包含24；双臂不同步、执行变速也降低固定滞后相似性。新方法识别70个完整单元（左37、右33），同时检查相似本体状态下视觉效果是否改变。</p><p><a href="pant_fail_ep0_joints.png" target="_blank">完整关节图 PNG</a> · <a href="pant_fail_ep0_joints.pdf" target="_blank">PDF</a> · <a href="diagnosis.json">逐项诊断数据</a></p><a href="pant_fail_ep0_joints.png" target="_blank"><img src="pant_fail_ep0_zoom.png" alt="EP0关节弧度、夹爪与新旧报警的同步对比" style="width:100%;height:auto"></a><p class="muted">当前动力学条件是已执行本体运动；真实命令响应模型需要补充正常的state、实际下发action与时间戳。页面没有把未训练的动力学残差当成已验证判据。</p></section>'
        if version in ('v10','v11'):
            template=template.replace('V9','V10').replace('V8','V9')
            template=template.replace('../../docs/V10_ACTION_DYNAMICS_CN.md','../../docs/V10_EARLY_REPETITION_CN.md')
            template=template.replace('../three_signal_v8_20260921/index.html','../action_dynamics_v9_20260921/index.html')
            template=template.replace('再做30帧均值和15帧持续过滤','再做10帧均值和5帧持续确认')
            template=template.replace('按关节路径对齐并累积相似单元','按关节路径对齐，按3秒时间常数衰减历史贡献并累积近期相似单元')
            card='<section class="card"><h2>报警时机：V9 与 V10</h2><p>相近姿态下的动作和视觉端点再次出现，且短时间内重复越来越密集，才抬高重复证据。没有人工故障时间标签；首次报警提前只是历史回归结果。正常训练参考与正常校准阈值均重新计算。</p><div class="scroll"><table><thead><tr><th>轨迹</th><th>V9 首次报警</th><th>V10 首次报警</th><th>提前</th><th>V10 首段长度</th><th>V10 ≥0.5秒报警段起点</th></tr></thead><tbody>'
            previous={e['id']:e for e in old['episodes']}
            for e in episodes:
                if not e['id'].startswith('pant_fail'):continue
                before=previous[e['id']]['summary']['first_alarm_frame'];after=e['summary']['first_alarm_frame']
                fmt=lambda t:'未报警' if t is None else f'{t/30:.2f} 秒'
                gain='—' if before is None or after is None else f'{(before-after)/30:.2f} 秒'
                from .audit_alarm_latency import alarm_segments
                segments=alarm_segments(e['result']['alarm']);first_length=0 if not segments else segments[0][1]-segments[0][0]
                stable=next((a for a,b in segments if b-a>=15),None)
                card+=f'<tr><td>{e["id"]}</td><td>{fmt(before)}</td><td>{fmt(after)}</td><td>{gain}</td><td>{first_length} 帧</td><td>{fmt(stable)}</td></tr>'
            card+='</tbody></table></div><p class="muted">EP2首次报警只有1帧，EP1首次报警也较短；不能把短促触发称为持续介入。≥0.5秒报警段起点是事后评估，不是在线模型偷看未来，也没有增加报警判据。</p><p><a href="latency_analysis.json">逐项计算与延迟分解</a> · <a href="pant_fail_ep0_latency.pdf">导出 PDF</a></p><a href="pant_fail_ep0_latency.png"><img src="pant_fail_ep0_latency.png" alt="EP0关节、重复分位与V9/V10报警时机对比" style="width:100%;height:auto"></a><p class="muted">最终风险图中青色为V10，灰色为V9；分别对各版本的正常校准分布排名，因此百分位可读，原始分数不直接横比。</p></section>'
            template=template.replace("[[r.risk_percentile.map(v=>100*v),'#72dfcd']]","[[e.baseline.risk_percentile.map(v=>100*v),'#8d9aaa'],[r.risk_percentile.map(v=>100*v),'#72dfcd']]")
        if version=='v11':
            template=template.replace('V10','V11').replace('V9','V10')
            template=template.replace('../../docs/V11_EARLY_REPETITION_CN.md','../../docs/V11_COMMAND_CONSTRAINTS_CN.md')
            template=template.replace('../action_dynamics_v9_20260921/index.html','../early_repetition_v10_20260921/index.html')
            template=template.replace('取最大值，再做','重复项尾部强度乘以1.5后取最大值，再做')
            template=template.replace('只用三项证据作判断','三项证据：重复权重提高至1.5')
            card=card.replace('V10','V11').replace('V9','V10')
            card=card.replace('latency_analysis.json','command_audit/audit.json').replace('逐项计算与延迟分解','真实命令审计数据')
            card=card.replace('pant_fail_ep0_latency.pdf','command_audit/pant_fail_ep000000.pdf').replace('pant_fail_ep0_latency.png','command_audit/pant_fail_ep000000_zoom.png')
            card=card.replace('EP2首次报警只有1帧，EP1首次报警也较短；不能把短促触发称为持续介入。','表中同时给出首次触发与持续段，短促触发不等同于持续介入。')
            card=card.replace('正常训练参考与正常校准阈值均重新计算。','重复尾部强度权重提高到1.5，正常校准阈值重新计算。')
            card='<section class="card"><h2>新提供的真实命令日志</h2><p>四条日志与pant_fail EP0–EP3的实测关节对应。预测chunk为弧度/米，执行tick为度/毫米；约60Hz执行循环包含provider与插值命令。原始chunk、requested、sent、measured必须分开。</p><p><a href="command_audit/index.html">打开四条日志的命令、限幅、反馈与报警对照</a></p><p class="muted">这些命令约束尚未进入硬报警；当前缺少正常实际命令的独立校准数据。下方报警改善来自重复权重调整。</p></section>'+card
        template=template.replace('<section class="card"><h2>逐帧查看</h2>',card+'<section class="card"><h2>逐帧查看</h2>')
    (out/'index.html').write_text(template.replace('__REPORT_JSON__',json.dumps(report,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')))
    save_json(dict(passed=True,episodes=len(episodes),bundle_sha256=sha(bundle),report_sha256=sha(out/'data/report.json'),html_sha256=sha(out/'index.html')),out/'data_validation.json')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['v8','v9','v10','v11'],default='v8');main(parser.parse_args().version)
