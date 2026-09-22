"""Render recorded simulation evidence; never changes scores or success labels."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def report(run):
    run = Path(run)
    events = [json.loads(line) for line in (run/"events.jsonl").read_text().splitlines() if line]
    if not events:
        raise ValueError("No executed observations to plot")
    time = np.asarray([e["time"] for e in events])
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout="constrained")
    axes[0].plot(time, [e["risk"]["risk_percentile"] for e in events], label="Frozen RSI risk percentile")
    axes[0].plot(time, [int(e["alarm"]) for e in events], alpha=.5, label="Alarm bit used by controller")
    axes[0].set_ylim(-.03, 1.03)
    axes[0].legend(loc="upper left")
    for ax, key in zip(axes[1:], ("projection_distance", "action_unit_repetition", "line_energy")):
        ax.plot(time, [e["risk"]["ood_signals"][key] for e in events])
        ax.set_ylabel(key.replace("_", " "))
    for ax in axes:
        for e in events:
            if e["source"] != "vla":
                ax.axvspan(e["time"], e["time"]+.04, color="orange", alpha=.2)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Native simulation time (seconds)")
    fig.savefig(run/"ood_evidence.png", dpi=140)
    plt.close(fig)
    states, physical = [], []
    for event in events:
        with np.load(run/"observations"/f'{event["step"]:06d}.npz') as obs:
            states.append(obs["state"])
            physical.append(obs["measured_gripper_openings"] if "measured_gripper_openings" in obs else [np.nan]*2)
    states, physical = np.asarray(states), np.asarray(physical)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for arm, ax, offset in zip(("left", "right"), axes[:2], (0, 7)):
        for joint in range(6):
            ax.plot(time, states[:, offset+joint], label=f"j{joint+1}")
        ax.set_ylabel(f"{arm} joints (rad)")
        ax.legend(ncol=6, loc="upper left")
    for i, name in enumerate(("left", "right")):
        axes[2].plot(time, states[:, i*7+6], "--", label=f"{name} native previous target")
        axes[2].plot(time, physical[:, i], label=f"{name} physical opening")
    axes[2].set_ylabel("Normalized gripper opening")
    axes[2].legend(ncol=2)
    for ax in axes:
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Native simulation time (seconds)")
    fig.savefig(run/"robot_state.png", dpi=140)
    plt.close(fig)
    records = {}
    for name in ("loop_summary.json", "native_outcome.json", "vla/vla_metadata.json", "control_test.json"):
        if (run/name).exists():
            records[name] = json.loads((run/name).read_text())
    configuration = run/"run_config.json"
    if configuration.exists():
        provider = json.loads(configuration.read_text()).get("resolved_provider", {})
        records["provider"] = {key: provider[key] for key in ("model", "base_url", "max_requests") if key in provider}
    api = sorted((run/"recovery").glob("api_attempt_*.json"))
    if api:
        records["api_attempts"] = [json.loads(path.read_text()) for path in api]
        records["recovery_decisions"] = [json.loads(path.read_text())
            for path in sorted((run/"recovery").glob("recovery_*.json"))]
    warning = "此运行注入了测试报警，只验证控制交接。" if "control_test.json" in records else "本页展示模型的实际报警与原生任务结果。"
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>{html.escape(run.name)} — RSI 仿真记录</title>
<style>body{{max-width:1200px;margin:32px auto;padding:0 16px;font:16px/1.6 sans-serif;background:#f5f7fa;color:#182638}}video,img{{width:100%;background:white}}pre{{white-space:pre-wrap;background:white;padding:20px}}.note{{padding:16px;background:#fff1c4}}</style>
<h1>{html.escape(run.name)}</h1><p class="note">{warning} 当前 RSI 来自实机折衣数据，未完成仿真任务校准。无报警不等于任务正常；任务成功只读取原生判定。</p>
<video controls preload="metadata" src="sensors.mp4"></video>
<p>视角顺序：顶视、左腕、右腕。图表橙色区间表示由恢复策略执行。</p>
<img src="ood_evidence.png" alt="三项 OOD 证据与报警"><img src="robot_state.png" alt="关节与夹爪状态">
<h2>原始结果</h2><pre>{html.escape(json.dumps(records,ensure_ascii=False,indent=2))}</pre>
<p><a href="events.jsonl">已确认执行记录</a> · <a href="commands_requested.jsonl">请求记录</a> · <a href="native_metadata.json">仿真与本体信息</a></p></html>'''
    (run/"index.html").write_text(page)
    print(run/"index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    report(parser.parse_args().run)
