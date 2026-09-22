"""Resumable native RoboDojo collection, exclusively controlled by pi05."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time

from .cli import resources, SimClient, VLAClient
from .controller import Controller, LoopConfig
from .workers import Worker


ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(robodojo, trials=30, tasks=None):
    root = Path(robodojo)
    config = root/"task/RoboDojo/config"
    layout = root/"Assets/Eval_Layout/RoboDojo/arx_x5"
    groups = {int(group.name): list(group.glob("*.json")) for group in layout.iterdir()
              if group.is_dir() and group.name.isdigit()}
    names = sorted(p.stem for p in config.glob("*.yml") if not p.stem.startswith("_"))
    if tasks:
        if set(tasks)-set(names):
            raise ValueError("Unknown requested task")
        names = [name for name in names if name in tasks]
    result = []
    for task in names:
        if not (root/"task/RoboDojo/tasks"/f"{task}.py").is_file():
            raise ValueError(f"Missing native task implementation: {task}")
        pattern = re.compile(re.escape(task)+r"_\d+\.json")
        available = {group: sorted((p for p in paths if pattern.fullmatch(p.name)),
                                  key=lambda p: int(p.stem.rsplit("_", 1)[1]))
                     for group, paths in sorted(groups.items())}
        # Native SeedManager uses the sorted file's position, not its suffix.
        candidates = [dict(eval_seed=group, layout_id=index, layout_file=str(paths[index]))
                      for index in range(max(map(len, available.values()), default=0))
                      for group, paths in available.items() if index < len(paths)]
        if len(candidates) < trials:
            raise ValueError(f"Insufficient distinct layouts for {task}: {len(candidates)}")
        result.append(dict(task=task, target=trials, candidates=candidates,
                           task_config_sha256=digest(config/f"{task}.yml")))
    return result


class NoMonitor:
    def reset(self):
        pass

    def observe(self, obs):
        return dict(alarm=False, monitor_enabled=False)


def missing_layout_assets(robodojo, candidate):
    """Wait for actual native object resources instead of burning trial slots."""
    root = Path(robodojo)/"Assets/Object/RoboDojo"
    layout = json.loads(Path(candidate["layout_file"]).read_text())
    missing = []
    for kind in ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid"):
        for category, instances in layout.get(kind, {}).items():
            for instance in instances:
                group = "Clutter" if instance.get("type") == "cluttered" else kind
                folder = root/group/category/f'{instance["category_idx"]:05d}'
                metadata = folder/"metadata.json"
                if not metadata.is_file():
                    missing.append(str(metadata))
                elif not json.loads(metadata.read_text()).get("geometry"):
                    missing.append(str(metadata)+": missing geometry")
                if not any((folder/name).is_file() for name in ("object.usdz", "object.usd")):
                    missing.append(str(folder/"object.usd[z]"))
    return sorted(set(missing))


def assess(run, kind):
    """Only acknowledged, native-terminal VLA-only episodes enter the dataset."""
    run = Path(run)
    outcome = json.loads((run/"native_outcome.json").read_text())
    summary = json.loads((run/"loop_summary.json").read_text())
    metadata = json.loads((run/"vla/vla_metadata.json").read_text())
    if not outcome["complete"] or not outcome["valid_for_success_rate"]:
        raise ValueError("Incomplete or unstable native episode")
    if not summary["complete"] or summary["reason"] != "native_terminal":
        raise ValueError("Episode ended before native terminal")
    if summary["interventions"] or summary["recovery_mode"] != "disabled":
        raise ValueError("Recovery is forbidden during collection")
    if metadata["checkpoint_kind"] != kind:
        raise ValueError("Unexpected checkpoint identity")
    events = [json.loads(line) for line in (run/"events.jsonl").read_text().splitlines()]
    count = outcome["native_control_steps"]
    if not count or len(events) != count or summary["steps"] != count:
        raise ValueError("Native steps and execution log disagree")
    if any(e["source"] != "vla" or not e["acknowledged"] or e["step"] != i or e["next_step"] != i+1
           or e["risk"].get("monitor_enabled") is not False for i, e in enumerate(events)):
        raise ValueError("Non-VLA action, enabled monitor, or invalid execution acknowledgement")
    if len(list((run/"observations").glob("*.npz"))) != count+1:
        raise ValueError("Incomplete observation sequence")
    return dict(label="success" if outcome["native_success"] else "failure", steps=count,
                vla_calls=summary["vla_calls"], elapsed_seconds=summary["elapsed_seconds"])


def verify_rgb(run):
    """Decode every stored native RGB frame and compare its original byte hash."""
    import av
    import numpy as np
    from .recording import CAMERAS
    run = Path(run)
    expected = {name: [] for name in CAMERAS}
    for path in sorted((run/"observations").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for name in CAMERAS:
                expected[name].append(str(data[f"{name}_sha256"]))
    for name in CAMERAS:
        count = 0
        with av.open(str(run/"rgb"/f"{name}.mkv")) as container:
            for frame in container.decode(video=0):
                if count >= len(expected[name]) or hashlib.sha256(frame.to_ndarray(format="rgb24").tobytes()).hexdigest() != expected[name][count]:
                    raise ValueError(f"Lossless RGB verification failed: {name}, {count}")
                count += 1
        if count != len(expected[name]):
            raise ValueError(f"Lossless RGB frame count mismatch: {name}")
    return dict(camera_count=len(CAMERAS), frames_per_camera=len(expected[CAMERAS[0]]), all_sha256_match=True)


def start_vla(plan, output):
    p = plan["resources"]
    kind = plan["checkpoint_kind"]
    env = dict(PYTHONPATH=os.pathsep.join([str(ROOT), p["openpi"]+"/src", p["openpi"]+"/packages/openpi-client/src"]),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=str(plan["vla_gpu"]),
               XLA_PYTHON_CLIENT_PREALLOCATE="false", OPENPI_DATA_HOME=p["openpi_cache"])
    args = ["--checkpoint", p["pi05_base" if kind == "base" else "pi05_demo"], "--kind", kind,
            "--norm-asset", "arx_x5_sim", "--output", str(output/"policy_bootstrap")]
    if kind == "base":
        args += ["--norm-stats", p["sim_normalization"]]
    return Worker("rsi_loop.vla", args, output=output/f"policy_{time.time_ns()}.log", env=env)


def collect_one(plan, task, candidate, run, seed, vla):
    p = plan["resources"]
    run.mkdir(parents=True, exist_ok=False)
    config = dict(task=task, eval_seed=candidate["eval_seed"], layout_id=candidate["layout_id"],
                  vla_checkpoint_kind=plan["checkpoint_kind"], inference_seed=seed,
                  monitor_enabled=False, gpt=dict(enabled=False), rgb_storage="lossless_video",
                  layout_sha256=digest(candidate["layout_file"]), collection_source_sha256=plan["source_sha256"])
    write_json(run/"run_config.json", dict(config=config, resources=p))
    vla.call("begin_episode", output=str(run/"vla"), seed=seed)
    env = dict(PYTHONPATH=os.pathsep.join([str(ROOT), p["robodojo"], p["robodojo"]+"/XPolicyLab"]),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", OMNI_KIT_ACCEPT_EULA="YES",
               VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json",
               ROBODOJO_RUN_ID=run.name+f"_{time.time_ns()}")
    kit = f'--/renderer/multiGpu/enabled=false --/renderer/activeGpu={plan["sim_gpu"]}'
    kit += " --/app/updateOrder/checkForHydraRenderComplete=1000 --/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true"
    if p.get("nvidia_extensions"):
        kit += " --ext-folder " + p["nvidia_extensions"]
    sim = Worker("rsi_loop.simulator", ["--output", str(run), "--task", task,
                 "--eval-seed", str(candidate["eval_seed"]), "--device", f'cuda:{plan["sim_gpu"]}',
                 "--rgb-storage", "lossless_video", "--headless", "--enable_cameras", "--kit_args", kit],
                 output=run/"simulator.log", env=env)
    try:
        native = sim.call("metadata")
        cfg = LoopConfig(vla_execute_steps=10, recovery_mode="disabled", max_steps=native["max_steps"])
        Controller(SimClient(sim), VLAClient(vla), NoMonitor(), None, cfg, run).run(candidate["layout_id"])
    finally:
        sim.close()
    result = assess(run, plan["checkpoint_kind"])
    result["rgb_verification"] = verify_rgb(run)
    result["stored_bytes"] = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    return result


def task_stats(plan, records):
    result = []
    for task in plan["tasks"]:
        rows = [r for r in records if r["task"] == task["task"]]
        counts = Counter(r.get("label", "interrupted") for r in rows)
        consecutive = 0
        for row in reversed(rows):
            if row.get("label") in ("success", "failure"):
                break
            consecutive += 1
        result.append(dict(task=task["task"], target=task["target"], attempts=len(rows),
            success=counts["success"], failure=counts["failure"], errors=counts["error"]+counts["interrupted"],
            completed=counts["success"]+counts["failure"], consecutive_errors=consecutive,
            latest_run=rows[-1]["run"] if rows else None))
    return result


def report(output, plan, records, state):
    stats = task_stats(plan, records)
    total = sum(r["completed"] for r in stats)
    data = dict(updated_at=now(), pid=os.getpid(), checkpoint_kind=plan["checkpoint_kind"],
                target=sum(t["target"] for t in plan["tasks"]), completed=total,
                free_gib=round(shutil.disk_usage(output).free/2**30, 1),
                api_calls=0, monitor_enabled=False, tasks=stats, **state)
    active = data.get("active")
    if active:
        path = output/active/"events.jsonl"
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size-16384))
                lines = stream.read().splitlines()
            try:
                data["current_step"] = json.loads(lines[-1])["next_step"]
            except (IndexError, ValueError):
                pass
    write_json(output/"status.json", data)
    write_json(output/"dataset_index.json", records)
    rows = []
    for row in stats:
        link = f'<a href="{html.escape(row["latest_run"])}/sensors.mp4">最近录像</a>' if row["latest_run"] else ""
        rows.append(f'<tr><td>{html.escape(row["task"])}</td><td>{row["completed"]}/{row["target"]}</td><td>{row["success"]}</td><td>{row["failure"]}</td><td>{row["errors"]}</td><td>{link}</td></tr>')
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>π0.5 RoboDojo 批量采集</title><style>body{{font:16px/1.6 sans-serif;max-width:1150px;margin:30px auto;padding:0 16px}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:6px;text-align:left}}pre{{white-space:pre-wrap}}</style>
<h1>π0.5 RoboDojo 批量采集</h1><p>权重：{plan["checkpoint_kind"]}；有效完成 {total}/{data["target"]}。成功、失败均保留；运行错误不计入有效实验。</p>
<p>仅 π0.5 控制，GPT 调用 0，OOD 监测关闭；原生任务判定与时限不变。三路原生 RGB 无损保存并逐帧校验。每 30 秒刷新。</p>
<pre>{html.escape(json.dumps({k:v for k,v in data.items() if k!='tasks'}, ensure_ascii=False, indent=2))}</pre>
<table><tr><th>任务</th><th>完成</th><th>成功</th><th>失败</th><th>错误</th><th>录像</th></tr>{''.join(rows)}</table>
<p><a href="dataset_index.json">全部实验索引</a> · <a href="manifest.json">固定实验计划</a> · <a href="runner.log">运行日志</a></p></html>'''
    temporary = output/"index.html.tmp"
    temporary.write_text(text)
    temporary.replace(output/"index.html")


def run_campaign(output):
    output = Path(output).resolve()
    lock = (output/".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((output/"manifest.json").read_text())
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f"Collection source changed; preserve campaign identity: {name}")
    records = [json.loads(p.read_text()) for p in sorted((output/"attempts").glob("*.json"))]
    for record in records:
        if record.get("label") == "running":
            record.update(label="interrupted", error="Previous collector exited before committing this attempt")
            write_json(output/"attempts"/f'{record["attempt_id"]:06d}.json', record)
    state = dict(status="starting", active=None)
    stopped = threading.Event()
    def heartbeat():
        while not stopped.wait(15):
            report(output, plan, [dict(row) for row in records], dict(state))
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    def graceful_stop(signum, frame):
        (output/"STOP").touch()
        state["stop_requested"] = True
    signal.signal(signal.SIGTERM, graceful_stop)
    signal.signal(signal.SIGINT, graceful_stop)
    vla = None
    try:
        report(output, plan, records, state)
        while not (output/"STOP").exists():
            stats = task_stats(plan, records)
            if all(r["completed"] >= r["target"] for r in stats):
                state.update(status="completed", active=None)
                break
            eligible = [(row, task) for row, task in zip(stats, plan["tasks"])
                        if row["completed"] < row["target"] and row["attempts"] < len(task["candidates"])
                        and row["consecutive_errors"] < 3]
            if not eligible:
                state.update(status="blocked_by_task_errors", active=None)
                break
            if shutil.disk_usage(output).free < plan["minimum_free_gib"]*2**30:
                state.update(status="paused_low_disk", active=None)
                if vla is not None:
                    vla.close(); vla = None
                time.sleep(30)
                continue
            waiting, selected = {}, None
            for row, task in sorted(eligible, key=lambda pair: (pair[0]["attempts"], pair[0]["task"])):
                missing = missing_layout_assets(plan["resources"]["robodojo"], task["candidates"][row["attempts"]])
                if missing:
                    waiting[task["task"]] = missing
                    continue
                selected = row, task
                break
            state["waiting_for_assets"] = waiting
            if selected is None:
                state.update(status="paused_missing_assets", active=None)
                if vla is not None:
                    vla.close(); vla = None
                time.sleep(30)
                continue
            row, task = selected
            candidate = task["candidates"][row["attempts"]]
            attempt_id = len(records)
            relative = f'episodes/{task["task"]}/attempt_{row["attempts"]:03d}'
            seed = plan["inference_seed"]+attempt_id
            record = dict(attempt_id=attempt_id, task=task["task"], run=relative, label="running",
                          inference_seed=seed, started_at=now(), **candidate)
            records.append(record)
            write_json(output/"attempts"/f"{attempt_id:06d}.json", record)
            state.update(status="running", active=relative)
            print(json.dumps(dict(event="start", **record)), flush=True)
            try:
                if vla is None:
                    vla = start_vla(plan, output)
                record.update(collect_one(plan, task["task"], candidate, output/relative, seed, vla))
            except Exception as error:
                import traceback
                traceback.print_exc()
                record.update(label="error", error=f"{type(error).__name__}: {error}")
                if vla is not None:
                    vla.close(); vla = None
            record["ended_at"] = now()
            write_json(output/"attempts"/f"{attempt_id:06d}.json", record)
            print(json.dumps(dict(event="finish", **record)), flush=True)
            state["active"] = None
        if (output/"STOP").exists():
            state.update(status="stopped", active=None)
    finally:
        if vla is not None:
            vla.close()
        stopped.set()
        thread.join(timeout=30)
        report(output, plan, records, state)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--resources", default="resources.local.json")
    parser.add_argument("--checkpoint-kind", choices=("base", "demo"), default="base")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--vla-gpu", type=int, default=0)
    parser.add_argument("--sim-gpu", type=int, default=1)
    parser.add_argument("--inference-seed", type=int, default=20260922)
    parser.add_argument("--minimum-free-gib", type=float, default=50)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.command == "run":
        run_campaign(output)
        return
    if args.trials < 1 or args.minimum_free_gib <= 0:
        raise ValueError("Positive trial count and free-space reserve required")
    paths = resources(args.resources)
    tasks = make_plan(paths["robodojo"], args.trials, args.tasks)
    identity = ROOT/"configs"/f"pi05_{args.checkpoint_kind}_identity.json"
    plan = dict(created_at=now(), checkpoint_kind=args.checkpoint_kind, identity_sha256=digest(identity),
        trials_per_task=args.trials, task_count=len(tasks), total_target=len(tasks)*args.trials,
        vla_gpu=args.vla_gpu, sim_gpu=args.sim_gpu, inference_seed=args.inference_seed,
        minimum_free_gib=args.minimum_free_gib, resources=paths, tasks=tasks,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/"rsi_loop").glob("*.py"))},
        protocol=dict(recovery="disabled", monitor="disabled", control="native_25Hz", rgb="lossless_h264rgb_crf0",
                      native_limits=True, layout_order="interleaved_groups_sorted_native_indices",
                      error_policy="retain errors; advance layout; pause task after 3 consecutive errors"))
    output.mkdir(parents=True, exist_ok=False)
    (output/"attempts").mkdir()
    write_json(output/"manifest.json", plan)
    report(output, plan, [], dict(status="prepared", active=None))
    print(json.dumps(dict(output=str(output), tasks=len(tasks), total_target=plan["total_target"])))


if __name__ == "__main__":
    main()
