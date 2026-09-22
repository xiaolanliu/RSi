"""Conda-rsi entry points for integration checks, simulation and demo creation."""
import argparse
from dataclasses import replace
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tomllib


def resources(path):
    root = Path(path).resolve().parent
    values = json.loads(Path(path).read_text())
    return {key: str((root/Path(value).expanduser()).resolve()) for key, value in values.items() if value}


def doctor(args):
    paths = resources(args.resources)
    packages = {}
    for name in ("numpy", "torch", "jax", "jaxlib", "flax", "orbax-checkpoint", "isaacsim", "isaaclab", "av", "jsonschema"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = dict(python=sys.executable, prefix=sys.prefix, conda_prefix=os.environ.get("CONDA_PREFIX"), packages=packages,
                  resources={key: dict(path=value, exists=Path(value).exists()) for key, value in paths.items()},
                  live_gpt_enabled=False)
    if args.hashes:
        import hashlib
        root = Path(__file__).resolve().parents[1]
        identity = json.loads((root/"configs/pi05_base_identity.json").read_text())
        checked = []
        for row in identity["files"]:
            path = Path(paths["pi05_base"])/row["path"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"Original pi05_base parameter identity mismatch: {path}")
            checked.append(row["path"])
        result["verified_original_parameter_files"] = checked
    print(json.dumps(result, indent=2))
    return 0 if all(packages.values()) and all(x["exists"] for x in result["resources"].values()) else 1


class SimClient:
    def __init__(self, worker):
        self.worker, self.obs = worker, None

    def reset(self, seed):
        self.obs = self.worker.call("reset", seed=seed)
        return self.obs

    def step(self, action, source):
        self.obs = self.worker.call("step", action=action, source=source, identity=self.obs.identity)
        return self.obs

    def close(self):
        self.worker.close()


class VLAClient:
    def __init__(self, worker):
        self.worker = worker

    def reset(self):
        self.worker.call("reset")

    def infer(self, obs):
        result = self.worker.call("infer", observation=obs)
        if result["identity"] != obs.identity:
            raise RuntimeError("VLA response is stale")
        return result["actions"]


def evaluate(args):
    from .controller import Controller, LoopConfig
    from .workers import Worker
    from .monitor import StreamingWan, CausalMonitor
    from .recovery import MockRecovery, ContextRecovery, ResponsesTransport, SYSTEM_PROMPT
    from .kinematics import DualMotion
    from agent_closed_loop.monitor import OnlineMonitor
    paths = resources(args.resources)
    config = tomllib.loads(Path(args.config).read_text())
    cfg = LoopConfig(**config["loop"])
    if args.mode:
        cfg = replace(cfg, recovery_mode=args.mode)
    if args.max_steps:
        cfg = replace(cfg, max_steps=args.max_steps)
    if cfg.recovery_mode == "live" and not (args.allow_live_gpt and config["gpt"]["enabled"]):
        raise ValueError("Live mode requires both gpt.enabled=true and --allow-live-gpt")
    if cfg.recovery_mode == "live":
        gpt = config["gpt"]
        if gpt["key_env"] != "RSI_SIM_OPENAI_API_KEY":
            raise ValueError("Simulation live mode requires its dedicated RSI_SIM_OPENAI_API_KEY")
        if not gpt.get("model") or not os.environ.get(gpt["key_env"]):
            raise ValueError("Set a GPT model and the dedicated simulation key before starting workers")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from dataclasses import asdict
    (output/"run_config.json").write_text(json.dumps(dict(config=config, effective_loop=asdict(cfg), resources=paths), indent=2))
    project = str(Path(__file__).resolve().parents[1])
    common = {"PYTHONPATH": project, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    vla_env = dict(common, CUDA_VISIBLE_DEVICES=str(config["vla_gpu"]), XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   OPENPI_DATA_HOME=paths["openpi_cache"])
    vla_env["PYTHONPATH"] = os.pathsep.join([project, paths["openpi"]+"/src", paths["openpi"]+"/packages/openpi-client/src"])
    vla_args = ["--checkpoint", paths["pi05_base"], "--norm-asset", config["norm_asset"], "--output", str(output/"vla")]
    if paths.get("sim_normalization"):
        vla_args += ["--norm-stats", paths["sim_normalization"]]
    vla = Worker("rsi_loop.vla", vla_args, output=output/"vla.log", env=vla_env)
    sim = None
    try:
        sim_env = dict(common, OMNI_KIT_ACCEPT_EULA="YES", VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json")
        sim_env["PYTHONPATH"] = os.pathsep.join([project, paths["robodojo"], paths["robodojo"]+"/XPolicyLab"])
        kit = f'--/renderer/multiGpu/enabled=false --/renderer/activeGpu={config["sim_gpu"]}'
        kit += " --/app/updateOrder/checkForHydraRenderComplete=1000 --/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true"
        if paths.get("nvidia_extensions"):
            kit += " --ext-folder " + paths["nvidia_extensions"]
        sim = Worker("rsi_loop.simulator", ["--output", str(output), "--task", config["task"],
                     "--eval-seed", str(config["eval_seed"]), "--device", f'cuda:{config["sim_gpu"]}',
                     "--headless", "--enable_cameras", "--kit_args", kit], output=output/"simulator.log", env=sim_env)
        monitor = CausalMonitor(OnlineMonitor("models/v11/fold_all.pt", device=config["monitor_device"]),
                                StreamingWan(paths["wan_vae"], config["vision_device"]), config["gripper_open_width_m"])
        recovery = MockRecovery(steps=min(3, cfg.max_recovery_steps))
        if cfg.recovery_mode == "live":
            sys.path.insert(0, paths["gpt_policy"]+"/src")
            from .demonstration import Demonstration
            demo = Demonstration(config["gpt"]["demo_mp4"], task=config["task"], cache=output/"demo") if config["gpt"].get("demo_mp4") else None
            def motion(obs, plan):
                meta = sim.call("metadata")
                return DualMotion(meta["robot_descriptions"], meta["control_dt"])(obs, plan)
            gpt = config["gpt"]
            prompt = Path(gpt["system_prompt_file"]).read_text() if gpt.get("system_prompt_file") else SYSTEM_PROMPT
            recovery = ContextRecovery(ResponsesTransport(**{k: gpt[k] for k in ("model", "enabled", "base_url", "key_env", "timeout")}),
                motion, demo=demo, system_prompt=prompt, max_steps=cfg.max_recovery_steps, output=output/"recovery")
        result = Controller(SimClient(sim), VLAClient(vla), monitor, recovery, cfg, output).run(config["layout_id"])
        print(json.dumps(result, indent=2))
    finally:
        if sim is not None:
            sim.close()
        vla.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor")
    p.add_argument("--resources", default="resources.local.json")
    p.add_argument("--hashes", action="store_true", help="Verify all original pi05_base parameters against published identities")
    p = sub.add_parser("evaluate")
    p.add_argument("--resources", default="resources.local.json")
    p.add_argument("--config", default="configs/loop.toml")
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("mock", "disabled", "live"))
    p.add_argument("--max-steps", type=int)
    p.add_argument("--allow-live-gpt", action="store_true")
    p = sub.add_parser("promote-demo")
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("prepare-context", help="Write a real-observation request without any API call")
    p.add_argument("--run", required=True)
    p.add_argument("--step", type=int)
    p.add_argument("--demo")
    p.add_argument("--resources", default="resources.local.json")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        raise SystemExit(doctor(args))
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "prepare-context":
        prepare_context(args)
    else:
        from .demonstration import promote_success
        print(promote_success(args.run, args.output))


def prepare_context(args):
    import numpy as np
    from .contracts import Observation
    from .recovery import ContextRecovery
    run = Path(args.run)
    events = [json.loads(line) for line in (run/"events.jsonl").read_text().splitlines()]
    if args.step is None:
        args.step = events[-1]["step"]
    event = next(e for e in events if e["step"] == args.step)
    with np.load(run/"observations"/f"{args.step:06d}.npz", allow_pickle=False) as data:
        obs = Observation(str(data["episode_id"]), args.step, float(data["time"]), data["state"],
            {k: data[k] for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")}, str(data["instruction"]),
            data["eef_positions"], data["eef_quaternions"],
            measured_gripper_openings=data["measured_gripper_openings"] if "measured_gripper_openings" in data else None)
    demo = None
    if args.demo:
        paths = resources(args.resources)
        sys.path.insert(0, paths["gpt_policy"]+"/src")
        from .demonstration import Demonstration
        demo = Demonstration(args.demo, task=obs.instruction, cache=Path(args.output).parent/"demo_cache")
    past = [e for e in events if e["step"] < args.step and e["time"] >= obs.time-3]
    selected = np.linspace(0, len(past)-1, min(3, len(past)), dtype=int) if past else []
    history = []
    for index in selected:
        e = past[index]
        with np.load(run/"observations"/f'{e["step"]:06d}.npz', allow_pickle=False) as data:
            history.append(dict(**{k: e[k] for k in ("step", "time", "source", "alarm")},
                                state=data["state"].tolist(), image_top=data["cam_high"]))
    recovery = ContextRecovery(None, None, demo=demo, output=Path(args.output).parent)
    payload = recovery.request(obs, event["risk"], history)
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(json.dumps(dict(request=args.output, step=args.step, api_calls=0, demo_present=demo is not None)))


if __name__ == "__main__":
    main()
