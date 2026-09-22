"""Native RoboDojo actions and success criteria behind a single-owner worker."""
import argparse
import importlib
import json
from pathlib import Path
import uuid

import numpy as np

from .contracts import Observation, action_array
from .workers import serve


class NoAutonomousPolicy:
    def __init__(self, **kwargs):
        pass

    def call(self, func_name, **kwargs):
        if func_name != "reset":
            raise RuntimeError("RSI owns policy execution")

    def close(self):
        pass


class Session:
    def __init__(self, env, output, task):
        self.env, self.output, self.task = env, Path(output), task
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode_id = None
        self.tick, self.obs, self.writer = 0, None, None
        self.env._stream_vision = lambda *a, **kw: None
        self.metadata = dict(control_dt=1/env.obs_manager.collect_freq,
                             max_steps=int(env.step_lim), task=task, action_semantics="absolute_joint_targets",
                             gripper_semantics="normalized_opening_0_to_1")

    def observe(self):
        raw = self.env.get_obs()
        state = np.concatenate([np.r_[raw["state"][f"{a}_arm_joint_state"], raw["state"][f"{a}_ee_joint_state"]]
                                for a in ("left", "right")]).astype(np.float32)
        measured = {}
        for robot in self.env.robot_manager.robot_list:
            name = robot.arm_name.split("_")[0]
            if robot.type == "target" and name in ("left", "right"):
                value = float(self.env.robot_manager.get_end_effector_real_val(robot)[0][0])
                low, high = robot.gripper_scale
                measured[name] = (value-low)/(high-low) if robot.gripper_move["sign"] == 1 else (high-value)/(high-low)
        openings = np.asarray([measured[name] for name in ("left", "right")], np.float32)
        images = {}
        for name, aliases in dict(cam_high=("cam_high", "cam_head", "head_camera", "top_camera"),
                                  cam_left_wrist=("cam_left_wrist", "left_camera"),
                                  cam_right_wrist=("cam_right_wrist", "right_camera")).items():
            source = next((key for key in aliases if key in raw["vision"]), None)
            if source is None:
                raise ValueError(f"Missing native camera {name}")
            images[name] = np.asarray(raw["vision"][source]["color"], np.uint8)[..., :3]
            if images[name].ndim != 3:
                raise ValueError("Expected latest native RGB frame, not a temporal stack")
        poses = np.asarray([raw["state"][f"{a}_ee_pose"] for a in ("left", "right")], np.float32)
        ended = bool(self.env.end_flag[0])
        success = ended and bool(self.env.success[0])
        truncated = ended and not success and self.tick >= self.env.step_lim
        self.obs = Observation(self.episode_id, self.tick, self.tick*self.metadata["control_dt"], state,
            images, raw["instruction"], poses[:, :3], poses[:, 3:], ended and not truncated, truncated, success, openings)
        from PIL import Image, ImageOps
        frame = np.concatenate([np.asarray(ImageOps.pad(Image.fromarray(image), (640, 480)))
                                for image in images.values()], axis=1)
        self.writer.append_data(frame)
        np.savez_compressed(self.output / "observations" / f"{self.tick:06d}.npz", state=state,
                            time=self.obs.time, episode_id=self.episode_id, instruction=self.obs.instruction,
                            measured_gripper_openings=openings,
                            eef_positions=poses[:, :3], eef_quaternions=poses[:, 3:], **images)
        return self.obs

    def reset(self, seed):
        if self.episode_id is not None:
            raise ValueError("One fresh native reset per worker; restart for the next episode")
        self.env.reset(seed=[seed])
        self.env.run_reward()
        if hasattr(self.env, "get_score"):
            self.env.get_score()
        groups = ("check_list", "final_check_list", "trigger_check_list")
        counts = {key: len(getattr(self.env.reward_manager, key)[0]) for key in groups}
        if not sum(counts.values()):
            raise RuntimeError("Native success conditions are empty; refusing vacuous success")
        if getattr(self.env, "interact", False) and hasattr(self.env, "query_support_arm_traj"):
            self.env.query_support_arm_traj(env_idx=0)
        self.episode_id = uuid.uuid4().hex
        (self.output/"observations").mkdir(exist_ok=True)
        import imageio.v2 as imageio
        self.writer = imageio.get_writer(str(self.output/"sensors.mp4"), fps=1/self.metadata["control_dt"],
            codec="libx264", pixelformat="yuv420p", macro_block_size=2, output_params=["-movflags", "+faststart"])
        self.metadata["native_conditions"] = counts
        descriptions = {}
        for robot in self.env.robot_manager.robot_list:
            name = robot.arm_name.split("_")[0]
            if robot.type != "target" or name not in ("left", "right"):
                continue
            key = self.env.robot_manager.robot_key[self.env.robot_manager.robot_list.index(robot)]
            root = key.data.root_pose_w[0].detach().cpu().numpy().copy()
            root[:3] -= self.env.robot_manager.scene.env_origins[0].detach().cpu().numpy()
            descriptions[name] = dict(urdf=str(robot.urdf_path), names=list(robot.arm_joints_name),
                base_link=robot.base_link, ee_link=robot.ee_link_name,
                root_pose=root.tolist(), limits=key.data.soft_joint_pos_limits[0, robot.arm_joint_indices].detach().cpu().tolist())
        self.metadata["robot_descriptions"] = descriptions
        (self.output/"native_metadata.json").write_text(json.dumps(self.metadata, indent=2))
        return self.observe()

    def step(self, action, source, identity):
        if self.obs is None or tuple(identity) != self.obs.identity or self.obs.terminated or self.obs.truncated:
            raise ValueError("Stale or terminal action request")
        a = action_array([action])[0]
        command = {key: value for arm, start in (("left", 0), ("right", 7))
                   for key, value in ((f"{arm}_arm_joint_state", a[start:start+6]),
                                      (f"{arm}_ee_joint_state", a[start+6:start+7]))}
        before = int(self.env.take_action_cnt[0])
        self.env.take_action(command)
        if int(self.env.take_action_cnt[0]) != before+1:
            raise RuntimeError("Native command did not execute exactly once")
        self.tick += 1
        return self.observe()

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        ended = self.obs is not None and (self.obs.terminated or self.obs.truncated)
        invalid = 0 in getattr(self.env, "unstable_envs", set())
        result = dict(complete=ended, valid_for_success_rate=ended and not invalid,
                      native_success=bool(ended and not invalid and self.obs.success),
                      native_control_steps=self.tick, native_step_limit=self.metadata["max_steps"],
                      status="native_completed" if ended and not invalid else "incomplete",
                      task=self.task)
        (self.output/"native_outcome.json").write_text(json.dumps(result, indent=2))

    def dispatch(self, op, args):
        if op == "metadata":
            return self.metadata
        if op not in ("reset", "step", "close"):
            raise ValueError("Unknown simulator operation")
        return getattr(self, op)(**args)


def main():
    import cv2  # Load before Kit prepends its bundled OpenCV dependencies.
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--socket", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task", default="build_tower")
    p.add_argument("--eval-seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.headless = args.enable_cameras = True
    from env.global_configs import BENCHMARK, ROOT_DIR, ENV_CONFIG_PATH
    registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    task_path = registry.task_config_path(str(Path(ROOT_DIR)/"task"/BENCHMARK/"config"), args.task)
    import yaml
    monitor = bool(yaml.safe_load(Path(task_path).read_text()).get("Articulation"))
    if monitor:
        from src.eval_client.physx_warning_monitor import get_monitor
        get_monitor().start(enabled=True)
    app = AppLauncher(args).app
    env = session = None
    try:
        from omegaconf import OmegaConf
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from src.eval_client import eval_env
        root = Path(ENV_CONFIG_PATH)
        evaluation = load_yaml(str(root/"arx_x5.yml"))
        evaluation.update(task_name=args.task, num_envs=1, device_id=0, eval_batch=False,
                          policy_name="Pi_05", additional_info="rsi", seed=args.eval_seed,
                          physx_monitor_enabled=monitor)
        values = {key: load_yaml(str(root/key/(evaluation["config"][key]+".yml"))) for key in ("sim", "scene", "camera", "robot")}
        values.update(eval_cfg=evaluation, deploy_cfg=dict(port=1, policy_name="Pi_05"), task_env=load_yaml(task_path))
        cfg = process_randomization(OmegaConf.create(values))
        cfg, _ = process_config(cfg, task_name=args.task)
        cfg.sim.scene.num_envs = cfg.eval_cfg.eval_num = 1
        cfg.camera.default_frequency = cfg.eval_cfg.observation.collect_freq
        cfg.sim.seed = [0]
        for robot in cfg.robot.robots:
            robot.need_planner = False
        eval_env.WsModelClient = NoAutonomousPolicy
        env = eval_env.create_eval_env(cfg, app)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output)/"resolved_simulator.json").write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
        session = Session(env, args.output, args.task)
        serve(args.socket, session.dispatch)
    finally:
        if session:
            session.close()
        if env:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
