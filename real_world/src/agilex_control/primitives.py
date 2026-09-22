"""Reusable agent tools. Task/object decisions stay outside these primitives."""

import json
import math
import socket
import time
from pathlib import Path

ARM_TOOLS = {
    "move_left": ("move", "left"),
    "move_right": ("move", "right"),
    "set_gripper_left": ("gripper", "left"),
    "set_gripper_right": ("gripper", "right"),
}


def validate_actuator_arguments(config, name, arguments):
    """Pure argument validation shared by individual calls and finite phases.

    Cartesian IK and current-state checks deliberately stay at execution time.
    """
    from .protocol import validate_joints
    from .trajectory import GripperTrajectory

    arguments = dict(arguments)
    operation = name
    allowed = {"duration_s", "settle_s", "output", "dry_run", "require_cameras"}
    if name in ARM_TOOLS:
        operation, arm = ARM_TOOLS[name]
        if "arm" in arguments and arguments["arm"] != arm:
            raise ValueError("Tool arm cannot be overridden")
        arguments["arm"] = arm
        allowed |= {"arm", "target"} if operation == "move" else {"arm", "width_mm", "effort_nm"}
    elif name == "move_both":
        allowed.add("targets")
        if not isinstance(arguments.get("targets"), dict) or set(arguments["targets"]) != {"left", "right"}:
            raise ValueError("Both left and right targets are required")
    else:
        raise ValueError("Unknown actuator primitive: " + name)
    if set(arguments) - allowed:
        raise ValueError("Unknown actuator arguments: " + str(sorted(set(arguments) - allowed)))
    for field in ("dry_run", "require_cameras"):
        if field in arguments and not isinstance(arguments[field], bool):
            raise ValueError(field + " must be boolean")
    if not arguments.get("dry_run", False):
        if not arguments.get("output"):
            raise ValueError("Actuator output path is required")
        duration = arguments.get("duration_s", 1.5 if operation == "gripper" else None)
        settle = arguments.get("settle_s", 1 if operation == "gripper" else 3)
        if (not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0
                or not isinstance(settle, (int, float)) or not math.isfinite(settle) or settle < 0):
            raise ValueError("Finite positive duration and nonnegative settle required")
    if operation == "gripper":
        GripperTrajectory(arguments["width_mm"], arguments.get("effort_nm", 1.0),
                          config, arguments["arm"])
    else:
        targets = (arguments["targets"] if name == "move_both"
                   else {arguments["arm"]: arguments["target"]})
        if any(arm not in config["arms"] for arm in targets):
            raise ValueError("Unknown arm for selected site")
        for target in targets.values():
            if not isinstance(target, dict) or ("joint_deg" in target) == ("xyz_mm" in target):
                raise ValueError("Choose exactly one target: joint_deg or xyz_mm")
            if set(target) - ({"joint_deg"} if "joint_deg" in target else {"xyz_mm", "rpy_deg"}):
                raise ValueError("Unknown target fields")
            if "joint_deg" in target:
                validate_joints(target["joint_deg"], config["joint_limits_deg"])
            else:
                for field in ("xyz_mm", "rpy_deg"):
                    if field in target and (not isinstance(target[field], (list, tuple))
                            or len(target[field]) != 3
                            or not all(isinstance(v, (int, float)) and math.isfinite(v)
                                       for v in target[field])):
                        raise ValueError("Cartesian coordinates and RPY require three finite values")
    return operation, arguments


def state(config, seconds=1):
    from .can_bus import CanBus

    if not 0.5 <= seconds <= 30:
        raise ValueError("Capture duration must be .5–30 seconds")
    with CanBus(config["arms"]) as bus:
        bus.pump(seconds)
        return dict(
            captured_unix=time.time(), states=bus.state(), control_frames=bus.events
        )


def call(config, name, arguments):
    result = dict(
        primitive=name, ok=False, started_unix=time.time(), data=None, error=None
    )
    try:
        if (
            config.get("expected_hostname")
            and socket.gethostname() != config["expected_hostname"]
        ):
            raise ValueError("Selected site configuration does not match this host")
        arguments = dict(arguments)
        operation = name
        if name in ARM_TOOLS or name == "move_both":
            operation, arguments = validate_actuator_arguments(config, name, arguments)
        if name == "get_state":
            data = state(config, arguments.get("seconds", 1))
        elif name == "observe":
            from .cameras import request
            from .ros_capture import capture as ros_capture, should_fallback

            include_state = config.get("observe_arm_state", True)
            if not isinstance(include_state, bool):
                raise ValueError("observe_arm_state must be boolean")
            if not include_state and not config.get(
                "observe_arm_state_unavailable_reason"
            ):
                raise ValueError(
                    "Unavailable arm state requires a site-specific reason"
                )
            output = Path(arguments["output"]).resolve()
            if output.exists():
                raise FileExistsError("Observation requires a new output directory")
            try:
                data = request(
                    config["runtime_dir"], dict(op="capture", output=str(output))
                )
            except (RuntimeError, FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                if not should_fallback(exc):
                    raise
                data = ros_capture(config, output)
                data["usb_capture_error"] = str(exc)
            if include_state:
                measured = state(config, 0.5)
                (output / "state.json").write_text(
                    json.dumps(measured, indent=2) + "\n"
                )
            else:
                measured = {"states": None, "captured_unix": None}
                data["state_unavailable_reason"] = config[
                    "observe_arm_state_unavailable_reason"
                ]
            data.update(
                states=measured["states"],
                state_captured_unix=measured["captured_unix"],
                output=str(output),
                synchronized_cameras=False,
                available_modalities=["rgb", "aligned_depth", "color_intrinsics"]
                + (["arm_state"] if include_state else []),
                missing_modalities=[] if include_state else ["arm_state"],
            )
            if data.get("source") == "ros":
                data["available_modalities"] = [
                    item if item != "aligned_depth" else "depth"
                    for item in data["available_modalities"]
                ]
                data.setdefault("warnings", []).append(
                    "RGBD came from ROS topics; depth is image_rect_raw and not aligned to color."
                )
            for camera in data["cameras"]:
                data["cameras"][camera]["files"] = {
                    k: str(output / camera / v)
                    for k, v in [
                        ("rgb", "rgb.jpg"),
                        ("depth", "depth.npy"),
                        ("metadata", "metadata.json"),
                    ]
                }
            (output / "observation.json").write_text(json.dumps(data, indent=2) + "\n")
        elif operation == "gripper" and name in ARM_TOOLS:
            from .motion import set_gripper
            width = arguments["width_mm"]
            effort = arguments.get("effort_nm", 1.0)
            if arguments.get("dry_run", False):
                result.update(
                    ok=True,
                    data={
                        "arm": arguments["arm"],
                        "width_mm": width,
                        "effort_nm": effort,
                        "executed": False,
                    },
                    finished_unix=time.time(),
                )
                return result
            data = set_gripper(
                config,
                arguments["arm"],
                width,
                effort,
                arguments.get("duration_s", 1.5),
                arguments.get("settle_s", 1),
                Path(arguments["output"]),
                arguments.get("require_cameras", True),
            )
            data = {
                k: v
                for k, v in data.items()
                if k not in ["commands", "samples", "control_frames"]
            }
            if "final" in data:
                final = data["final"][arguments["arm"]]
                data["gripper_feedback"] = {
                    k: v for k, v in final.items() if k.startswith("gripper_")
                }
            if data["status"] != "command_stream_completed":
                result["error"] = data.get("error", "Gripper command incomplete")
        elif name in ("move_left", "move_right", "move_both"):
            from .motion import move, move_both

            paired = name == "move_both"
            targets = (
                arguments["targets"]
                if paired
                else {arguments["arm"]: arguments["target"]}
            )
            plans, joint_targets = {}, {}
            measured, kinematics = None, None
            # Validate and plan every target before opening any action writer.
            for arm, target in targets.items():
                if "xyz_mm" in target:
                    from .kinematics import Kinematics

                    if measured is None:
                        measured = state(config, 0.5)
                        kinematics = Kinematics(config)
                    plan = kinematics.plan(
                        measured["states"], arm, target["xyz_mm"], target.get("rpy_deg")
                    )
                    joints = plan["q_target_deg"]
                else:
                    joints = target["joint_deg"]
                    plan = {"arm": arm, "q_target_deg": joints}
                plans[arm], joint_targets[arm] = plan, joints
            plan_result = (
                {"plans": plans} if paired else {"plan": plans[arguments["arm"]]}
            )
            if arguments.get("dry_run", False):
                result.update(
                    ok=True,
                    data={**plan_result, "executed": False},
                    finished_unix=time.time(),
                )
                return result
            timing = (
                arguments["duration_s"],
                arguments.get("settle_s", 3),
                Path(arguments["output"]),
                arguments.get("require_cameras", True),
            )
            if paired:
                data = move_both(config, joint_targets, *timing)
            else:
                arm = arguments["arm"]
                data = move(config, arm, joint_targets[arm], *timing)
            data = {
                k: v
                for k, v in data.items()
                if k not in ["commands", "samples", "control_frames"]
            }
            data.update(plan_result)
            if "final" in data and "baseline" in data:
                feedback = {}
                for arm, joints in joint_targets.items():
                    final, baseline = data["final"][arm], data["baseline"][arm]
                    if final.get("complete"):
                        feedback[arm] = dict(
                            joint_error_deg=[
                                a - b for a, b in zip(final["joint_deg"], joints)
                            ],
                            measured_xyz_change_mm=[
                                a - b
                                for a, b in zip(
                                    final["pose_mm_deg"][:3],
                                    baseline["pose_mm_deg"][:3],
                                )
                            ],
                        )
                if paired:
                    data["arm_feedback"] = feedback
                else:
                    data.update(feedback.get(arguments["arm"], {}))
            if data["status"] != "command_stream_completed":
                result["error"] = data.get("error", "Motion incomplete")
        elif name == "camera_status":
            from .cameras import request

            data = request(config["runtime_dir"], {"op": "status"})
        elif name == "stop":
            from .motion import stop_motion

            data = stop_motion(config)
        else:
            raise ValueError("Unknown primitive: " + name)
        result["data"] = data
        result["ok"] = result["error"] is None
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    result["finished_unix"] = time.time()
    return result
