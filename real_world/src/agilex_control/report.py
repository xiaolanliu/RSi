"""Compact measured evidence for a primitive or phase; no task-success inference."""

import math


def _delta(actual, target):
    if actual is None or target is None or len(actual) != len(target):
        return None
    return [round(a - b, 4) for a, b in zip(actual, target)]


def summarize(result, config=None):
    """Keep actuation and observation outcomes separate, retaining evidence paths."""
    phase = "action_status" in result and "steps" in result
    steps = result.get("steps", []) if phase else [{"result": result}]
    actions = [s["result"] for s in steps if isinstance(s.get("result"), dict)]
    observation = result.get("observation", {}) if phase else {}
    observed = observation.get("result")
    if not phase and result.get("primitive") == "observe":
        observed = result
    data = (observed or {}).get("data") or {}
    states = data.get("states")
    baseline, targets = {}, {}
    for action in actions:
        detail = action.get("data") or {}
        for arm, state in (detail.get("baseline") or {}).items():
            baseline.setdefault(arm, state)
        if not states:
            states = detail.get("final") or detail.get("states")
        plans = detail.get("plans", {})
        if detail.get("plan"):
            plans = {detail["plan"]["arm"]: detail["plan"]}
        targets.update(plans)
    # The last executed action is the freshest fallback if observation failed.
    if not data.get("states"):
        for action in reversed(actions):
            detail = action.get("data") or {}
            if detail.get("final") or detail.get("states"):
                states = detail.get("final") or detail["states"]
                break
    output = {
        "ok": result.get("ok", False),
        "status": result.get("status", result.get("primitive")),
        "action_status": result.get("action_status") if phase else (
            "not_executed" if (result.get("data") or {}).get("executed") is False
            else (result.get("data") or {}).get("status")),
        "observation_status": observation.get("status") if phase else (
            "completed" if observed and observed.get("ok") else "not_requested"),
        "error": result.get("error"),
        "observation_error": (observed or {}).get("error"),
        "task_success": None,
        "pose_reference": "selected arm base to link6; millimetres/degrees",
        "states_source": "post_observation" if data.get("states") else "primitive_feedback",
        "state_captured_unix": data.get("state_captured_unix", data.get("captured_unix")),
        "arms": {}, "images": [],
        "evidence": {"phase": result.get("output_dir"),
                     "steps": [s.get("result_file") for s in steps if s.get("result_file")],
                     "observation": observation.get("result_file")},
    }
    for arm, state in (states or {}).items():
        start = baseline.get(arm, {})
        plan = targets.get(arm, {})
        actual, target = state.get("pose_mm_deg"), plan.get("pose_target_mm_deg")
        xyz_error = _delta(actual[:3], target[:3]) if actual and target else None
        gripper_before, gripper_now = start.get("gripper_mm"), state.get("gripper_mm")
        effort_before, effort_now = start.get("gripper_effort_nm"), state.get("gripper_effort_nm")
        item = dict(
            complete=state.get("complete"), stale=state.get("stale"),
            actual_pose_mm_deg=actual, target_pose_mm_deg=target,
            joint_deg=state.get("joint_deg"),
            xyz_error_mm=xyz_error,
            position_error_mm=round(math.sqrt(sum(x*x for x in xyz_error)), 4) if xyz_error else None,
            joint_error_deg=_delta(state.get("joint_deg"), plan.get("q_target_deg")),
            orientation_error_deg=None,
            arm_status=state.get("arm_status"), error_code=state.get("error_code"),
            gripper_fault_bits=state.get("gripper_fault_bits"),
            gripper_mm=gripper_now, gripper_effort_nm=effort_now,
            gripper_change_mm=round(gripper_now-gripper_before, 4) if gripper_now is not None and gripper_before is not None else None,
            gripper_effort_change_nm=round(effort_now-effort_before, 4) if effort_now is not None and effort_before is not None else None,
        )
        if target is not None and config and state.get("joint_deg"):
            try:
                from scipy.spatial.transform import Rotation
                from .kinematics import Kinematics

                actual_r = Kinematics(config).transform(state["joint_deg"])[:3, :3]
                error_r = Rotation.from_euler("xyz", target[3:], degrees=True).inv() * Rotation.from_matrix(actual_r)
                item["orientation_error_deg"] = round(math.degrees(error_r.magnitude()), 4)
            except (ImportError, OSError, ValueError, KeyError) as exc:
                item["orientation_error_unavailable"] = str(exc)
        output["arms"][arm] = item
    for camera, meta in data.get("cameras", {}).items():
        files = meta.get("files", {})
        output["images"].append({"camera": camera, "serial": meta.get("serial"),
                                 "color_timestamp_ms": meta.get("color_timestamp_ms"),
                                 "timestamp_domain": meta.get("timestamp_domain"),
                                 "rgb": files.get("rgb"), "metadata": files.get("metadata"),
                                 "depth": files.get("depth")})
    output["cross_camera_hardware_sync"] = data.get("synchronized_cameras", False)
    if phase and "recording" in result:
        output["recording"] = result["recording"]
    if result.get("started_unix") is not None and result.get("finished_unix") is not None:
        output["elapsed_s"] = round(result["finished_unix"]-result["started_unix"], 4)
    output["steps"] = [{"primitive": s.get("primitive", (s.get("result") or {}).get("primitive")),
                        "status": s.get("status"),
                        "error": (s.get("result") or {}).get("error")} for s in steps]
    if not phase and result.get("primitive") == "stop":
        output["stop"] = result.get("data")
    if not phase and (result.get("data") or {}).get("executed") is False:
        output["plans"] = targets
    return output
