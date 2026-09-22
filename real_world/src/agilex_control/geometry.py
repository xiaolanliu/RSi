"""Read-only RGBD and rigid-point geometry in millimetres; no hardware access.

T_target_source maps source column coordinates into the target frame. Depth
arrays are raw depth units aligned to the recorded color frame. A link6-local
task point must be explicit: this module never assumes a fingertip TCP.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def _vector(value, length, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return value


def _rotation(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (3, 3) or not np.isfinite(value).all()
            or not np.allclose(value.T @ value, np.eye(3), atol=1e-5, rtol=0)
            or abs(np.linalg.det(value) - 1) > 1e-5):
        raise ValueError("Rotation must be a finite 3x3 proper orthonormal matrix")
    return value


def _rigid(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7, rtol=0)):
        raise ValueError("Transform must be a finite homogeneous 4x4 matrix")
    _rotation(value[:3, :3])
    return value


def _inverse(value):
    value = _rigid(value)
    result = np.eye(4)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ value[:3, 3]
    return result


def transform_points(T_target_source, points):
    """Transform a finite XYZ point or N x 3 points, preserving shape."""
    transform = _rigid(T_target_source)
    points = np.asarray(points, dtype=float)
    if points.ndim not in (1, 2) or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("Points must be finite XYZ or N x 3 values")
    return points @ transform[:3, :3].T + transform[:3, 3]


def link6_target(task_point_base_mm, rotation_matrix, point_link6_mm):
    """Return link6 XYZ satisfying task_point = R @ explicit_local_point + XYZ."""
    target = _vector(task_point_base_mm, 3, "task_point_base_mm")
    local = _vector(point_link6_mm, 3, "point_link6_mm")
    return target - _rotation(rotation_matrix) @ local


def rpy_rotation(rpy_deg):
    """Return the existing move convention Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = np.deg2rad(_vector(rpy_deg, 3, "rpy_deg"))
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def point_in_link6(base_point_mm, actual_T_base_link6):
    """Express an observed base-frame point in the captured link6 frame."""
    return transform_points(_inverse(actual_T_base_link6), _vector(base_point_mm, 3, "base_point_mm"))


def _dimensions(meta):
    dimensions = [meta["width"], meta["height"]]
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x <= 0 for x in dimensions):
        raise ValueError("Frame width and height must be positive integers")
    return tuple(int(x) for x in dimensions)


def pixel_rays(meta, pixels):
    """Deproject pixels to Z=1 rays using this frame's distortion convention.

    Nonzero inverse Brown coefficients use RealSense's existing deprojection;
    they are never interpreted as OpenCV forward Brown coefficients. Zero
    coefficients need only NumPy. OpenCV and RealSense imports are lazy.
    """
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if not np.isfinite(pixels).all():
        raise ValueError("Image pixels must be finite N x 2 values")
    focal = _vector([meta["fx"], meta["fy"]], 2, "Frame focal lengths")
    center = _vector([meta["cx"], meta["cy"]], 2, "Frame principal point")
    if np.any(focal <= 0):
        raise ValueError("Frame focal lengths must be positive")
    model = str(meta["model"]).split(".")[-1]
    if model not in ("none", "brown_conrady", "inverse_brown_conrady"):
        raise ValueError(f"Unsupported distortion model: {meta['model']}")
    coeffs = np.zeros(5) if model == "none" else _vector(meta["coeffs"], 5, "Distortion coefficients")
    if not len(pixels):
        return np.empty((0, 3))
    if model == "none" or not np.any(coeffs):
        rays = np.column_stack(((pixels - center) / focal, np.ones(len(pixels))))
    elif model == "brown_conrady":
        import cv2
        camera_matrix = np.array([[focal[0], 0, center[0]], [0, focal[1], center[1]], [0, 0, 1]])
        normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), camera_matrix, coeffs).reshape(-1, 2)
        rays = np.column_stack((normalized, np.ones(len(pixels))))
    else:
        import pyrealsense2 as rs
        intr = rs.intrinsics()
        intr.width, intr.height = _dimensions(meta)
        intr.fx, intr.fy, intr.ppx, intr.ppy = map(float, [*focal, *center])
        intr.model = rs.distortion.inverse_brown_conrady
        intr.coeffs = coeffs.tolist()
        rays = np.asarray([rs.rs2_deproject_pixel_to_point(intr, p.tolist(), 1.0)
                           for p in pixels], dtype=float).reshape(-1, 3)
        if np.any(rays[:, 2] <= 0):
            raise ValueError("Deprojection returned invalid forward rays")
        rays = rays / rays[:, 2:3]
    if not np.isfinite(rays).all():
        raise ValueError("Deprojection returned non-finite rays")
    return rays


def pixel_to_point(meta, depth, pixel, radius=2, min_valid=1):
    """Use median positive finite aligned depth near a pixel; return XYZ/stats.

    Patch spread is reported, not interpreted as object certainty. A boundary
    patch may mix objects. No task-specific depth range or target is assumed.
    """
    width, height = _dimensions(meta)
    pixel = _vector(pixel, 2, "pixel")
    if not (0 <= pixel[0] <= width - 1 and 0 <= pixel[1] <= height - 1):
        raise ValueError("Pixel lies outside the recorded frame")
    for value, name, minimum in [(radius, "radius", 0), (min_valid, "min_valid", 1)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    depth = np.asarray(depth, dtype=float)
    if depth.shape != (height, width):
        raise ValueError("Aligned depth dimensions disagree with frame intrinsics")
    scale = float(meta["depth_scale_m"]) * 1000
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Depth scale must be finite and positive")
    x, y = np.rint(pixel).astype(int)
    patch = depth[max(0, y - radius):min(height, y + radius + 1),
                  max(0, x - radius):min(width, x + radius + 1)]
    valid = patch[(patch > 0) & np.isfinite(patch)] * scale
    valid = valid[np.isfinite(valid)]
    if len(valid) < min_valid:
        raise ValueError(f"Only {len(valid)} valid depth samples; {min_valid} required")
    depth_mm = float(np.median(valid))
    point = pixel_rays(meta, pixel)[0] * depth_mm
    if not np.isfinite(point).all():
        raise ValueError("Depth deprojection returned a non-finite point")
    return {"pixel": pixel.tolist(), "point_camera_mm": point.tolist(), "depth_mm": depth_mm,
            "depth_stats": {"radius_px": int(radius), "valid_count": len(valid),
                            "patch_count": int(patch.size), "valid_fraction": len(valid) / patch.size,
                            "p10_mm": float(np.percentile(valid, 10)),
                            "p90_mm": float(np.percentile(valid, 90)),
                            "min_mm": float(valid.min()), "max_mm": float(valid.max())}}


def _json_input(value):
    if isinstance(value, dict):
        data, content, source = value, json.dumps(value, sort_keys=True, allow_nan=False).encode(), "inline"
    else:
        path = Path(value).resolve()
        content, source = path.read_bytes(), str(path)
        data = json.loads(content)
    return data, {"source": source, "sha256": hashlib.sha256(content).hexdigest()}


def _top_transform(calibration, arm):
    entry = calibration["arms"][arm]
    # T_base_d455 is the existing published schema, not a model assumption.
    key = "T_base_top_camera" if "T_base_top_camera" in entry else "T_base_d455"
    return _rigid(entry[key])


def _base_transform(calibration, target, source):
    if target == source:
        return np.eye(4)
    direct, reverse = f"T_{target}base_{source}base", f"T_{source}base_{target}base"
    if direct in calibration:
        return _rigid(calibration[direct])
    if reverse in calibration:
        return _inverse(calibration[reverse])
    return _top_transform(calibration, target) @ _inverse(_top_transform(calibration, source))


def _captured_link6(observation, config, arm):
    if arm not in config["arms"]:
        raise ValueError("Arm is absent from configuration")
    state = (observation.get("states") or {}).get(arm, {})
    if not state.get("complete") or state.get("stale") != []:
        raise ValueError("Geometry requires complete, non-stale captured joint feedback")
    joints = _vector(state["joint_deg"], 6, "Captured joint_deg")
    from .kinematics import Kinematics
    transform = Kinematics(config).transform(joints)
    source = {"arm": arm, "joint_deg": joints.tolist(),
              "state_captured_unix": observation.get("state_captured_unix"),
              "error_code": state.get("error_code"),
              "kinematics_source": config.get("fk_file", "configured mdh_mm_rad")}
    return transform, source


def observation_point(observation_dir, camera, pixel, target_arm, calibration, config,
                      radius=2, min_valid=1):
    """Map a point from a saved observe result to a selected physical base.

    Camera identity is matched by serial. Wrist transforms use the observation's
    actual joints through Kinematics.transform, including for cross-arm views.
    Historical observations are allowed for offline analysis; timestamps and
    synchronization limits are returned, not represented as live verification.
    """
    directory = Path(observation_dir).resolve()
    observation = json.loads((directory / "observation.json").read_text())
    calibration, calibration_source = _json_input(calibration)
    config, config_source = _json_input(config)
    if not config.get("site_id") or calibration.get("site_id") != config["site_id"]:
        raise ValueError("Calibration and configuration site identities disagree or are missing")
    if calibration.get("units") != "mm":
        raise ValueError("Calibration must explicitly use millimetres")
    if target_arm not in calibration["arms"] or target_arm not in config["arms"]:
        raise ValueError("Target arm is absent from calibration or configuration")
    if camera not in observation["cameras"] or camera not in config["cameras"] or Path(camera).name != camera:
        raise ValueError("Camera must name a configured camera in this observation")
    if "aligned_depth" not in observation.get("available_modalities", []):
        raise ValueError("Observation does not declare aligned depth")
    camera_dir = directory / camera
    meta = json.loads((camera_dir / "metadata.json").read_text())
    captured = observation["cameras"][camera]
    serial = str(meta["serial"])
    if serial != str(config["cameras"][camera]) or serial != str(captured["serial"]):
        raise ValueError("Camera serial disagrees with observation/configuration")
    for key in ("width", "height", "fx", "fy", "cx", "cy", "model", "coeffs", "depth_scale_m",
                "color_frame_number", "depth_frame_number", "received_unix"):
        if key in captured and meta.get(key) != captured[key]:
            raise ValueError(f"Frame metadata disagrees with observation: {key}")
    modes = calibration.get("camera_modes", {})
    if camera in modes and list(_dimensions(meta)) != list(modes[camera][:2]):
        raise ValueError("Frame dimensions changed from the calibrated mode; revalidate geometry")
    owners = [arm for arm, entry in calibration["arms"].items()
              if str(entry.get("wrist_camera_serial")) == serial]
    is_top = str(calibration.get("top_camera_serial")) == serial
    if len(owners) + int(is_top) != 1:
        raise ValueError("Camera serial must identify exactly one calibrated top or wrist camera")
    warnings, robot_source = [], None
    if not calibration.get("accepted_for_control", False):
        warnings.append("Calibration remains a candidate; this calculation does not accept it for control.")
    if is_top:
        transform, role = _top_transform(calibration, target_arm), "fixed_top"
    else:
        owner = owners[0]
        if owner not in config["arms"]:
            raise ValueError("Wrist owner is absent from configuration")
        interface = calibration["arms"][owner].get("interface")
        if interface is not None and interface != config["arms"][owner]:
            raise ValueError("Wrist arm interface differs from calibration identity")
        base_link, robot_source = _captured_link6(observation, config, owner)
        transform = _base_transform(calibration, target_arm, owner) @ base_link @ _rigid(
            calibration["arms"][owner]["T_link6_camera"])
        role = "wrist"
        warnings.append("Wrist image and feedback are not hardware synchronized; use settled captures.")
        if robot_source.get("error_code"):
            warnings.append("Captured robot state reports a fault; geometry is not motion authorization.")
    point = pixel_to_point(meta, np.load(camera_dir / "depth.npy", allow_pickle=False), pixel, radius, min_valid)
    point.update({"point_base_mm": transform_points(transform, point["point_camera_mm"]).tolist(),
                  "T_base_camera": _rigid(transform).tolist(), "units": "mm",
                  "frames": {"source": f"camera:{serial}", "target": f"base:{target_arm}",
                             "camera_role": role, "wrist_owner": owners[0] if owners else None},
                  "provenance": {"site_id": config["site_id"], "observation_dir": str(directory),
                                 "observation_captured_unix": observation.get("captured_unix"),
                                 "camera": camera, "frame_metadata": meta, "robot": robot_source,
                                 "calibration": {**calibration_source, "status": calibration.get("status"),
                                                 "accepted_for_control": calibration.get("accepted_for_control", False),
                                                 "board": calibration.get("board")},
                                 "config": config_source}, "warnings": warnings})
    return point


def run(request):
    """Evaluate one read-only geometry request; never opens cameras or CAN."""
    operation = request["operation"]
    if operation == "link6_target":
        if ("rotation_matrix" in request) == ("rpy_deg" in request):
            raise ValueError("Provide exactly one of rotation_matrix and rpy_deg")
        rotation = rpy_rotation(request["rpy_deg"]) if "rpy_deg" in request else _rotation(request["rotation_matrix"])
        xyz = link6_target(request["task_point_base_mm"], rotation, request["point_link6_mm"])
        target = {"xyz_mm": xyz.tolist()}
        if "rpy_deg" in request:
            target["rpy_deg"] = _vector(request["rpy_deg"], 3, "rpy_deg").tolist()
        result = {"operation": operation, "ok": True, "xyz_mm": xyz.tolist(), "units": "mm",
                  "target": target, "rotation_matrix": rotation.tolist(), "point_link6_mm": request["point_link6_mm"]}
        if "rpy_deg" not in request:
            result["note"] = "Matrix input determines XYZ; target omits RPY. Specify a consistent motion orientation when using this result."
    elif operation == "point_in_link6":
        if ("actual_T_base_link6" in request) == ("observation_dir" in request):
            raise ValueError("Provide exactly one of actual_T_base_link6 and observation_dir")
        provenance = None
        if "observation_dir" in request:
            directory = Path(request["observation_dir"]).resolve()
            observation = json.loads((directory / "observation.json").read_text())
            config, config_source = _json_input(request["config"])
            transform, robot = _captured_link6(observation, config, request["arm"])
            provenance = {"site_id": config.get("site_id"), "observation_dir": str(directory),
                          "robot": robot, "config": config_source}
        else:
            transform = _rigid(request["actual_T_base_link6"])
        base_point = request["point_base_mm"] if "point_base_mm" in request else request["base_point_mm"]
        point = point_in_link6(base_point, transform)
        result = {"operation": operation, "ok": True, "point_link6_mm": point.tolist(), "units": "mm",
                  "actual_T_base_link6": transform.tolist(), "provenance": provenance}
    elif operation == "observation_points":
        common = {key: request[key] for key in ("observation_dir", "camera", "target_arm", "calibration", "config")}
        points = []
        for index, item in enumerate(request["points"]):
            result = {"id": item.get("id", str(index))}
            try:
                result.update(observation_point(**common, pixel=item["pixel"],
                              radius=item.get("radius", request.get("radius", 2)),
                              min_valid=item.get("min_valid", request.get("min_valid", 1))))
                result["ok"] = True
            except (ValueError, KeyError, OSError, ImportError) as error:
                result.update({"ok": False, "error": f"{type(error).__name__}: {error}"})
            points.append(result)
        result = {"operation": operation, "ok": bool(points) and all(p["ok"] for p in points),
                  "points": points, "units": "mm"}
    else:
        raise ValueError(f"Unsupported geometry operation: {operation}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="JSON request path, or - for stdin")
    parser.add_argument("--output", help="Optional new JSON output path; never overwritten")
    args = parser.parse_args()
    request = json.load(sys.stdin) if args.request == "-" else json.loads(Path(args.request).read_text())
    result = run(request)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with Path(args.output).open("x") as stream:
            stream.write(serialized)
    sys.stdout.write(serialized)


if __name__ == "__main__":
    main()
