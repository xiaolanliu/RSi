#!/usr/bin/env python3
"""Offline ChArUco pose detection from existing RGBD observations; no hardware I/O.

T_camera_board maps board coordinates in millimetres into camera coordinates.
Residuals in pixels are undistorted pixel equivalents: normalized ray residuals
scaled by fx/fy. This deliberately avoids interpreting inverse Brown coefficients
as OpenCV's forward Brown model.

Run with ``python -m agilex_control.calibration_board --observation DIR
--board BOARD.json --output NEW_RESULT.json``. RGB images and frame metadata are
required; aligned depth is optional. Importing this module never opens hardware.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from .geometry import pixel_rays  # Historical import remains supported.


def depth_quality(depth_path, pixels, rays, meta, transform):
    if not depth_path.exists():
        return {"available": False}
    depth = np.load(depth_path, allow_pickle=False)
    if depth.shape != (meta["height"], meta["width"]):
        return {"available": False, "error": "Aligned depth dimensions disagree with RGB frame"}
    samples = []
    selected = []
    for index, (u, v) in enumerate(pixels):
        x, y = int(round(u)), int(round(v))
        patch = depth[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        valid = patch[(patch > 0) & np.isfinite(patch)]
        if valid.size >= 8:
            samples.append(float(np.median(valid)) * meta["depth_scale_m"] * 1000.0)
            selected.append(index)
    result = {"available": True, "valid_corner_count": len(samples)}
    if len(samples) < 6:
        return result
    points = rays[selected] * np.array(samples)[:, None]
    normal = transform[:3, 2]
    distances = (points - transform[:3, 3]) @ normal
    centered = points - np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    fitted = vh[-1]
    residual = centered @ fitted
    result.update({
        "pnp_plane_median_signed_distance_mm": float(np.median(distances)),
        "pnp_plane_median_absolute_distance_mm": float(np.median(np.abs(distances))),
        "fitted_plane_rmse_mm": float(np.sqrt(np.mean(residual ** 2))),
        "fitted_plane_normal_disagreement_deg": float(np.degrees(np.arccos(
            np.clip(abs(float(fitted @ normal)), 0, 1)))),
        "median_depth_mm": float(np.median(samples)),
        "note": "Aligned depth sampled at detected chessboard corners; validation only, not used to fit pose."
    })
    return result


def detect_camera(camera_dir, board, detector, minimum_corners=8):
    meta = json.loads((camera_dir / "metadata.json").read_text())
    image = cv2.imread(str(camera_dir / "rgb.jpg"))
    if image is None:
        raise ValueError("Cannot read RGB image: " + str(camera_dir))
    if image.shape[:2] != (meta["height"], meta["width"]):
        raise ValueError("Image dimensions disagree with frame intrinsics")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _, marker_ids = detector.detectBoard(gray)
    result = {"serial": meta["serial"], "metadata": meta,
              "marker_ids": [] if marker_ids is None else marker_ids.ravel().tolist(),
              "corner_ids": [] if ids is None else ids.ravel().tolist(),
              "corners_px": [] if corners is None else corners.reshape(-1, 2).tolist(),
              "eligible": False}
    count = 0 if ids is None else len(ids)
    result["corner_count"] = count
    result["corner_completeness"] = count / len(board.getChessboardCorners())
    if count < minimum_corners:
        result["rejection_reason"] = "fewer_than_minimum_corners"
        return result
    if board.checkCharucoCornersCollinear(ids):
        result["rejection_reason"] = "collinear_corners"
        return result
    object_points, image_points = board.matchImagePoints(corners, ids)
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    rays = pixel_rays(meta, pixels)
    normalized = np.ascontiguousarray(rays[:, :2])
    focal = np.array([meta["fx"], meta["fy"]])
    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points, normalized, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for rvec, tvec in zip(rvecs, tvecs):
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points, normalized, np.eye(3), None, rvec.copy(), tvec.copy())
        rotation, _ = cv2.Rodrigues(rvec)
        camera_points = object_points @ rotation.T + tvec.reshape(1, 3)
        if not np.all(camera_points[:, 2] > 0):
            continue
        errors = (camera_points[:, :2] / camera_points[:, 2:3] - normalized) * focal
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, tvec.ravel()
        candidates.append({"T_camera_board": transform.tolist(),
                           "reprojection_rmse_px": float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))),
                           "reprojection_max_px": float(np.max(np.linalg.norm(errors, axis=1)))})
    candidates.sort(key=lambda candidate: candidate["reprojection_rmse_px"])
    result["positive_depth_pnp_candidates"] = candidates
    if not candidates:
        result["rejection_reason"] = "no_positive_depth_solution"
        return result
    best = candidates[0]
    transform = np.array(best["T_camera_board"])
    ambiguous = False
    if len(candidates) > 1:
        other = np.array(candidates[1]["T_camera_board"])
        cosine = np.clip((np.trace(transform[:3, :3].T @ other[:3, :3]) - 1) / 2, -1, 1)
        separation = float(np.degrees(np.arccos(cosine)))
        result["second_solution_rotation_separation_deg"] = separation
        ambiguous = separation > 5 and candidates[1]["reprojection_rmse_px"] < max(
            best["reprojection_rmse_px"] * 1.25, best["reprojection_rmse_px"] + 0.1)
    result.update(best)
    result["normalized_rays_xy"] = normalized.tolist()
    result["object_points_mm"] = object_points.tolist()
    result["pnp_ambiguous"] = ambiguous
    result["image_hull_fraction"] = float(cv2.contourArea(cv2.convexHull(
        pixels.astype(np.float32)))) / (meta["width"] * meta["height"])
    result["depth_quality"] = depth_quality(camera_dir / "depth.npy", pixels, rays, meta, transform)
    result["eligible"] = not ambiguous and best["reprojection_rmse_px"] <= 1.5
    if not result["eligible"]:
        result["rejection_reason"] = "ambiguous_planar_pose" if ambiguous else "high_reprojection_error"
    return result


def detect_observation(observation, board_config, minimum_corners=8):
    if minimum_corners < 6:
        raise ValueError("minimum_corners must be at least 6")
    observation = Path(observation)
    spec = json.loads(Path(board_config).read_text())
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    marker_ids = np.asarray(spec["marker_ids"])
    expected_markers = spec["squares_x"] * spec["squares_y"] // 2
    if (marker_ids.ndim != 1 or len(marker_ids) != expected_markers
            or not np.issubdtype(marker_ids.dtype, np.integer)):
        raise ValueError(f"marker_ids must contain {expected_markers} integer IDs")
    if (np.any(marker_ids < 0) or np.any(marker_ids >= len(dictionary.bytesList))
            or len(np.unique(marker_ids)) != len(marker_ids)):
        raise ValueError("marker_ids must be unique and within the selected dictionary")
    board = cv2.aruco.CharucoBoard((spec["squares_x"], spec["squares_y"]),
                                  spec["square_length_mm"], spec["marker_length_mm"], dictionary,
                                  marker_ids.astype(np.int32))
    board.setLegacyPattern(spec.get("legacy_pattern", False))
    if board.getIds().ravel().tolist() != spec["marker_ids"]:
        raise ValueError("Board marker IDs differ from configuration")
    params = cv2.aruco.DetectorParameters()
    params.markerBorderBits = spec.get("marker_border_bits", 1)
    detector = cv2.aruco.CharucoDetector(board, detectorParams=params)
    cameras = {}
    for camera_dir in sorted(observation.iterdir()):
        if camera_dir.is_dir() and (camera_dir / "metadata.json").is_file():
            try:
                cameras[camera_dir.name] = detect_camera(camera_dir, board, detector, minimum_corners)
            except Exception as error:
                cameras[camera_dir.name] = {"eligible": False, "error": f"{type(error).__name__}: {error}"}
    return {"schema_version": 1, "observation": str(observation.resolve()),
            "board_config": spec, "translation_unit": "mm", "opencv_version": cv2.__version__,
            "projection_method": "Model-aware deprojection to normalized rays; IPPE candidates plus LM refinement",
            "reprojection_error_unit": "undistorted pixel equivalent (normalized ray residual times fx/fy)",
            "minimum_corners": minimum_corners, "cameras": cameras}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--board", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-corners", type=int, default=8)
    args = parser.parse_args()
    if args.minimum_corners < 6:
        parser.error("--minimum-corners must be at least 6")
    result = detect_observation(args.observation, args.board, args.minimum_corners)
    with open(args.output, "x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({name: {key: value.get(key) for key in
                           ("corner_count", "eligible", "reprojection_rmse_px", "rejection_reason", "error")}
                      for name, value in result["cameras"].items()}))


if __name__ == "__main__":
    main()
