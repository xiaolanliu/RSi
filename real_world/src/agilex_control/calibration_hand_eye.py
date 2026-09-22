#!/usr/bin/env python3
"""Offline fixed-board hand-eye analysis; no device or network access.

Run with ``python -m agilex_control.calibration_hand_eye --input ... --output ...``.
The JSON schema uses T_A_B to map frame B into frame A, with millimetre
translations. Each input contains one physical wrist camera and its robot arm.
Results are candidates, never automatically accepted for control.
Optional ``known_T_link6_camera`` reuses a previously validated, unchanged wrist
mount to update board/top-camera transforms without refitting the hand-eye.
It still requires three training observations; repeated static captures may
reduce random noise but cannot establish absolute accuracy or board scale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def matrix(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite 4x4 matrix")
    if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f"{name}: invalid homogeneous last row")
    r = a[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or abs(np.linalg.det(r) - 1) > 1e-5:
        raise ValueError(f"{name}: rotation is not in SO(3)")
    return a


def transform(r, t):
    a = np.eye(4)
    a[:3, :3], a[:3, 3] = r, np.asarray(t).reshape(3)
    return a


def inverse(a):
    r = a[:3, :3].T
    return transform(r, -r @ a[:3, 3])


def scaled(a, factor):
    a = a.copy()
    a[:3, 3] *= factor
    return a


def mean_transform(values):
    return transform(
        Rotation.from_matrix(np.stack([a[:3, :3] for a in values])).mean().as_matrix(),
        np.mean([a[:3, 3] for a in values], axis=0),
    )


def residual_summary(values, reference, ids):
    if not values:
        return None
    translation = np.array([np.linalg.norm(a[:3, 3] - reference[:3, 3]) for a in values])
    angle = np.array([
        np.rad2deg(Rotation.from_matrix(reference[:3, :3].T @ a[:3, :3]).magnitude())
        for a in values
    ])
    def stats(x):
        return {"rms": float(np.sqrt(np.mean(x**2))), "median": float(np.median(x)),
                "p95": float(np.percentile(x, 95)), "max": float(np.max(x))}
    return {"count": len(values), "translation_mm": stats(translation),
            "rotation_deg": stats(angle), "samples": [
                {"id": sample_id, "translation_mm": float(t), "rotation_deg": float(r)}
                for sample_id, t, r in zip(ids, translation, angle)
            ]}


def split_report(values, samples):
    train = [a for a, s in zip(values, samples) if s["split"] == "train"]
    if not train:
        return None
    reference = mean_transform(train)
    report = {"training_mean_transform": reference.tolist()}
    for split in ("train", "holdout"):
        chosen = [(a, s["id"]) for a, s in zip(values, samples) if s["split"] == split]
        report[split] = residual_summary([a for a, _ in chosen], reference,
                                         [sample_id for _, sample_id in chosen])
    return report


def rotation_excitation(samples):
    g = [s["G"] for s in samples]
    # Relative rotations expressed in a single robot-base frame.
    rotvecs = np.array([Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).as_rotvec()
                       for a, b in combinations(g, 2)])
    singular = np.linalg.svd(rotvecs, compute_uv=False)
    degrees = np.rad2deg(np.linalg.norm(rotvecs, axis=1))
    rank = int(np.linalg.matrix_rank(rotvecs, tol=1e-5))
    ratio = float(singular[1] / singular[0]) if singular[0] > 1e-12 else 0.0
    warnings = []
    if rank < 2:
        warnings.append("Fewer than two independent rotation axes: hand-eye is degenerate.")
    elif ratio < 0.15:
        warnings.append("Rotation axes are weakly separated; more second-axis excitation is advisable.")
    if degrees.max() < 15:
        warnings.append("Total rotation span is below 15 degrees; noise amplification is likely.")
    return {"pair_count": len(rotvecs), "rotation_vector_singular_values_rad": singular.tolist(),
            "numerical_rank": rank, "second_to_first_singular_ratio": ratio,
            "max_relative_rotation_deg": float(degrees.max()),
            "robot_translation_span_xyz_mm": np.ptp([a[:3, 3] for a in g], axis=0).tolist(),
            "warnings": warnings,
            "threshold_note": "15 degrees and 0.15 are sampling heuristics, not accuracy guarantees."}


def solve_translation_and_scale(samples, camera_to_link_rotation):
    """Fit R_g t_x + t_g + s R_g R_x t_c = t_b; rotations remain fixed.

    s corrects the board dimensions used for input PnP, so expected s is near 1.
    The scale column is projected off the six translation nuisance columns to
    explicitly expose scale degeneracy (e.g. fixed link6 position).
    """
    nuisance, scale_column, rhs = [], [], []
    for sample in samples:
        g, c = sample["G"], sample["C"]
        nuisance.append(np.hstack([g[:3, :3], -np.eye(3)]))
        scale_column.append(g[:3, :3] @ camera_to_link_rotation @ c[:3, 3])
        rhs.append(-g[:3, 3])
    nuisance, v, rhs = np.vstack(nuisance), np.concatenate(scale_column), np.concatenate(rhs)
    design = np.column_stack([nuisance, v])
    solution, _, rank, singular = np.linalg.lstsq(design, rhs, rcond=None)
    projected = v - nuisance @ np.linalg.lstsq(nuisance, v, rcond=None)[0]
    information = float(projected @ projected)
    norms = np.linalg.norm(design, axis=0)
    normalized_singular = np.linalg.svd(design / np.maximum(norms, 1e-15), compute_uv=False)
    residual = design @ solution - rhs
    dof = max(len(rhs) - int(rank), 1)
    conditional_sigma = float(np.sqrt((residual @ residual / dof) / information)) if information > 1e-12 else None
    result = {
        "rank": int(rank), "required_rank": 7,
        "design_singular_values_mixed_units": singular.tolist(),
        "column_normalized_singular_values": normalized_singular.tolist(),
        "column_normalized_condition": float(normalized_singular[0] / normalized_singular[-1])
            if normalized_singular[-1] > 1e-15 else None,
        "scale_observable_component_norm_mm": float(np.sqrt(information)),
        "conditional_scale_standard_error": conditional_sigma,
        "standard_error_note": "Conditional least-squares noise estimate with rotations held fixed; excludes FK, intrinsics, board warp and systematic errors.",
        "equation_residual_rms_mm": float(np.sqrt(np.mean(residual**2))),
        "identifiable": bool(rank == 7 and information > 1e-8),
    }
    if result["identifiable"]:
        result.update({"scale_correction": float(solution[6]),
                       "T_link6_camera": transform(camera_to_link_rotation, solution[:3]).tolist(),
                       "T_base_board_translation_mm": solution[3:6].tolist()})
        if solution[6] <= 0:
            result["identifiable"] = False
            result["failure"] = "Non-positive scale: inconsistent data or inadequate excitation."
    else:
        result["failure"] = "Joint translation/scale fit is rank deficient; no scale estimate is usable."
    return result


def evaluate(samples, x, scale):
    boards = [s["G"] @ x @ scaled(s["C"], scale) for s in samples]
    result = {"scale_correction": float(scale), "T_link6_camera": x.tolist(),
              "base_board_consistency": split_report(boards, samples)}
    top_samples, top_transforms, top_boards = [], [], []
    for sample, board in zip(samples, boards):
        if sample.get("top") is not None:
            top_samples.append(sample)
            top_boards.append(scaled(sample["top"], scale))
            top_transforms.append(board @ inverse(top_boards[-1]))
    if top_samples:
        result["base_top_camera_consistency"] = split_report(top_transforms, top_samples)
        result["top_camera_board_stationarity"] = split_report(top_boards, top_samples)
    return result


def parse_samples(data):
    samples, ids = [], set()
    for index, raw in enumerate(data["samples"]):
        sample_id = str(raw.get("id", index))
        if sample_id in ids:
            raise ValueError(f"Duplicate sample id {sample_id}")
        ids.add(sample_id)
        split = raw.get("split", "train")
        if split not in ("train", "holdout"):
            raise ValueError(f"{sample_id}: split must be train or holdout")
        samples.append({"id": sample_id, "split": split,
                        "G": matrix(raw["T_base_link6"], sample_id + ":T_base_link6"),
                        "C": matrix(raw["T_camera_board"], sample_id + ":T_camera_board"),
                        "top": matrix(raw["top_T_camera_board"], sample_id + ":top_T_camera_board")
                            if raw.get("top_T_camera_board") is not None else None})
    if sum(s["split"] == "train" for s in samples) < 3:
        raise ValueError("At least three training poses are required; at least ten are recommended.")
    return samples


def solve(data):
    import cv2
    samples = parse_samples(data)
    train = [s for s in samples if s["split"] == "train"]
    known_x = matrix(data["known_T_link6_camera"], "known_T_link6_camera") \
        if "known_T_link6_camera" in data else None
    excitation = rotation_excitation(train) if known_x is None else {
        "required": False, "warnings": [],
        "reason": "Known hand-eye transform is reused unchanged; no rotational fit is performed."
    }
    warnings = list(excitation["warnings"])
    if len(train) < 10 and known_x is None:
        warnings.append("Fewer than ten training poses; sample coverage requires careful review.")
    if not any(s["split"] == "holdout" for s in samples):
        warnings.append("No held-out poses: fit residuals do not provide independent validation.")
    if not data.get("board", {}).get("physical_dimensions_verified", False):
        warnings.append(
            "Input board dimensions are approximate. Estimated scale is a robot-model-based inference, not a physical measurement."
            if known_x is None else
            "Input board dimensions are unverified. Scale correction 1 preserves the supplied PnP dimensions; the resulting transforms remain scale-dependent candidates."
        )
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "metadata": {k: v for k, v in data.items() if k != "samples"},
              "units": "millimetres; rotation residuals in degrees",
              "opencv_version": cv2.__version__, "accepted_for_control": False,
              "rotation_excitation": excitation, "warnings": warnings, "methods": {}}
    if known_x is not None:
        report["known_hand_eye"] = evaluate(samples, known_x, 1.0)
        report["interpretation"] = (
            "The supplied T_link6_camera is reused unchanged; no hand-eye or board-scale fitting occurs. "
            "Scale correction 1 means using the board dimensions already assumed by input PnP, not claiming they are exact. "
            "Training means define board/top-camera references and held-out observations never enter those means. "
            "Repeated static captures may reduce random noise; low residuals do not validate scale or absolute accuracy. "
            "This mode assumes the known transform still belongs to this physical wrist/arm and its unchanged mount."
        )
        return report
    if excitation["numerical_rank"] < 2:
        report["failure"] = "Degenerate rotations; no calibration candidate was produced."
        return report
    rg = [s["G"][:3, :3] for s in train]
    tg = [s["G"][:3, 3:4] for s in train]
    rc = [s["C"][:3, :3] for s in train]
    tc = [s["C"][:3, 3:4] for s in train]
    for name in ("PARK", "TSAI", "HORAUD"):
        try:
            r, t = cv2.calibrateHandEye(rg, tg, rc, tc, method=getattr(cv2, "CALIB_HAND_EYE_" + name))
            x = matrix(transform(r, t), name + ":result")
            method = {"fixed_board_scale": evaluate(samples, x, 1.0)}
            joint = solve_translation_and_scale(train, r)
            method["joint_scale_diagnostics"] = joint
            if joint["identifiable"]:
                s = joint["scale_correction"]
                method["estimated_board_scale"] = evaluate(samples, np.asarray(joint["T_link6_camera"]), s)
                method["estimated_board_dimensions_mm"] = {
                    key: float(data["board"][key]) * s
                    for key in ("square_length_mm", "marker_length_mm", "scale_bar_mm")
                    if key in data.get("board", {})
                }
            report["methods"][name] = method
        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            report["methods"][name] = {"failure": str(exc)}
    report["interpretation"] = (
        "Training means define the references; held-out samples never enter fitting. "
        "Compare methods and held-out residuals before selecting any transform. "
        "Low residuals measure internal consistency, not robot absolute accuracy. "
        "Uniform scale estimation cannot detect or repair anisotropic printing, warped paper, "
        "wrong wrist/arm association, incorrect distortion model, or moving board."
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = args.input.read_bytes()
    report = solve(json.loads(content))
    report["input_path"] = str(args.input.resolve())
    report["input_sha256"] = hashlib.sha256(content).hexdigest()
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(serialized)
    print(json.dumps({"output": str(args.output.resolve()), "methods": list(report["methods"]),
                      "accepted_for_control": False, "warnings": report["warnings"]}))


if __name__ == "__main__":
    main()
