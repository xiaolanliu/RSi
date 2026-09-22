"""Replay a complete recorded episode, export scores and a standalone HTML plot."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def read_episode(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    inputs = ([("normalized_input", 76)] if "normalized_input" in arrays
              else [("states", 14), ("visual", 48)])
    length = None
    for key, dim in inputs:
        value = arrays.get(key)
        if value is None or value.ndim != 2 or value.shape[1] != dim:
            raise ValueError(f"Expected {key}[T,{dim}] in {path}")
        if length is not None and len(value) != length:
            raise ValueError("State and visual lengths differ")
        length = len(value)
        if length < 2 or not np.isfinite(value).all():
            raise ValueError("Episode must contain at least two finite frames")
    times = arrays.get("timestamps", np.arange(length, dtype=np.float64) / 30)
    if times.shape != (length,) or not np.isfinite(times).all():
        raise ValueError("Expected finite timestamps[T]")
    if not np.allclose(np.diff(times), 1 / 30, atol=1e-4, rtol=0):
        raise ValueError("Model requires consecutive observations at 30 FPS; do not skip frames")
    arrays["timestamps"] = times
    return arrays


def online(arrays, bundle, device="cpu", state_layout="joints12_grippers2", seed=0):
    from agent_closed_loop.monitor import OnlineMonitor
    monitor = OnlineMonitor(bundle, device=device, episode_seed=seed)
    if "normalized_input" in arrays:
        outputs = [monitor.step_normalized(x, debug=True) for x in arrays["normalized_input"]]
    else:
        outputs = [monitor.step(s, v, state_layout=state_layout, debug=True)
                   for s, v in zip(arrays["states"], arrays["visual"])]
    result = {key: np.asarray([row[key] for row in outputs]) for key in (
        "phase_probs", "confirmed_phase", "accepted_phase", "alarm",
        "risk_score", "frame_risk_score", "risk_percentile")}
    result["evidence"] = np.asarray([row["debug"]["evidence"] for row in outputs])
    result["threshold"] = np.asarray(monitor.threshold)
    return result


def offline(arrays, checkpoint, device="cpu", state_layout="joints12_grippers2"):
    from compile import CompILE, CompILEConfig
    from compile.subtasks import monotonic_boundary_projection
    from agent_closed_loop.action_units import packed_state
    if "states" not in arrays:
        raise ValueError("Offline model needs raw states and visual, not normalized_input")
    # Match the original offline report's precision, separately from the online encoder.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = CompILE(CompILEConfig(**ck["config"])).to(device).eval()
    model.load_state_dict(ck["model"], strict=True)
    states = np.stack([packed_state(s, state_layout) for s in arrays["states"]])
    actions = np.zeros_like(states)
    actions[:-1] = np.diff(states, axis=0)
    valid = torch.ones((1, len(states)), dtype=torch.bool, device=device)
    action_valid = valid.clone()
    action_valid[:, -1] = False
    batch = dict(states=torch.as_tensor(states, device=device)[None],
                 actions=torch.as_tensor(actions, device=device)[None],
                 visual_features=torch.as_tensor(arrays["visual"], dtype=torch.float32, device=device)[None],
                 valid_mask=valid, action_valid_mask=action_valid)
    with torch.inference_mode():
        result = model(batch, sample_latents=False)
    probabilities = result.segment_masks[0].T.cpu().numpy()
    boundaries = monotonic_boundary_projection(result.boundary_probs[0], len(states))
    return dict(phase_probs=probabilities, boundaries=np.asarray(boundaries),
                phase_argmax=probabilities.argmax(-1) + 1)


def check(result, expected_path, device="cpu"):
    with np.load(expected_path, allow_pickle=False) as expected:
        for key in expected.files:
            if key.endswith("_cpu"):
                continue  # Independent original-source CPU reference, if supplied.
            if key not in result:
                raise AssertionError(f"Missing output {key}")
            reference_key = key + "_cpu" if str(device).startswith("cpu") and key + "_cpu" in expected.files else key
            if key in ("alarm", "confirmed_phase", "accepted_phase", "boundaries"):
                np.testing.assert_array_equal(result[key], expected[reference_key], err_msg=key)
            else:
                np.testing.assert_allclose(result[key], expected[reference_key], atol=8e-5, rtol=3e-4, err_msg=key)


def write_outputs(result, times, folder, *, title):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "predictions.npz", timestamps=times, **result)
    is_online = "alarm" in result
    fields = ["frame", "time_seconds", *[f"phase_{i}" for i in range(1, 6)]]
    if is_online:
        fields += ["confirmed_phase", "accepted_phase", "alarm", "risk_score", "frame_risk_score",
                   "risk_percentile", "projection_distance", "action_unit_repetition", "line_energy"]
    with (folder / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for i, t in enumerate(times):
            row = [i, float(t), *result["phase_probs"][i].tolist()]
            if is_online:
                row += [int(result[k][i]) for k in ("confirmed_phase", "accepted_phase", "alarm")]
                row += [float(result[k][i]) for k in ("risk_score", "frame_risk_score", "risk_percentile")]
                row += result["evidence"][i].tolist()
            writer.writerow(row)
    alarms = np.flatnonzero(result.get("alarm", []))
    summary = dict(episode=title, mode="online" if is_online else "offline", frames=len(times))
    if is_online:
        summary.update(first_alarm_frame=int(alarms[0]) if len(alarms) else None,
                       first_alarm_seconds=float(times[alarms[0]]) if len(alarms) else None,
                       alarm_frames=int(len(alarms)), threshold=float(result["threshold"]))
    else:
        summary["boundaries_one_based"] = result["boundaries"].tolist()
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    payload = dict(summary=summary, times=times.tolist(), **{k:v.tolist() for k,v in result.items()})
    # Embedded data: usable without server/CDN or access to the original machine.
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    template = (Path(__file__).with_name("report.html")).read_text()
    (folder / "index.html").write_text(template.replace("__PAYLOAD__", encoded))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["online", "offline"], default="online")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--state-layout", choices=["joints12_grippers2", "left7_right7"], default="joints12_grippers2")
    parser.add_argument("--expected", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = args.checkpoint or Path("models/v11/fold_all.pt" if args.mode == "online"
                                        else "models/offline_2000/checkpoint_epoch_2000.pt")
    values = read_episode(args.input)
    runner = online if args.mode == "online" else offline
    result = runner(values, checkpoint, args.device, args.state_layout)
    if args.expected:
        check(result, args.expected, args.device)
    summary = write_outputs(result, values["timestamps"], args.output, title=args.input.stem)
    summary["reference_check"] = "passed" if args.expected else "not requested"
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
