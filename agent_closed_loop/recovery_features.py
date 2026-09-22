"""Causal, bounded evidence of repetitive motion and limited observed progress.

The inputs are chronological per-frame poses, *recorded* action poses and visual
features. They are not policy-predicted action chunks. In particular, a recorder
may copy state into action columns; these features then describe observed motion
and cannot establish command tracking errors or policy intent. Neither repetition
nor low observed progress is itself a label of failure or irrecoverability.

All output channels lie in [0, 1]. Scale arrays must be estimated from the normal
training split by the caller, never from the episode being scored. Position
differences remove offsets; scales only set units. Windows are 0.5, 1, 2 and 4
seconds. Repetition compares the two adjacent windows of that length, so the
longest channel needs eight seconds of history. No value at t uses data after t.
"""

from __future__ import annotations

import numpy as np


WINDOW_SECONDS = (0.5, 1.0, 2.0, 4.0)
GLOBAL_NAMES = (
    "tracking_pose_error", "action_speed", "state_speed", "visual_speed",
)
WINDOW_NAMES = (
    "history_fraction", "action_repeat_similarity", "action_repeat_match",
    "action_motion", "state_motion", "visual_motion",
    "state_path_efficiency", "visual_path_efficiency",
    "state_return_proximity", "visual_return_proximity",
    "repeat_without_progress",
)


def behavior_feature_names() -> list[str]:
    """Stable serialization order: four frame channels and 4 x 11 windows."""
    names = list(GLOBAL_NAMES)
    for seconds in WINDOW_SECONDS:
        suffix = f"{seconds:g}s".replace(".", "p")
        names.extend(f"{name}_{suffix}" for name in WINDOW_NAMES)
    return names


def _array(values, name: str) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite [frames, dimensions] matrix")
    return array


def _scale(values, dimensions: int, name: str) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (dimensions,) or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be a finite nonnegative vector of length {dimensions}")
    return np.maximum(array, 1e-6)


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    prefix = np.concatenate((np.zeros((1,) + values.shape[1:]), np.cumsum(values, axis=0)), axis=0)
    end = np.arange(1, len(values) + 1)
    return prefix[end] - prefix[np.maximum(0, end - window)]


def _bounded(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 0.0)
    return values / (1.0 + values)


def causal_behavior_features(
    states,
    actions,
    visuals,
    *,
    state_scale,
    action_scale,
    visual_scale,
    fps: float = 30,
) -> tuple[np.ndarray, list[str]]:
    """Return float32 [T, 48] channels with a fixed, documented feature order.

    Global channels are x/(1+x) of RMS command/state pose discrepancy and the
    current action/state/visual speed (standard deviations per second). The
    first frame has zero speed because no preceding observation is available.

    For each L-second action block, compare its delta sequence with the preceding
    block. Positive cosine and amplitude-sensitive match are multiplied by an
    activity gate e/(e+0.05), where e is joint RMS action speed. Both are zero
    until two complete blocks exist. A constant held command has zero evidence.
    No policy chunk boundary is assumed: blocks slide by one recorded frame.

    Progress channels cover the same 2L-second history: net displacement divided
    by traveled path (zero for zero path); RMS speeds; and 1/(1+net displacement).
    Net/path is geometric efficiency, not semantic task completion. The compound
    loop channel is active repetition * (1-state efficiency) * (1-visual
    efficiency), preventing steadily progressing repeated motion from being
    treated identically to a closed cycle. An explicit history fraction exposes
    partial early windows instead of filling them with future or padded frames.
    """
    states = _array(states, "states")
    actions = _array(actions, "actions")
    visuals = _array(visuals, "visuals")
    if states.shape != actions.shape or len(visuals) != len(states):
        raise ValueError("states/actions must have equal shapes and visuals the same frame count")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    state_scale = _scale(state_scale, states.shape[1], "state_scale")
    action_scale = _scale(action_scale, actions.shape[1], "action_scale")
    visual_scale = _scale(visual_scale, visuals.shape[1], "visual_scale")
    names = behavior_feature_names()
    if not len(states):
        return np.empty((0, len(names)), dtype=np.float32), names

    normalized = (states / state_scale, actions / action_scale, visuals / visual_scale)
    delta = [np.concatenate((np.zeros_like(x[:1]), np.diff(x, axis=0)), axis=0) for x in normalized]
    ds, da, dv = delta
    step_energy = [np.mean(d * d, axis=1) for d in delta]
    es, ea, ev = step_energy
    step_length = [np.sqrt(e) for e in step_energy]
    tracking = np.sqrt(np.mean(((actions - states) / state_scale) ** 2, axis=1))
    columns = [_bounded(tracking), _bounded(np.sqrt(ea) * fps),
               _bounded(np.sqrt(es) * fps), _bounded(np.sqrt(ev) * fps)]
    time = np.arange(len(states))

    for seconds in WINDOW_SECONDS:
        block = max(1, int(round(seconds * fps)))
        history = 2 * block
        count = np.minimum(time, history)
        denominator = np.maximum(count, 1)
        available = (time >= history).astype(np.float64)

        # Dot products align current pose deltas with deltas one block earlier.
        dot = np.zeros(len(states), dtype=np.float64)
        if block < len(states):
            dot[block:] = np.mean(da[block:] * da[:-block], axis=1)
        dot = _rolling_sum(dot, block)
        recent_energy = _rolling_sum(ea, block)
        previous_energy = np.zeros(len(states), dtype=np.float64)
        if block < len(states):
            previous_energy[block:] = recent_energy[:-block]
        energy = np.maximum(recent_energy + previous_energy, 0.0)
        cosine = np.clip(dot / np.maximum(np.sqrt(np.maximum(recent_energy * previous_energy, 0)), 1e-12), 0, 1)
        amplitude_match = np.exp(-2 * np.maximum(energy - 2 * dot, 0) / np.maximum(energy, 1e-12))
        speed = np.sqrt(energy / history) * fps
        activity = speed / (speed + 0.05)
        repeat = cosine * activity * available
        match = amplitude_match * activity * available

        motions = [_bounded(np.sqrt(np.maximum(_rolling_sum(e, history), 0) / denominator) * fps)
                   for e in step_energy]
        nets = [np.sqrt(np.mean((x - x[np.maximum(0, time - history)]) ** 2, axis=1))
                for x in (normalized[0], normalized[2])]
        paths = [_rolling_sum(step_length[i], history) for i in (0, 2)]
        efficiencies = [np.clip(net / np.maximum(path, 1e-12), 0, 1)
                        for net, path in zip(nets, paths)]
        loop = repeat * (1 - efficiencies[0]) * (1 - efficiencies[1])
        columns.extend((np.minimum(time / history, 1), repeat, match,
                        motions[1], motions[0], motions[2],
                        efficiencies[0], efficiencies[1],
                        1 / (1 + nets[0]), 1 / (1 + nets[1]), loop))

    features = np.stack(columns, axis=1)
    if not np.isfinite(features).all():
        raise ValueError("Feature computation overflowed; inspect input units/scales")
    return np.clip(features, 0, 1).astype(np.float32), names
