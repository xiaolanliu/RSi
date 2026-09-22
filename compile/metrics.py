"""Evaluation helpers for CompILE outputs."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
from torch import Tensor

from .model import CompILEOutput


def boundary_positions(output: CompILEOutput, *, include_terminal: bool = False) -> Tensor:
    """Return one-based predicted boundary values.

    The model stores boundary slot ``j`` as the one-based boundary ``j + 1``.
    The terminal boundary is represented by the last segment and can be
    omitted when comparing internal boundaries to annotations.
    """

    positions = output.boundary_positions + 1
    return positions if include_terminal else positions[:, :-1]


def boundary_f1(
    predicted: Sequence[int],
    target: Sequence[int],
    *,
    tolerance: int = 0,
) -> float:
    """Set-based F1 used by the paper for boundary recovery."""

    remaining = list(int(value) for value in target)
    true_positive = 0
    for value in predicted:
        matches = [index for index, target_value in enumerate(remaining) if abs(int(value) - target_value) <= tolerance]
        if matches:
            true_positive += 1
            remaining.pop(matches[0])
    precision = true_positive / max(len(predicted), 1)
    recall = true_positive / max(len(target), 1)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def boundary_detection_metrics(
    predicted: Sequence[int],
    target: Sequence[int],
    *,
    tolerance: int,
) -> dict[str, Any]:
    """Boundary detection metrics with one-to-one temporal matching.

    Inputs are sorted internally. The two-pointer matcher gives the maximum
    number of matches for a fixed symmetric tolerance on a time line. Exact
    ordinal errors are reported separately when both inputs have equal size.
    """

    predicted_sorted = sorted(int(value) for value in predicted)
    target_sorted = sorted(int(value) for value in target)
    i = j = true_positive = 0
    matched_errors: list[int] = []
    while i < len(predicted_sorted) and j < len(target_sorted):
        error = predicted_sorted[i] - target_sorted[j]
        if abs(error) <= tolerance:
            true_positive += 1
            matched_errors.append(abs(error))
            i += 1
            j += 1
        elif predicted_sorted[i] < target_sorted[j] - tolerance:
            i += 1
        else:
            j += 1
    precision = true_positive / len(predicted_sorted) if predicted_sorted else 0.0
    recall = true_positive / len(target_sorted) if target_sorted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ordinal_errors = (
        [abs(prediction - truth) for prediction, truth in zip(predicted_sorted, target_sorted)]
        if len(predicted_sorted) == len(target_sorted)
        else []
    )
    signed_ordinal_errors = (
        [prediction - truth for prediction, truth in zip(predicted_sorted, target_sorted)]
        if len(predicted_sorted) == len(target_sorted)
        else []
    )
    return {
        "tolerance_frames": int(tolerance),
        "true_positives": true_positive,
        "false_positives": len(predicted_sorted) - true_positive,
        "false_negatives": len(target_sorted) - true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit_rate": true_positive / max(len(target_sorted), 1),
        "matched_absolute_errors": matched_errors,
        "ordinal_absolute_errors": ordinal_errors,
        "ordinal_signed_errors": signed_ordinal_errors,
    }


def ordinal_phase_metrics(
    predicted_boundaries: Sequence[int],
    target_boundaries: Sequence[int],
    *,
    length: int,
) -> dict[str, Any]:
    """Compare two ordered segmentations with the same phase ordering.

    Boundaries are zero-based starts of the following phase. Intervals are
    half-open and cover ``[0, length)``. This metric is appropriate for five
    ordered weak phases, but not for permutation-invariant latent code IDs.
    """

    if length <= 0:
        raise ValueError("length must be positive")
    predicted = [0, *map(int, predicted_boundaries), int(length)]
    target = [0, *map(int, target_boundaries), int(length)]
    if len(predicted) != len(target):
        raise ValueError("predicted and target must contain the same number of phases")
    for name, values in (("predicted", predicted), ("target", target)):
        if values != sorted(values) or values[0] != 0 or values[-1] != length:
            raise ValueError(f"{name} boundaries must be ordered within [0, length]")
    ious: list[float] = []
    overlaps: list[int] = []
    for index in range(len(predicted) - 1):
        pred_start, pred_end = predicted[index], predicted[index + 1]
        target_start, target_end = target[index], target[index + 1]
        intersection = max(0, min(pred_end, target_end) - max(pred_start, target_start))
        union = max(pred_end, target_end) - min(pred_start, target_start)
        ious.append(intersection / union if union else 1.0)
        overlaps.append(intersection)
    return {
        "per_phase_iou": ious,
        "mean_phase_iou": sum(ious) / len(ious),
        "frame_phase_accuracy": sum(overlaps) / length,
    }


def action_mae(
    output: CompILEOutput,
    actions: Tensor,
    valid_mask: Tensor,
    action_valid_mask: Tensor | None = None,
) -> float:
    """Mean absolute error of the mask-weighted policy prediction."""

    error = (output.action_prediction - actions).abs()
    frame_error = error.mean(dim=-1)
    if action_valid_mask is not None:
        valid_mask = valid_mask & action_valid_mask.bool()
    return float(frame_error.masked_select(valid_mask).mean().detach().cpu())


def code_usage(output: CompILEOutput) -> Tensor:
    """Average posterior code usage over the segment axis and batch."""

    return output.code_probs.detach().mean(dim=(0, 1))
