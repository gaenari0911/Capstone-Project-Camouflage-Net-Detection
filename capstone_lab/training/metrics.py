from __future__ import annotations

from collections.abc import Sequence
from typing import Any


IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


def _binary_iou(prediction: Sequence[int], target: Sequence[int]) -> float:
    if len(prediction) != len(target) or not prediction:
        raise ValueError("paired masks must be non-empty and have identical flattened shapes")
    pred = [bool(value) for value in prediction]
    truth = [bool(value) for value in target]
    intersection = sum(left and right for left, right in zip(pred, truth, strict=True))
    union = sum(left or right for left, right in zip(pred, truth, strict=True))
    return 1.0 if union == 0 else intersection / union


def _average_precision(matches: list[bool], target_count: int) -> float:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    true_positives = 0
    false_positives = 0
    recalls = [0.0]
    precisions = [1.0]
    for matched in matches:
        if matched:
            true_positives += 1
        else:
            false_positives += 1
        recalls.append(true_positives / target_count)
        precisions.append(true_positives / (true_positives + false_positives))
    recalls.append(1.0)
    precisions.append(0.0)
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    return sum(
        (recalls[index] - recalls[index - 1]) * precisions[index]
        for index in range(1, len(recalls))
        if recalls[index] != recalls[index - 1]
    )


def evaluate_paired_masks(
    predictions: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]],
    confidences: Sequence[float],
    *,
    operating_confidence: float = 0.25,
    operating_iou: float = 0.50,
) -> dict[str, Any]:
    """Evaluate a deterministic one-prediction/one-target-per-image mask fixture.

    This intentionally small adapter validates metric naming and arithmetic. It is not
    a replacement for the full Ultralytics matching/evaluation loop used by real runs.
    """

    if not (len(predictions) == len(targets) == len(confidences)) or not targets:
        raise ValueError("predictions, targets, and confidences must have the same positive length")
    if not 0.0 <= operating_confidence <= 1.0 or not 0.0 <= operating_iou <= 1.0:
        raise ValueError("operating point values must be in [0, 1]")
    rows = sorted(
        (
            {
                "confidence": float(confidence),
                "mask_iou": _binary_iou(prediction, target),
                "source_index": index,
            }
            for index, (prediction, target, confidence) in enumerate(
                zip(predictions, targets, confidences, strict=True)
            )
        ),
        key=lambda item: (-item["confidence"], item["source_index"]),
    )
    target_count = len(targets)
    aps = {
        f"{threshold:.2f}": _average_precision(
            [row["mask_iou"] >= threshold for row in rows], target_count
        )
        for threshold in IOU_THRESHOLDS
    }
    selected = [row for row in rows if row["confidence"] >= operating_confidence]
    true_positives = sum(row["mask_iou"] >= operating_iou for row in selected)
    false_positives = len(selected) - true_positives
    false_negatives = target_count - true_positives
    precision = true_positives / len(selected) if selected else 0.0
    recall = true_positives / target_count
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "metric_scope": "paired_binary_mask_fixture",
        "mask": {
            "map50_95": sum(aps.values()) / len(aps),
            "map50": aps["0.50"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "operating_point": {
                "confidence": operating_confidence,
                "mask_iou": operating_iou,
            },
            "counts": {
                "tp": true_positives,
                "fp": false_positives,
                "fn": false_negatives,
                "targets": target_count,
            },
            "ap_by_iou": aps,
            "pair_ious_by_confidence": rows,
        },
        "box": None,
    }
