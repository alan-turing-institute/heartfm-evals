"""Segmentation metrics."""

from __future__ import annotations

import numpy as np

from heartfm_evals.constants import CLASS_NAMES, NUM_CLASSES


def dice_score(pred: np.ndarray, true: np.ndarray, class_idx: int) -> float:
    """Dice coefficient for a single class."""
    pred_c = pred == class_idx
    true_c = true == class_idx
    intersection = (pred_c & true_c).sum()
    return float(2 * intersection / (pred_c.sum() + true_c.sum() + 1e-8))


def macro_dice(
    pred: np.ndarray,
    true: np.ndarray,
    num_classes: int = NUM_CLASSES,
    exclude_bg: bool = True,
) -> float:
    """Macro-averaged Dice across foreground classes."""
    start = 1 if exclude_bg else 0
    return float(
        np.mean([dice_score(pred, true, c) for c in range(start, num_classes)])
    )


def per_sample_dice_score(pred: np.ndarray, true: np.ndarray, class_idx: int) -> float:
    """Per-sample Dice coefficient for a single class.

    Returns ``NaN`` when the class is absent in the ground truth for that sample.
    """
    pred_c = pred == class_idx
    true_c = true == class_idx
    if true_c.sum() == 0:
        return float("nan")
    intersection = (pred_c & true_c).sum()
    return float(2 * intersection / (pred_c.sum() + true_c.sum() + 1e-8))


def per_sample_macro_dice(
    pred: np.ndarray,
    true: np.ndarray,
    num_classes: int = NUM_CLASSES,
    exclude_bg: bool = True,
) -> float:
    """NaN-aware macro Dice across classes for a single sample."""
    start = 1 if exclude_bg else 0
    scores = [per_sample_dice_score(pred, true, c) for c in range(start, num_classes)]
    if np.all(np.isnan(scores)):
        return float("nan")
    return float(np.nanmean(scores))


def per_sample_dice_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> dict[str, float]:
    """Return per-class and macro Dice metrics for one sample."""
    metrics = {
        f"dice_{CLASS_NAMES.get(c, f'C{c}')}": per_sample_dice_score(pred, true, c)
        for c in range(num_classes)
    }
    metrics["macro_dice"] = per_sample_macro_dice(pred, true, num_classes=num_classes)
    return metrics
