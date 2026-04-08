"""Segmentation metrics."""

from __future__ import annotations

import numpy as np

from heartfm_evals.constants import NUM_CLASSES


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
    return float(np.mean([dice_score(pred, true, c) for c in range(start, num_classes)]))
