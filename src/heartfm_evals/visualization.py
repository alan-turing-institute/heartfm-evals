"""Visualization utilities for segmentation results."""

from __future__ import annotations

import numpy as np

from heartfm_evals.constants import CLASS_COLORS


def overlay_labels(
    label_map: np.ndarray,
    h: int,
    w: int,
    class_colors: dict | None = None,
) -> np.ndarray:
    """Create an RGBA overlay image from a label map of shape ``(h, w)``."""
    if class_colors is None:
        class_colors = CLASS_COLORS
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for cls, rgba in class_colors.items():
        mask = label_map == cls
        for ch in range(4):
            overlay[:, :, ch][mask] = rgba[ch]
    return overlay
