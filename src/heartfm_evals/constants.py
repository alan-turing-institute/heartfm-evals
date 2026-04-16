"""Shared constants for the heartfm-evals package."""

from __future__ import annotations

import torchvision.transforms as T

# ── ImageNet normalisation ────────────────────────────────────────────────────
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
imagenet_normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# ── Patch / image geometry ────────────────────────────────────────────────────
PATCH_SIZE: int = 16
IMAGE_SIZE: int = 192
GRID_SIZE: int = IMAGE_SIZE // PATCH_SIZE  # 12

# ── ACDC segmentation classes (shared across ACDC / M&M / M&M2) ──────────────
NUM_CLASSES: int = 4
CLASS_NAMES: dict[int, str] = {0: "BG", 1: "RV", 2: "MYO", 3: "LV"}
CLASS_COLORS: dict[int, tuple[float, float, float, float]] = {
    0: (0, 0, 0, 0),
    1: (0, 0, 1, 0.4),
    2: (0, 1, 0, 0.4),
    3: (1, 0, 0, 0.4),
}

# ── 3-D volume constants ──────────────────────────────────────────────────────
SAX_TARGET_DEPTH: int = 16
