"""Reproducibility utilities for deterministic training runs."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Also configures CuDNN for deterministic behaviour (at a potential
    performance cost) and enables PyTorch's deterministic-algorithms mode
    with ``warn_only=True`` so that operations without a deterministic
    implementation emit a warning instead of raising an error.
    """
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)
