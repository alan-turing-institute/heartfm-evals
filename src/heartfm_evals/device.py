"""Device detection utility."""

from __future__ import annotations

import torch


def detect_device(override: str | None = None) -> torch.device:
    """Return the best available device (MPS → CUDA → CPU).

    Parameters
    ----------
    override:
        If given, return ``torch.device(override)`` directly.
    """
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
