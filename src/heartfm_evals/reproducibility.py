"""Reproducibility utilities for deterministic training runs."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# cuBLAS needs a fixed workspace size to give deterministic results on CUDA >= 10.2;
# without it ``use_deterministic_algorithms`` warns on every cuBLAS call. It is read
# when cuBLAS initialises, so it must be set before the first CUDA matmul.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy and PyTorch, and ask PyTorch for deterministic kernels.

    ``torch.manual_seed`` seeds the CPU, CUDA and MPS generators. CuDNN is put in
    deterministic mode (at a potential performance cost), ``CUBLAS_WORKSPACE_CONFIG``
    is set if the environment does not already set it, and PyTorch's
    deterministic-algorithms mode is enabled with ``warn_only=True``, so an op
    without a deterministic implementation warns instead of raising.

    This makes runs repeatable on CPU. On CUDA it does **not** make training
    bit-identical: the backward pass of ``F.interpolate`` in bilinear/trilinear
    mode, which every segmentation decoder uses, has no deterministic CUDA
    implementation. Seeded GPU runs share their initial weights, dropout masks and
    data order, but can still drift slightly from one run to the next.

    Call it at the start of a run *and* again immediately before building a
    trainable model, so that the model's initial weights depend only on ``seed``
    and not on whatever consumed the global RNG before (e.g. constructing a
    pretrained backbone, which randomly initialises it before loading weights).
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
