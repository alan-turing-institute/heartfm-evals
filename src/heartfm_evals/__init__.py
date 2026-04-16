"""
heartfm-evals: A modular package for evaluating foundation model performance on various clinical tasks involving cardiac MRI and CT images.
"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version(__name__)

__all__ = (
    "__version__",
    "backbones",
    "caching",
    "constants",
    "data",
    "decoders",
    "device",
    "features",
    "losses",
    "metrics",
    "reproducibility",
    "training",
    "visualization",
)
