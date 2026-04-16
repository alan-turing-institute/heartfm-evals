"""Dense linear probe for pixel-level segmentation using frozen DINOv3 features.

.. deprecated::
    This module is a backward-compatibility shim.  All code has been moved to
    dedicated modules:

    - Constants → :mod:`heartfm_evals.constants`
    - Backbone configs → :mod:`heartfm_evals.backbones`
    - Metrics → :mod:`heartfm_evals.metrics`
    - Visualization → :mod:`heartfm_evals.visualization`
    - Datasets → :mod:`heartfm_evals.data`
    - Feature extraction → :mod:`heartfm_evals.features`
    - Caching → :mod:`heartfm_evals.caching`
    - Losses → :mod:`heartfm_evals.losses`
    - Decoders → :mod:`heartfm_evals.decoders`
    - Training → :mod:`heartfm_evals.training`

    Import from the new modules directly.
"""

from __future__ import annotations

# ── Re-exports for backward compatibility ─────────────────────────────────────
from heartfm_evals.backbones import DINOV3_CONFIGS as MODEL_CONFIGS  # noqa: F401
from heartfm_evals.caching import (  # noqa: F401
    CachedFeatureDataset,
    cache_features,
)
from heartfm_evals.constants import (  # noqa: F401
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    PATCH_SIZE,
)
from heartfm_evals.data import ACDCSliceDataset  # noqa: F401
from heartfm_evals.decoders import DenseLinearProbe  # noqa: F401
from heartfm_evals.features import (  # noqa: F401
    extract_multilayer_features,
    preprocess_slice,
)
from heartfm_evals.losses import CombinedLoss, DiceLoss  # noqa: F401
from heartfm_evals.metrics import dice_score, macro_dice  # noqa: F401
from heartfm_evals.training import evaluate, train_one_epoch  # noqa: F401
from heartfm_evals.visualization import overlay_labels  # noqa: F401
