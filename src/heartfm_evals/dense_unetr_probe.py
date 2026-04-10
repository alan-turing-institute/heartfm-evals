"""3D UNetR decoder probes for DINOv3 and CineMA backbone evaluation.

.. deprecated::
    This module is a backward-compatibility shim.  All code has been moved to
    dedicated modules:

    - Constants → :mod:`heartfm_evals.constants`
    - Decoders → :mod:`heartfm_evals.decoders`
    - Feature extraction → :mod:`heartfm_evals.features`
    - Caching → :mod:`heartfm_evals.caching`
    - Losses → :mod:`heartfm_evals.losses`
    - Training → :mod:`heartfm_evals.training`

    Import from the new modules directly.
"""

from __future__ import annotations

# ── Re-exports for backward compatibility ─────────────────────────────────────
from heartfm_evals.caching import (  # noqa: F401
    CachedCinemaVolumeDataset,
    CachedVolumeDataset,
    cache_cinema_volume_features,
    cache_dino_volume_features,
    cache_sam_volume_features,
)
from heartfm_evals.constants import (  # noqa: F401
    CLASS_NAMES,
    GRID_SIZE,
    IMAGE_SIZE,
    NUM_CLASSES,
    PATCH_SIZE,
    SAX_TARGET_DEPTH,
)
from heartfm_evals.decoders import (  # noqa: F401
    CineMAUNetRDecoder,
    DINOv3UNetRDecoder,
)
from heartfm_evals.features import (  # noqa: F401
    _pad_volume_z,
    extract_cinema_volume_features,
    extract_dino_volume_features,
    extract_sam_volume_features,
    preprocess_slice,
)
from heartfm_evals.losses import MaskedVolumeLoss  # noqa: F401
from heartfm_evals.metrics import dice_score, macro_dice  # noqa: F401
from heartfm_evals.training import (  # noqa: F401
    _batch_to_device,
    evaluate_vol,
    train_one_epoch_vol,
)
