"""Logistic-regression classification probe for patient-level pathology prediction.

Supports four frozen backbones with a shared downstream protocol:

    1. **DINOv3**: extract CLS token or GAP of patch tokens per 2D slice →
       one ``(embed_dim,)`` embedding per slice.
    2. **CineMA**: run ``feature_forward()`` on the 3D SAX volume and
       extract the CLS token or global-mean-pool all spatial tokens → one
       ``(embed_dim,)`` embedding per cardiac-phase volume.
    3. **SAM**: run ``get_image_embeddings()`` on each 2D slice (converted to
       RGB via ``SamImageProcessor``), global-average-pool the spatial feature
       map ``(C, h, w)`` → one ``(C,)`` embedding per slice (no CLS token).
    4. **SAM2**: run the Hiera vision encoder on each 2D slice, global-average-
       pool the last-stage hidden state → one ``(embed_dim,)`` embedding per slice.

DINOv3, SAM, and SAM2 produce one embedding per 2D slice; CineMA produces one
embedding per 3D volume.  Patient-level features are built identically:
mean-pool ED embeddings, mean-pool ES embeddings, concatenate →
``(2 × embed_dim,)`` vector → sklearn LogisticRegression with L2 penalty
and C-sweep via stratified CV.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from heartfm_evals.dense_linear_probe import preprocess_slice

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_PATHOLOGY_CLASSES: dict[str, dict[str, int]] = {
    "acdc": {"NOR": 0, "DCM": 1, "HCM": 2, "MINF": 3, "RV": 4},
    "mnm": {"NOR": 0, "DCM": 1, "HCM": 2, "ARV": 3, "HHD": 4},
    "mnm2": {"NOR": 0, "HCM": 1, "ARR": 2, "CIA": 3, "FALL": 4, "LV": 5},
}


def get_pathology_classes(dataset: str) -> dict[str, int]:
    """Return the pathology class mapping for a dataset."""
    return DATASET_PATHOLOGY_CLASSES[dataset]


# Backward-compatible aliases (ACDC defaults)
PATHOLOGY_CLASSES = DATASET_PATHOLOGY_CLASSES["acdc"]
PATHOLOGY_NAMES = {v: k for k, v in PATHOLOGY_CLASSES.items()}
PATHOLOGY_NAMES_LONG = {
    "NOR": "Normal",
    "DCM": "Dilated Cardiomyopathy",
    "HCM": "Hypertrophic Cardiomyopathy",
    "MINF": "Myocardial Infarction",
    "RV": "Abnormal Right Ventricle",
}
NUM_PATHOLOGIES = len(PATHOLOGY_CLASSES)

# Same sweep range as dinov3 eval/log_regression.py
C_POWER_RANGE = torch.linspace(-6, 5, 45)


# ── Feature Extraction ─────────────────────────────────────────────────────────
@torch.inference_mode()
def extract_cls_features(
    backbone: nn.Module,
    cinema_dataset,
    device: torch.device | None = None,
    pooling: str = "cls",
) -> dict[str, dict]:
    """Extract DINOv3 features from all slices in a CineMA dataset.

    Groups features by patient ID, keeping ED and ES frames separate.

    Args:
        backbone: Frozen DINOv3 backbone in eval mode.
        cinema_dataset: CineMA EndDiastoleEndSystoleDataset.
        device: Device for inference.
        pooling: ``"cls"`` for the CLS token, ``"gap"`` for global average
            pooling of patch tokens.

    Returns:
        Dict keyed by patient ID, each containing:
            - "ed_features": Tensor (n_ed_slices, embed_dim)
            - "es_features": Tensor (n_es_slices, embed_dim)
    """
    patient_features: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: {"ed": [], "es": []}
    )

    for sample_idx in tqdm(range(len(cinema_dataset)), desc="Extracting CLS features"):
        sample = cinema_dataset[sample_idx]
        image_3d = sample["sax_image"]  # (1, H, W, z)
        n_slices = sample["n_slices"]
        pid = sample["pid"]
        is_ed = sample["is_ed"]
        frame_key = "ed" if is_ed else "es"

        for z in range(n_slices):
            image_2d = image_3d[0, :, :, z]  # (H, W) in [0, 1]
            img = preprocess_slice(image_2d)  # (1, 3, H, W)
            if device is not None:
                img = img.to(device)

            feats = backbone.get_intermediate_layers(
                img, n=1, return_class_token=True, norm=True
            )
            # feats is tuple of length 1: ((patch_tokens, cls_token),)
            if pooling == "cls":
                token = feats[0][1].squeeze(0).cpu()  # (embed_dim,)
            else:  # gap
                token = feats[0][0].squeeze(0).mean(dim=0).cpu()  # (embed_dim,)
            patient_features[pid][frame_key].append(token)

    # Stack per-patient features into tensors
    result: dict[str, dict] = {}
    for pid, frames in patient_features.items():
        result[pid] = {
            "ed_features": torch.stack(frames["ed"])
            if frames["ed"]
            else torch.empty(0),
            "es_features": torch.stack(frames["es"])
            if frames["es"]
            else torch.empty(0),
        }

    return result


# ── Feature Caching ────────────────────────────────────────────────────────────
@torch.inference_mode()
def cache_cls_features(
    backbone: nn.Module,
    cinema_dataset,
    cache_dir: Path,
    device: torch.device | None = None,
    pooling: str = "cls",
) -> list[dict]:
    """Extract and cache DINOv3 features to disk, one .pt file per slice.

    Saves ``{"cls_token": Tensor(embed_dim,)}`` per file.  Skips files that
    already exist so the function is safe to re-run.

    Args:
        backbone: Frozen DINOv3 backbone in eval mode.
        cinema_dataset: CineMA EndDiastoleEndSystoleDataset.
        cache_dir: Directory to save cached features.
        device: Device for inference.
        pooling: ``"cls"`` for the CLS token, ``"gap"`` for global average
            pooling of patch tokens.

    Returns:
        Manifest — list of dicts with keys: path, pid, is_ed, z_idx.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample_idx in tqdm(range(len(cinema_dataset)), desc="Caching CLS features"):
        sample = cinema_dataset[sample_idx]
        image_3d = sample["sax_image"]  # (1, H, W, z)
        n_slices = sample["n_slices"]
        pid = sample["pid"]
        is_ed = sample["is_ed"]
        frame = "ed" if is_ed else "es"

        for z in range(n_slices):
            fname = f"{pid}_{frame}_z{z:02d}.pt"
            fpath = cache_dir / fname

            if fpath.exists():
                manifest.append({"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z})
                continue

            image_2d = image_3d[0, :, :, z]  # (H, W) in [0, 1]
            img = preprocess_slice(image_2d)  # (1, 3, H, W)
            if device is not None:
                img = img.to(device)

            feats = backbone.get_intermediate_layers(
                img, n=1, return_class_token=True, norm=True
            )
            if pooling == "cls":
                token = feats[0][1].squeeze(0).cpu()  # (embed_dim,)
            else:  # gap
                token = feats[0][0].squeeze(0).mean(dim=0).cpu()  # (embed_dim,)

            torch.save({"cls_token": token}, fpath)
            manifest.append({"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z})

    return manifest


def load_cached_cls_features(
    manifest: list[dict],
) -> dict[str, dict]:
    """Load cached CLS tokens from disk and group by patient.

    Args:
        manifest: Output of :func:`cache_cls_features`.

    Returns:
        Dict keyed by patient ID, each containing:
            - "ed_features": Tensor (n_ed_slices, embed_dim)
            - "es_features": Tensor (n_es_slices, embed_dim)
        Same structure as :func:`extract_cls_features`.
    """
    patient_features: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: {"ed": [], "es": []}
    )

    for entry in manifest:
        data = torch.load(entry["path"], weights_only=True)
        frame_key = "ed" if entry["is_ed"] else "es"
        patient_features[entry["pid"]][frame_key].append(data["cls_token"])

    result: dict[str, dict] = {}
    for pid, frames in patient_features.items():
        result[pid] = {
            "ed_features": torch.stack(frames["ed"])
            if frames["ed"]
            else torch.empty(0),
            "es_features": torch.stack(frames["es"])
            if frames["es"]
            else torch.empty(0),
        }

    return result


# ── CineMA Feature Extraction ─────────────────────────────────────────────────
# CineMA's 3D ViT uses the full SAX depth as one patch dimension (patch_size
# [192, 192, 16]).  16 is the conservative upper bound across training datasets
# (MNMS 4-16, ACDC 5-11, RESCAN 8-18 slices) resampled to 10 mm z-spacing.
CINEMA_SAX_TARGET_DEPTH = 16


@torch.inference_mode()
def _extract_cinema_volume_token(
    backbone: nn.Module,
    sax_volume: torch.Tensor,
    device: torch.device | None = None,
    target_depth: int = CINEMA_SAX_TARGET_DEPTH,
    pooling: str = "cls",
) -> torch.Tensor:
    """Run CineMA on one SAX volume and return a single embedding.

    Steps:
        1. Pad/truncate the volume depth to ``target_depth`` (CineMA's expected
           SAX depth).
        2. Run ``backbone.feature_forward()`` → dict with ``"cls"`` and
           ``"sax"`` keys.
        3. Return the CLS token or GAP of spatial tokens depending on
           ``pooling``.

    Args:
        backbone: Frozen CineMA backbone in eval mode.
        sax_volume: ``(1, H, W, z)`` tensor in [0, 1].
        device: Device for inference.
        target_depth: Expected SAX depth for CineMA (default 16).
        pooling: ``"cls"`` for the CLS token, ``"gap"`` for global average
            pooling of spatial tokens.

    Returns:
        Single tensor of shape ``(embed_dim,)`` on CPU.
    """
    vol = sax_volume
    z = int(vol.shape[-1])

    if z > target_depth:
        vol = vol[..., :target_depth]
    elif z < target_depth:
        vol = F.pad(vol, (0, target_depth - z), mode="constant", value=0.0)

    batch = {"sax": vol.unsqueeze(0).to(device=device, dtype=torch.float32)}
    out = backbone.feature_forward(batch)

    if pooling == "cls":
        return out["cls"].squeeze(0).squeeze(0).cpu()  # (embed_dim,)
    else:  # gap
        return out["sax"].squeeze(0).mean(dim=0).cpu()  # (embed_dim,)


@torch.inference_mode()
def cache_cinema_cls_features(
    backbone: nn.Module,
    cinema_dataset,
    cache_dir: Path,
    device: torch.device | None = None,
    pooling: str = "cls",
) -> list[dict]:
    """Extract and cache CineMA embeddings, one .pt per frame.

    For each 3D SAX volume (ED or ES), runs the CineMA backbone once and
    extracts a single ``(embed_dim,)`` vector via CLS token or GAP
    (see :func:`_extract_cinema_volume_token`).  Saved as
    ``{"cls_token": Tensor(embed_dim,)}`` — the same format used by
    :func:`cache_cls_features` for DINOv3 — so :func:`load_cached_cls_features`
    and :func:`build_patient_features` work identically for all backbones.

    Args:
        backbone: Frozen CineMA backbone in eval mode.
        cinema_dataset: CineMA EndDiastoleEndSystoleDataset.
        cache_dir: Directory to save cached tokens.
        device: Device for inference.
        pooling: ``"cls"`` for the CLS token, ``"gap"`` for global average
            pooling of spatial tokens.

    Returns:
        Manifest — list of dicts with keys ``path``, ``pid``, ``is_ed``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample_idx in tqdm(range(len(cinema_dataset)), desc="Caching CineMA features"):
        sample = cinema_dataset[sample_idx]
        image_3d = sample["sax_image"]  # (1, H, W, z)
        pid = sample["pid"]
        is_ed = sample["is_ed"]
        frame = "ed" if is_ed else "es"

        fpath = cache_dir / f"{pid}_{frame}.pt"

        if not fpath.exists():
            token = _extract_cinema_volume_token(
                backbone, image_3d, device, pooling=pooling
            )
            torch.save({"cls_token": token}, fpath)

        manifest.append({"path": fpath, "pid": pid, "is_ed": is_ed})

    return manifest


@torch.inference_mode()
def cache_sam_cls_features(
    sam_model: nn.Module,
    image_processor,
    cinema_dataset,
    cache_dir: Path,
    device: torch.device | None = None,
) -> list[dict]:
    """Extract and cache global-average-pooled SAM vision-encoder embeddings, one .pt per slice.

    For each 2D slice, runs SAM's vision encoder (bypassing the neck) and
    global-average-pools the spatial feature map ``(h, w, C)`` → ``(C,)``
    vector, saved as ``{"cls_token": Tensor(C,)}`` so downstream functions
    (:func:`load_cached_cls_features`, :func:`build_patient_features`) work
    identically across all backbones.  ``C`` equals the encoder's
    ``hidden_size`` (768 / 1024 / 1280 for base / large / huge).

    Args:
        sam_model: Frozen ``SamModel`` in eval mode.
        image_processor: ``SamImageProcessor`` for pre-processing slices.
        cinema_dataset: CineMA EndDiastoleEndSystoleDataset.
        cache_dir: Directory to save cached tokens.
        device: Device for inference.

    Returns:
        Manifest — list of dicts with keys ``path``, ``pid``, ``is_ed``,
        ``z_idx``.
    """
    from PIL import Image

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample_idx in tqdm(range(len(cinema_dataset)), desc="Caching SAM features"):
        sample = cinema_dataset[sample_idx]
        image_3d = sample["sax_image"]  # (1, H, W, z)
        n_slices = int(sample["n_slices"])
        pid = sample["pid"]
        is_ed = sample["is_ed"]
        frame = "ed" if is_ed else "es"

        for z in range(n_slices):
            fname = f"{pid}_{frame}_z{z:02d}.pt"
            fpath = cache_dir / fname

            if fpath.exists():
                manifest.append({"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z})
                continue

            image_2d = image_3d[0, :, :, z]  # (H, W) in [0, 1]
            img_np = (image_2d.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            pil = Image.fromarray(img_np, mode="L").convert("RGB")

            proc = image_processor(images=pil, return_tensors="pt")
            pixel_values = proc["pixel_values"].to(device)

            ve = sam_model.vision_encoder
            hidden = ve.patch_embed(pixel_values)
            if ve.pos_embed is not None:
                hidden = hidden + ve.pos_embed
            for layer in ve.layers:
                hidden = layer(hidden)
            # hidden: (1, h, w, C) — before the neck projection
            cls_token = hidden.squeeze(0).mean(dim=(0, 1)).cpu()  # (C,)

            torch.save({"cls_token": cls_token}, fpath)
            manifest.append({"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z})

    return manifest


def build_patient_features(
    cls_features: dict[str, dict],
    pathology_map: dict[str, str],
    pathology_classes: dict[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Aggregate per-patient CLS features: mean-pool ED + mean-pool ES, concatenate.

    Args:
        cls_features: Output of extract_cls_features().
        pathology_map: {pid: pathology_string} from metadata.
        pathology_classes: {class_name: int_label} mapping. Defaults to ACDC classes.

    Returns:
        features: (N_patients, 2 * embed_dim) float32 tensor.
        labels: (N_patients,) integer tensor.
        pids: List of patient IDs in the same order.
    """
    if pathology_classes is None:
        pathology_classes = PATHOLOGY_CLASSES

    features_list = []
    labels_list = []
    pids = []

    for pid in sorted(cls_features.keys()):
        if pid not in pathology_map:
            logger.warning("Skipping %s: no pathology label found", pid)
            continue

        ed_feats = cls_features[pid]["ed_features"]  # (n_ed, dim)
        es_feats = cls_features[pid]["es_features"]  # (n_es, dim)

        ed_mean = ed_feats.mean(dim=0)  # (dim,)
        es_mean = es_feats.mean(dim=0)  # (dim,)
        patient_feat = torch.cat([ed_mean, es_mean])  # (2*dim,)

        pathology = pathology_map[pid]
        if pathology not in pathology_classes:
            logger.warning("Skipping %s: unknown pathology '%s'", pid, pathology)
            continue
        features_list.append(patient_feat)
        labels_list.append(pathology_classes[pathology])
        pids.append(pid)

    features = torch.stack(features_list).to(dtype=torch.float32)
    labels = torch.tensor(labels_list, dtype=torch.long)
    return features, labels, pids


# ── Metadata Helpers ───────────────────────────────────────────────────────────
def get_pathology_map(meta_df) -> dict[str, str]:
    """Extract {pid: pathology} mapping from an ACDC metadata DataFrame."""
    return dict(zip(meta_df["pid"], meta_df["pathology"], strict=False))


def validate_split_pathology_labels(
    train_pathology_map: dict[str, str],
    pathology_classes: dict[str, int] | None = None,
    val_pathology_map: dict[str, str] | None = None,
    test_pathology_map: dict[str, str] | None = None,
) -> dict[str, set[str]]:
    """Validate split label coverage and emit warnings for suspicious cases.

    This function reports train/val/test label coverage diagnostics. It warns
    when validation or test splits contain labels that are absent from training
    (or unknown to ``pathology_classes``), because this can invalidate metrics.

    Args:
        train_pathology_map: ``{pid: pathology}`` map for training split.
        pathology_classes: ``{class_name: int_label}`` mapping. Defaults to
            ACDC classes.
        val_pathology_map: Optional ``{pid: pathology}`` map for val split.
        test_pathology_map: Optional ``{pid: pathology}`` map for test split.

    Returns:
        Dict with set diagnostics for missing/extra labels in each split.
    """
    if pathology_classes is None:
        pathology_classes = PATHOLOGY_CLASSES

    expected = set(pathology_classes.keys())
    train_labels = set(train_pathology_map.values())

    train_missing = expected - train_labels
    train_unknown = train_labels - expected

    # Unknown labels in training are always suspicious and should be warned.
    # Missing expected labels in training can be legitimate in subset runs, so
    # they are reported in diagnostics without warning.
    if train_unknown:
        warnings.warn(
            "Training labels mismatch pathology_classes. "
            f"missing={sorted(train_missing)}, unknown={sorted(train_unknown)}",
            stacklevel=2,
        )

    diagnostics: dict[str, set[str]] = {
        "train_missing": train_missing,
        "train_unknown": train_unknown,
        "val_unknown": set(),
        "val_unseen_from_train": set(),
        "test_unknown": set(),
        "test_unseen_from_train": set(),
    }

    split_maps = {
        "val": val_pathology_map,
        "test": test_pathology_map,
    }
    for split_name, split_map in split_maps.items():
        if split_map is None:
            continue

        split_labels = set(split_map.values())
        unknown = split_labels - expected
        unseen_from_train = (split_labels - train_labels) & expected

        diagnostics[f"{split_name}_unknown"] = unknown
        diagnostics[f"{split_name}_unseen_from_train"] = unseen_from_train

        if unknown:
            warnings.warn(
                f"{split_name} split has labels not in pathology_classes: "
                f"{sorted(unknown)}",
                stacklevel=2,
            )
        if unseen_from_train:
            warnings.warn(
                f"{split_name} split has labels absent from training split: "
                f"{sorted(unseen_from_train)}",
                stacklevel=2,
            )

    return diagnostics


# ── Logistic Regression C-Sweep with Stratified K-Fold CV ─────────────────────
def sweep_C_and_train(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 10,
    max_iter: int = 1_000,
    tol: float = 1e-12,
    val_features: torch.Tensor | None = None,
    val_labels: torch.Tensor | None = None,
) -> tuple[float, Pipeline, list[dict]]:
    """Sweep regularisation strength C, then retrain on all training data.

    When ``val_features`` and ``val_labels`` are provided, each C is evaluated
    on the dedicated validation set (single split). Otherwise, stratified
    k-fold CV is used.

    Args:
        features: (N, D) float64 feature matrix (all training patients).
        labels: (N,) int label vector.
        n_folds: Number of stratified CV folds (ignored when val data provided).
        max_iter: Maximum iterations for L-BFGS.
        tol: Convergence tolerance.
        val_features: Optional (M, D) float64 validation feature matrix.
        val_labels: Optional (M,) int validation label vector.

    Returns:
        best_C: The regularisation strength with highest mean CV accuracy.
        pipeline: Pipeline(StandardScaler + LogisticRegression) fitted on all data.
        sweep_results: List of dicts with keys C, mean_cv_acc, std_cv_acc.
    """
    ALL_C = (10.0**C_POWER_RANGE).tolist()

    X = features.numpy()
    y = labels.numpy()

    use_val_split = val_features is not None and val_labels is not None
    if use_val_split:
        assert val_features is not None
        assert val_labels is not None
        X_val = val_features.numpy()
        y_val = val_labels.numpy()
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        folds = list(skf.split(X, y))

    best_mean_acc = -1.0
    best_C = ALL_C[0]
    sweep_results: list[dict] = []

    for C in tqdm(ALL_C, desc="C-sweep (CV)"):
        if use_val_split:
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            solver="lbfgs",
                            C=C,
                            max_iter=max_iter,
                            tol=tol,
                        ),
                    ),
                ]
            )
            pipe.fit(X, y)
            mean_acc = float(accuracy_score(y_val, pipe.predict(X_val)))
            std_acc = 0.0
        else:
            fold_accs = []
            for train_idx, val_idx in folds:
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                solver="lbfgs",
                                C=C,
                                max_iter=max_iter,
                                tol=tol,
                            ),
                        ),
                    ]
                )
                pipe.fit(X[train_idx], y[train_idx])
                fold_accs.append(accuracy_score(y[val_idx], pipe.predict(X[val_idx])))
            mean_acc = float(np.mean(fold_accs))
            std_acc = float(np.std(fold_accs))

        sweep_results.append({"C": C, "mean_cv_acc": mean_acc, "std_cv_acc": std_acc})

        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_C = C

    logger.info("Best C = %.4g (mean CV accuracy = %.4f)", best_C, best_mean_acc)

    # Retrain on all data with best C
    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    C=best_C,
                    max_iter=max_iter,
                    tol=tol,
                ),
            ),
        ]
    )
    final_pipeline.fit(X, y)

    return best_C, final_pipeline, sweep_results


def evaluate_classification(
    model: Pipeline | LogisticRegression,
    features: torch.Tensor,
    labels: torch.Tensor,
    pathology_classes: dict[str, int] | None = None,
) -> dict:
    """Evaluate a fitted classification pipeline or model.

    Args:
        model: A sklearn Pipeline (StandardScaler + LogisticRegression)
               or a bare LogisticRegression.
        pathology_classes: {class_name: int_label} mapping. Defaults to ACDC classes.

    Returns:
        Dict with keys: accuracy, macro_f1,
        per_class_sensitivity, per_class_specificity, macro_sensitivity,
        macro_specificity, confusion_matrix, classification_report,
        predictions, probabilities.
    """
    if pathology_classes is None:
        pathology_classes = PATHOLOGY_CLASSES
    num_classes = len(pathology_classes)

    X = features.numpy()
    y_true = labels.numpy()

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Per-class sensitivity (TPR) and specificity (TNR)
    per_class_sensitivity: dict[str, float] = {}
    per_class_specificity: dict[str, float] = {}
    total = cm.sum()

    for cls_name, cls_idx in pathology_classes.items():
        tp = cm[cls_idx, cls_idx]
        fn = cm[cls_idx, :].sum() - tp
        fp = cm[:, cls_idx].sum() - tp
        tn = total - tp - fn - fp

        per_class_sensitivity[cls_name] = (
            float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        )
        per_class_specificity[cls_name] = (
            float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        )

    macro_sensitivity = float(
        sum(per_class_sensitivity.values()) / len(per_class_sensitivity)
    )
    macro_specificity = float(
        sum(per_class_specificity.values()) / len(per_class_specificity)
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=list(pathology_classes.keys()),
        output_dict=True,
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "per_class_sensitivity": per_class_sensitivity,
        "per_class_specificity": per_class_specificity,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": y_pred,
        "probabilities": y_prob,
    }


# ── Binary Disease Detection ─────────────────────────────────────────────────
def binarize_labels(labels: torch.Tensor, nor_idx: int = 0) -> torch.Tensor:
    """Map multi-class labels to binary: NOR → 0, all disease → 1."""
    return (labels != nor_idx).long()


def evaluate_binary_detection(
    probabilities: np.ndarray,
    labels: torch.Tensor,
    nor_idx: int = 0,
) -> dict:
    """Evaluate binary disease detection from multi-class probabilities.

    Derives binary disease probability as ``1 - P(NOR)`` from the
    softmax output, so no retraining is needed.

    Args:
        probabilities: (N, C) array of class probabilities from the multi-class model.
        labels: (N,) tensor of integer labels.
        nor_idx: Column index for the NOR (normal) class. Defaults to 0.

    Returns:
        Dict with keys: accuracy, f1, sensitivity, specificity, roc_auc,
        binary_probs, binary_labels, binary_predictions.
    """
    y_binary = binarize_labels(labels, nor_idx=nor_idx).numpy()
    disease_prob = 1.0 - probabilities[:, nor_idx]
    y_pred = (disease_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_binary, y_pred))
    f1 = float(f1_score(y_binary, y_pred, average="binary"))
    auc = float(roc_auc_score(y_binary, disease_prob))

    # Sensitivity (TPR) and Specificity (TNR)
    tp = int(((y_pred == 1) & (y_binary == 1)).sum())
    fn = int(((y_pred == 0) & (y_binary == 1)).sum())
    fp = int(((y_pred == 1) & (y_binary == 0)).sum())
    tn = int(((y_pred == 0) & (y_binary == 0)).sum())

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": auc,
        "binary_probs": disease_prob,
        "binary_labels": y_binary,
        "binary_predictions": y_pred,
    }
