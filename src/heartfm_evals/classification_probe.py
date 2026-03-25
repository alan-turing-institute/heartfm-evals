"""Logistic-regression classification probe for patient-level pathology prediction.

Supports three frozen backbones with a shared downstream protocol:

    1. **DINOv3**: extract final-layer CLS token per 2D slice →
       one ``(embed_dim,)`` embedding per slice.
    2. **CineMA**: run ``feature_forward()`` on the 3D SAX volume and
       global-mean-pool all ``(gx*gy*gz)`` spatial tokens → one
       ``(embed_dim,)`` embedding per cardiac-phase volume.

    3. **SAM**: run ``get_image_embeddings()`` on each 2D slice (converted to
       RGB via ``SamImageProcessor``), global-average-pool the spatial feature
       map ``(C, h, w)`` → one ``(C,)`` embedding per slice.

DINOv3 and SAM produce one embedding per 2D slice; CineMA produces one
embedding per 3D volume.  Patient-level features are built identically:
mean-pool ED embeddings, mean-pool ES embeddings, concatenate →
``(2 × embed_dim,)`` vector → sklearn LogisticRegression with L2 penalty
and C-sweep via stratified CV.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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

import torch.nn.functional as F

from heartfm_evals.dense_linear_probe import preprocess_slice

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
PATHOLOGY_CLASSES = {"NOR": 0, "DCM": 1, "HCM": 2, "MINF": 3, "RV": 4}
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
) -> dict[str, dict]:
    """Extract final-layer CLS token features from all slices in a CineMA dataset.

    Groups features by patient ID, keeping ED and ES frames separate.

    Args:
        backbone: Frozen DINOv3 backbone in eval mode.
        cinema_dataset: CineMA EndDiastoleEndSystoleDataset.
        device: Device for inference.

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

            # Extract final-layer CLS token
            feats = backbone.get_intermediate_layers(
                img, n=1, return_class_token=True, norm=True
            )
            # feats is tuple of length 1: ((patch_tokens, cls_token),)
            cls_token = feats[0][1].squeeze(0).cpu()  # (embed_dim,)
            patient_features[pid][frame_key].append(cls_token)

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
) -> list[dict]:
    """Extract and cache final-layer CLS tokens to disk, one .pt file per slice.

    Saves ``{"cls_token": Tensor(embed_dim,)}`` per file.  Skips files that
    already exist so the function is safe to re-run.

    Args:
        backbone: Frozen DINOv3 backbone in eval mode.
        cinema_dataset: CineMA EndDiastoleEndSystoleDataset.
        cache_dir: Directory to save cached CLS tokens.
        device: Device for inference.

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
            cls_token = feats[0][1].squeeze(0).cpu()  # (embed_dim,)

            torch.save({"cls_token": cls_token}, fpath)
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
CINEMA_SAX_TARGET_DEPTH = 16


@torch.inference_mode()
def _extract_cinema_volume_token(
    backbone: nn.Module,
    sax_volume: torch.Tensor,
    device: torch.device | None = None,
    target_depth: int = CINEMA_SAX_TARGET_DEPTH,
) -> torch.Tensor:
    """Run CineMA on one SAX volume and return a single global-mean-pooled token.

    Following the CineMA paper, we mean-pool **all** spatial tokens from the 3D
    token grid into a single ``(embed_dim,)`` vector per cardiac-phase volume.

    Steps:
        1. Pad/truncate the volume depth to ``target_depth`` (CineMA's expected
           SAX depth).
        2. Run ``backbone.feature_forward()`` → ``(1, gx*gy*gz, C)`` tokens.
        3. Mean-pool across the token dimension → ``(C,)`` embedding.

    Args:
        backbone: Frozen CineMA backbone in eval mode.
        sax_volume: ``(1, H, W, z)`` tensor in [0, 1].
        device: Device for inference.
        target_depth: Expected SAX depth for CineMA (default 16).

    Returns:
        Single tensor of shape ``(embed_dim,)`` on CPU.
    """
    vol = sax_volume
    z = int(vol.shape[-1])

    if z > target_depth:
        vol = vol[..., :target_depth]
    elif z < target_depth:
        vol = F.pad(vol, (0, target_depth - z), mode="constant", value=0.0)

    # Forward pass — get all spatial tokens
    batch = {"sax": vol.unsqueeze(0).to(device=device, dtype=torch.float32)}
    tokens = backbone.feature_forward(batch)["sax"]  # (1, n_tokens, C)

    # Global mean-pool across all spatial tokens
    return tokens.squeeze(0).mean(dim=0).cpu()  # (C,)


@torch.inference_mode()
def cache_cinema_cls_features(
    backbone: nn.Module,
    cinema_dataset,
    cache_dir: Path,
    device: torch.device | None = None,
) -> list[dict]:
    """Extract and cache global-mean-pooled CineMA embeddings, one .pt per frame.

    For each 3D SAX volume (ED or ES), runs the CineMA backbone once and
    global-mean-pools all spatial tokens into a single ``(embed_dim,)`` vector
    (see :func:`_extract_cinema_volume_token`).  Saved as
    ``{"cls_token": Tensor(embed_dim,)}`` — the same format used by
    :func:`cache_cls_features` for DINOv3 — so :func:`load_cached_cls_features`
    and :func:`build_patient_features` work identically for all backbones.

    Args:
        backbone: Frozen CineMA backbone in eval mode.
        cinema_dataset: CineMA EndDiastoleEndSystoleDataset.
        cache_dir: Directory to save cached tokens.
        device: Device for inference.

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
            token = _extract_cinema_volume_token(backbone, image_3d, device)
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
    """Extract and cache global-average-pooled SAM embeddings, one .pt per slice.

    For each 2D slice, runs SAM's image encoder and global-average-pools the
    spatial feature map ``(C, h, w)`` → ``(C,)`` vector, saved as
    ``{"cls_token": Tensor(C,)}`` so downstream functions
    (:func:`load_cached_cls_features`, :func:`build_patient_features`) work
    identically across all backbones.

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

            feats = sam_model.get_image_embeddings(pixel_values)  # (1, C, h, w)
            cls_token = feats.squeeze(0).mean(dim=(1, 2)).cpu()   # (C,)

            torch.save({"cls_token": cls_token}, fpath)
            manifest.append({"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z})

    return manifest


def build_patient_features(
    cls_features: dict[str, dict],
    pathology_map: dict[str, str],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Aggregate per-patient CLS features: mean-pool ED + mean-pool ES, concatenate.

    Args:
        cls_features: Output of extract_cls_features().
        pathology_map: {pid: pathology_string} from metadata.

    Returns:
        features: (N_patients, 2 * embed_dim) float64 tensor.
        labels: (N_patients,) integer tensor.
        pids: List of patient IDs in the same order.
    """
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

        features_list.append(patient_feat)
        labels_list.append(PATHOLOGY_CLASSES[pathology_map[pid]])
        pids.append(pid)

    features = torch.stack(features_list).to(dtype=torch.float64)
    labels = torch.tensor(labels_list, dtype=torch.long)
    return features, labels, pids


# ── Metadata Helpers ───────────────────────────────────────────────────────────
def get_pathology_map(meta_df) -> dict[str, str]:
    """Extract {pid: pathology} mapping from an ACDC metadata DataFrame."""
    return dict(zip(meta_df["pid"], meta_df["pathology"], strict=False))


# ── Logistic Regression C-Sweep with Stratified K-Fold CV ─────────────────────
def sweep_C_and_train(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 10,
    max_iter: int = 1_000,
    tol: float = 1e-12,
) -> tuple[float, Pipeline, list[dict]]:
    """Sweep regularisation strength C using stratified k-fold CV, retrain on all data.

    Each fold fits a StandardScaler on the training portion and applies it to
    the held-out portion, so feature normalisation never leaks test information.

    Args:
        features: (N, D) float64 feature matrix (all training patients).
        labels: (N,) int label vector.
        n_folds: Number of stratified CV folds.
        max_iter: Maximum iterations for L-BFGS.
        tol: Convergence tolerance.

    Returns:
        best_C: The regularisation strength with highest mean CV accuracy.
        pipeline: Pipeline(StandardScaler + LogisticRegression) fitted on all data.
        sweep_results: List of dicts with keys C, mean_cv_acc, std_cv_acc.
    """
    ALL_C = (10.0**C_POWER_RANGE).tolist()

    X = features.numpy()
    y = labels.numpy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    folds = list(skf.split(X, y))

    best_mean_acc = -1.0
    best_C = ALL_C[0]
    sweep_results: list[dict] = []

    for C in tqdm(ALL_C, desc="C-sweep (CV)"):
        fold_accs = []
        for train_idx, val_idx in folds:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    solver="lbfgs", C=C,
                    max_iter=max_iter, tol=tol,
                )),
            ])
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
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs", C=best_C,
            max_iter=max_iter, tol=tol,
        )),
    ])
    final_pipeline.fit(X, y)

    return best_C, final_pipeline, sweep_results


def evaluate_classification(
    model: Pipeline | LogisticRegression,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """Evaluate a fitted classification pipeline or model.

    Args:
        model: A sklearn Pipeline (StandardScaler + LogisticRegression)
               or a bare LogisticRegression.

    Returns:
        Dict with keys: accuracy, macro_f1,
        per_class_sensitivity, per_class_specificity, macro_sensitivity,
        macro_specificity, confusion_matrix, classification_report,
        predictions, probabilities.
    """
    X = features.numpy()
    y_true = labels.numpy()

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_PATHOLOGIES)))

    # Per-class sensitivity (TPR) and specificity (TNR)
    per_class_sensitivity: dict[str, float] = {}
    per_class_specificity: dict[str, float] = {}
    total = cm.sum()

    for cls_name, cls_idx in PATHOLOGY_CLASSES.items():
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
        labels=list(range(NUM_PATHOLOGIES)),
        target_names=list(PATHOLOGY_CLASSES.keys()),
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
def binarize_labels(labels: torch.Tensor) -> torch.Tensor:
    """Map 5-way labels to binary: NOR (0) → 0, all disease → 1."""
    return (labels != PATHOLOGY_CLASSES["NOR"]).long()


def evaluate_binary_detection(
    probabilities: np.ndarray,
    labels: torch.Tensor,
) -> dict:
    """Evaluate binary disease detection from 5-way class probabilities.

    Derives binary disease probability as ``1 - P(NOR)`` from the 5-way
    softmax output, so no retraining is needed.

    Args:
        probabilities: (N, 5) array of class probabilities from the 5-way model.
        labels: (N,) tensor of 5-way integer labels.

    Returns:
        Dict with keys: accuracy, f1, sensitivity, specificity, roc_auc,
        binary_probs, binary_labels, binary_predictions.
    """
    nor_idx = PATHOLOGY_CLASSES["NOR"]
    y_binary = binarize_labels(labels).numpy()
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
