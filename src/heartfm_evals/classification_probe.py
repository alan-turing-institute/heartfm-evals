"""Logistic-regression classification probe for patient-level pathology prediction.

Adapts the DINOv3 paper's Fine-S logistic regression evaluation protocol
(sklearn LogisticRegression with L-BFGS solver) to classify ACDC cardiac MRI
patients into 5 pathology classes using frozen CLS token features.

Protocol:
    frozen DINOv3 backbone → extract final-layer CLS token per 2D slice →
    mean-pool ED slices + mean-pool ES slices per patient → concatenate →
    sklearn LogisticRegression with L2 penalty and C-sweep on validation set.
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
)
from tqdm import tqdm

from heartfm_evals.dense_linear_probe import preprocess_slice

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
PATHOLOGY_CLASSES = {"NOR": 0, "DCM": 1, "HCM": 2, "MINF": 3, "RV": 4}
PATHOLOGY_NAMES = {v: k for k, v in PATHOLOGY_CLASSES.items()}
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

    for sample_idx in tqdm(
        range(len(cinema_dataset)), desc="Caching CLS features"
    ):
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
                manifest.append(
                    {"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z}
                )
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
            manifest.append(
                {"path": fpath, "pid": pid, "is_ed": is_ed, "z_idx": z}
            )

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


# ── Logistic Regression C-Sweep ────────────────────────────────────────────────
def sweep_C_and_train(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    max_iter: int = 1_000,
    tol: float = 1e-12,
) -> tuple[float, LogisticRegression, list[dict]]:
    """Sweep regularisation strength C, pick best on val, retrain on train+val.

    Follows the DINOv3 Fine-S protocol: sklearn LogisticRegression with L-BFGS
    solver, L2 penalty, sweeping C = 10^k for k in linspace(-6, 5, 45).

    Args:
        train_features: (N_train, D) float64.
        train_labels: (N_train,) int.
        val_features: (N_val, D) float64.
        val_labels: (N_val,) int.
        max_iter: Maximum iterations for L-BFGS.
        tol: Convergence tolerance.

    Returns:
        best_C: The regularisation strength with highest val accuracy.
        model: LogisticRegression fitted on train+val with best_C.
    """
    ALL_C = (10.0**C_POWER_RANGE).tolist()

    train_X = train_features.numpy()
    train_y = train_labels.numpy()
    val_X = val_features.numpy()
    val_y = val_labels.numpy()

    best_acc = -1.0
    best_C = ALL_C[0]
    sweep_results: list[dict] = []

    for C in tqdm(ALL_C, desc="C-sweep"):
        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=C,
            max_iter=max_iter,
            tol=tol,
        )
        clf.fit(train_X, train_y)
        val_acc = accuracy_score(val_y, clf.predict(val_X))
        sweep_results.append({"C": C, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            best_C = C

    logger.info("Best C = %.4g (val accuracy = %.4f)", best_C, best_acc)

    # Retrain on train + val with best C
    combined_X = np.concatenate([train_X, val_X])
    combined_y = np.concatenate([train_y, val_y])

    final_model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=best_C,
        max_iter=max_iter,
        tol=tol,
    )
    final_model.fit(combined_X, combined_y)

    return best_C, final_model, sweep_results


def evaluate_classification(
    model: LogisticRegression,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """Evaluate a fitted logistic regression model.

    Returns:
        Dict with keys: accuracy, per_class_accuracy, confusion_matrix,
        classification_report, predictions, probabilities.
    """
    X = features.numpy()
    y_true = labels.numpy()

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    acc = accuracy_score(y_true, y_pred)

    # Per-class accuracy
    per_class_acc = {}
    for cls_name, cls_idx in PATHOLOGY_CLASSES.items():
        mask = y_true == cls_idx
        if mask.sum() > 0:
            per_class_acc[cls_name] = accuracy_score(y_true[mask], y_pred[mask])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_PATHOLOGIES)))
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
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": y_pred,
        "probabilities": y_prob,
    }
